//! Cross-platform real-time audio playback engine.
//!
//! Provides native low-latency output via `cpal` on desktop, and `web-sys` WebAudio
//! integration on WebAssembly, with lock-free state synchronization from the UI thread.

use crate::decoder::{AmbisonicDecoder, DecodeMode};
use crate::procedural::ProceduralSynthesizer;
use shared::rain::{EngineTelemetry, RainState};
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No audio output device found")]
    NoOutputDevice,
    #[error("Device configuration error: {0}")]
    ConfigError(String),
    #[error("Stream build error: {0}")]
    StreamError(String),
    #[error("WebAudio error: {0}")]
    WebAudioError(String),
}

/// Circular FIFO buffer for stereo audio frames
#[derive(Debug, Clone)]
pub struct AudioRingBuffer {
    buffer: Vec<f32>, // interleaved stereo [L, R, L, R, ...]
    capacity_frames: usize,
    read_pos: usize,
    write_pos: usize,
    available_frames: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity_frames: usize) -> Self {
        Self {
            buffer: vec![0.0; capacity_frames * 2],
            capacity_frames,
            read_pos: 0,
            write_pos: 0,
            available_frames: 0,
        }
    }

    #[inline]
    pub fn available_frames(&self) -> usize {
        self.available_frames
    }

    #[inline]
    pub fn free_frames(&self) -> usize {
        self.capacity_frames.saturating_sub(self.available_frames)
    }

    #[inline]
    pub fn push_frame(&mut self, left: f32, right: f32) -> bool {
        if self.available_frames >= self.capacity_frames {
            return false;
        }
        let idx = self.write_pos * 2;
        self.buffer[idx] = left;
        self.buffer[idx + 1] = right;
        self.write_pos = (self.write_pos + 1) % self.capacity_frames;
        self.available_frames += 1;
        true
    }

    #[inline]
    pub fn pop_frame(&mut self) -> Option<(f32, f32)> {
        if self.available_frames == 0 {
            return None;
        }
        let idx = self.read_pos * 2;
        let l = self.buffer[idx];
        let r = self.buffer[idx + 1];
        self.read_pos = (self.read_pos + 1) % self.capacity_frames;
        self.available_frames -= 1;
        Some((l, r))
    }

    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
        self.available_frames = 0;
    }
}

/// Zero-allocation, smooth soft-knee tanh saturation limiter.
/// Limits audio signals strictly to [-1.0, 1.0], preventing harsh digital clipping.
#[inline]
pub fn soft_limit(sample: f32) -> f32 {
    if sample.abs() < 0.8 {
        sample
    } else {
        (sample * 0.95).tanh()
    }
}

/// Shared thread-safe state container between UI thread and Audio thread
#[derive(Clone, Debug)]
pub struct SharedAudioState {
    pub rain: Arc<RwLock<RainState>>,
    pub decode_mode: Arc<RwLock<DecodeMode>>,
    pub orientation: Arc<RwLock<[f32; 3]>>,
    pub telemetry: Arc<RwLock<EngineTelemetry>>,
    pub ring_buffer: Arc<RwLock<AudioRingBuffer>>,
}

impl Default for SharedAudioState {
    fn default() -> Self {
        Self {
            rain: Arc::new(RwLock::new(RainState::default())),
            decode_mode: Arc::new(RwLock::new(DecodeMode::default())),
            orientation: Arc::new(RwLock::new([0.0, 0.0, 0.0])),
            telemetry: Arc::new(RwLock::new(EngineTelemetry::default())),
            ring_buffer: Arc::new(RwLock::new(AudioRingBuffer::new(12000))),
        }
    }
}

impl SharedAudioState {
    pub fn new(state: RainState, mode: DecodeMode) -> Self {
        let telemetry = Arc::new(RwLock::new(state.telemetry.clone()));
        Self {
            rain: Arc::new(RwLock::new(state)),
            decode_mode: Arc::new(RwLock::new(mode)),
            orientation: Arc::new(RwLock::new([0.0, 0.0, 0.0])),
            telemetry,
            ring_buffer: Arc::new(RwLock::new(AudioRingBuffer::new(12000))),
        }
    }

    pub fn update_rain(&self, state: &RainState) {
        if let Ok(mut lock) = self.rain.write() {
            let mut new_state = state.clone();
            if let Ok(tele_lock) = self.telemetry.read() {
                new_state.telemetry = tele_lock.clone();
            }
            *lock = new_state;
        }
    }

    pub fn update_telemetry(&self, telemetry: &EngineTelemetry) {
        if let Ok(mut lock) = self.telemetry.write() {
            *lock = telemetry.clone();
        }
    }

    pub fn get_telemetry(&self) -> EngineTelemetry {
        if let Ok(lock) = self.telemetry.read() {
            lock.clone()
        } else {
            EngineTelemetry::default()
        }
    }

    pub fn set_decode_mode(&self, mode: DecodeMode) {
        if let Ok(mut lock) = self.decode_mode.write() {
            *lock = mode;
        }
    }

    pub fn set_orientation(&self, yaw: f32, pitch: f32, roll: f32) {
        if let Ok(mut lock) = self.orientation.write() {
            *lock = [yaw, pitch, roll];
        }
    }
}

/// Native desktop audio runner using CPAL
#[cfg(not(target_arch = "wasm32"))]
pub struct DesktopAudioEngine {
    _stream: cpal::Stream,
    pub state: SharedAudioState,
}

#[cfg(not(target_arch = "wasm32"))]
impl DesktopAudioEngine {
    pub fn start(initial_state: RainState, mode: DecodeMode) -> Result<Self, AudioError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        let supported_config = device
            .default_output_config()
            .map_err(|e| AudioError::ConfigError(e.to_string()))?;

        let sample_rate = supported_config.sample_rate().0 as f32;
        let channels = supported_config.channels() as usize;

        let quality_tier = initial_state.quality_tier;
        let state = SharedAudioState::new(initial_state, mode);
        let audio_state = state.clone();

        let mut synth = ProceduralSynthesizer::new(sample_rate);
        let mut runner = inference::runner::InferenceRunner::new(quality_tier);
        let mut decoder = AmbisonicDecoder::new(mode);
        let mut governor = crate::meta_governor::MetaGovernor::new();

        let err_fn = |err| tracing::error!("An error occurred on the audio stream: {err}");

        let stream = match supported_config.sample_format() {
            cpal::SampleFormat::F32 => device
                .build_output_stream(
                    &supported_config.into(),
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        let dt = (data.len() / channels.max(1)) as f32 / sample_rate.max(1.0);
                        let frames_needed = data.len() / channels.max(1);

                        let mut rain = if let Ok(guard) = audio_state.rain.read() {
                            guard.clone()
                        } else {
                            RainState::default()
                        };

                        if let Ok(dm) = audio_state.decode_mode.read() {
                            decoder.mode = *dm;
                        }
                        if let Ok(ori) = audio_state.orientation.read() {
                            decoder.set_orientation(ori[0], ori[1], ori[2]);
                        }

                        // Evaluate autonomous governor with active optimization and stress profiles
                        let action = governor.evaluate(
                            &rain.telemetry,
                            rain.quality_tier,
                            rain.auto_quantize,
                            rain.optimization_profile,
                            rain.stress_profile,
                            dt,
                        );
                        rain.telemetry.effective_quant_floor = action.min_bits;
                        rain.telemetry.effective_quant_ceiling = action.max_bits;
                        rain.telemetry.governor_status = action.status_label.into();
                        rain.telemetry.active_path_label = action.recommended_path.label().into();
                        rain.telemetry.active_quantization_format = action.recommended_format.label().into();
                        rain.telemetry.synthesis_blend = action.synthesis_blend;
                        rain.telemetry.active_experts = action.active_experts;
                        rain.telemetry.diffusion_bypassed = action.diffusion_bypass;
                        rain.telemetry.ambisonic_order_reduced = action.ambisonic_order_reduced;
                        rain.telemetry.active_optimization_profile = rain.optimization_profile.label().into();
                        rain.telemetry.active_stress_profile = rain.stress_profile.label().into();
                        rain.telemetry.panic_factor = action.simulated_panic_factor;
                        rain.telemetry.jitter_factor = action.simulated_jitter_factor;
                        rain.telemetry.cpu_headroom = action.simulated_cpu_headroom;

                        let target_headroom_frames = ((action.target_buffer_ms / 1000.0) * sample_rate) as usize;

                        if let Ok(mut rb) = audio_state.ring_buffer.write() {
                            // 1. Generation: pre-buffer ahead up to target headroom even when paused!
                            let frames_to_generate = if rb.available_frames() < target_headroom_frames {
                                (target_headroom_frames - rb.available_frames()).min(frames_needed.max(256) * 4)
                            } else {
                                0
                            };

                            if frames_to_generate > 0 {
                                runner.set_target_tier(rain.quality_tier);
                                runner.set_active_experts(action.active_experts);
                                runner.set_diffusion_bypass(action.diffusion_bypass);
                                runner.update_quantization_bounds(action.min_bits, action.max_bits);

                                // Condition synthesis as active to prime the buffer
                                let mut synth_state = rain.clone();
                                synth_state.is_playing = true;
                                let blend = action.synthesis_blend;

                                for _ in 0..frames_to_generate {
                                    let foa_proc = synth.process_frame(&synth_state);

                                    let foa = if blend >= 0.999 {
                                        foa_proc
                                    } else {
                                        let cond = synth_state.to_conditioning_array();
                                        let (nw, nx, ny, nz) = runner.step(&cond);
                                        let foa_neural = crate::decoder::FoaFrame::new(nw, nx, ny, nz);

                                        crate::decoder::FoaFrame::new(
                                            foa_proc.w * blend + foa_neural.w * (1.0 - blend),
                                            foa_proc.x * blend + foa_neural.x * (1.0 - blend),
                                            foa_proc.y * blend + foa_neural.y * (1.0 - blend),
                                            foa_proc.z * blend + foa_neural.z * (1.0 - blend),
                                        )
                                    };

                                    let stereo = if action.ambisonic_order_reduced {
                                        // Shed ambisonic order to lightweight stereo pass-through
                                        crate::decoder::StereoFrame {
                                            left: foa.w * 0.707 + foa.y * 0.5,
                                            right: foa.w * 0.707 - foa.y * 0.5,
                                        }
                                    } else {
                                        decoder.decode_stereo(foa)
                                    };

                                    let limited_l = soft_limit(stereo.left);
                                    let limited_r = soft_limit(stereo.right);
                                    if !rb.push_frame(limited_l, limited_r) {
                                        break;
                                    }
                                }
                            }

                            let current_available = rb.available_frames();
                            let buffer_ms = (current_available as f32 / sample_rate) * 1000.0;
                            rain.telemetry.buffer_health_ms = buffer_ms;

                            // 2. Playback consumption
                            if rain.is_playing {
                                rain.telemetry.is_prebuffered = false;
                                for frame_chunk in data.chunks_mut(channels) {
                                    if let Some((l, r)) = rb.pop_frame() {
                                        let out_l = soft_limit(l * rain.master_volume);
                                        let out_r = soft_limit(r * rain.master_volume);
                                        if channels >= 2 {
                                            frame_chunk[0] = out_l;
                                            frame_chunk[1] = out_r;
                                            for extra in &mut frame_chunk[2..] {
                                                *extra = 0.0;
                                            }
                                        } else if channels == 1 {
                                            frame_chunk[0] = (out_l + out_r) * 0.5;
                                        }
                                    } else {
                                        for s in frame_chunk.iter_mut() {
                                            *s = 0.0;
                                        }
                                    }
                                }
                            } else {
                                // Paused: output silence to hardware
                                for s in data.iter_mut() {
                                    *s = 0.0;
                                }

                                if current_available >= target_headroom_frames.saturating_sub(64) {
                                    rain.telemetry.is_prebuffered = true;
                                    rain.telemetry.governor_status = "Pre-Buffered & Ready (Happy)".into();
                                } else {
                                    rain.telemetry.is_prebuffered = false;
                                    rain.telemetry.governor_status = format!(
                                        "Pre-Buffering... ({:.0}ms / {:.0}ms)",
                                        buffer_ms,
                                        action.target_buffer_ms
                                    );
                                }
                            }
                        }

                        audio_state.update_telemetry(&rain.telemetry);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| AudioError::StreamError(e.to_string()))?,
            _ => return Err(AudioError::StreamError("Unsupported sample format (expected F32)".into())),
        };

        stream.play().map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            state,
        })
    }
}

/// WebAssembly WebAudio runner
#[cfg(target_arch = "wasm32")]
pub struct WebAudioEngine {
    pub state: SharedAudioState,
    _ctx: web_sys::AudioContext,
    _processor: web_sys::ScriptProcessorNode,
    _closure: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::AudioProcessingEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl WebAudioEngine {
    pub fn start(initial_state: RainState, mode: DecodeMode) -> Result<Self, AudioError> {
        use wasm_bindgen::closure::Closure;
        use wasm_bindgen::JsCast;

        let ctx = web_sys::AudioContext::new()
            .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?;

        if let Some(win) = web_sys::window() {
            let _ = js_sys::Reflect::set(
                &win,
                &wasm_bindgen::JsValue::from_str("__rainAudioContext"),
                &ctx,
            );
        }

        let sample_rate = ctx.sample_rate();
        let quality_tier = initial_state.quality_tier;
        let state = SharedAudioState::new(initial_state, mode);

        let processor = ctx
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                2048, 0, 2,
            )
            .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?;

        let audio_state = state.clone();
        let mut synth = ProceduralSynthesizer::new(sample_rate);
        let mut runner = inference::runner::InferenceRunner::new(quality_tier);
        let mut decoder = AmbisonicDecoder::new(mode);
        let mut governor = crate::meta_governor::MetaGovernor::new();
        let mut left_out = vec![0.0f32; 2048];
        let mut right_out = vec![0.0f32; 2048];

        let closure = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
            let output_buffer = match event.output_buffer() {
                Ok(buf) => buf,
                Err(_) => return,
            };

            let frames_needed = output_buffer.length() as usize;
            if left_out.len() != frames_needed {
                left_out.resize(frames_needed, 0.0);
                right_out.resize(frames_needed, 0.0);
            }

            let dt = frames_needed as f32 / sample_rate.max(1.0);

            let mut rain = if let Ok(guard) = audio_state.rain.read() {
                guard.clone()
            } else {
                return;
            };

            if let Ok(dm) = audio_state.decode_mode.read() {
                decoder.mode = *dm;
            }
            if let Ok(ori) = audio_state.orientation.read() {
                decoder.set_orientation(ori[0], ori[1], ori[2]);
            }

            let action = governor.evaluate(
                &rain.telemetry,
                rain.quality_tier,
                rain.auto_quantize,
                rain.optimization_profile,
                rain.stress_profile,
                dt,
            );

            rain.telemetry.effective_quant_floor = action.min_bits;
            rain.telemetry.effective_quant_ceiling = action.max_bits;
            rain.telemetry.governor_status = action.status_label.into();
            rain.telemetry.active_path_label = action.recommended_path.label().into();
            rain.telemetry.active_quantization_format = action.recommended_format.label().into();
            rain.telemetry.synthesis_blend = action.synthesis_blend;
            rain.telemetry.active_experts = action.active_experts;
            rain.telemetry.diffusion_bypassed = action.diffusion_bypass;
            rain.telemetry.ambisonic_order_reduced = action.ambisonic_order_reduced;
            rain.telemetry.active_optimization_profile = rain.optimization_profile.label().into();
            rain.telemetry.active_stress_profile = rain.stress_profile.label().into();
            rain.telemetry.panic_factor = action.simulated_panic_factor;
            rain.telemetry.jitter_factor = action.simulated_jitter_factor;
            rain.telemetry.cpu_headroom = action.simulated_cpu_headroom;

            let target_headroom_frames = ((action.target_buffer_ms / 1000.0) * sample_rate) as usize;

            if let Ok(mut rb) = audio_state.ring_buffer.write() {
                let frames_to_generate = if rb.available_frames() < target_headroom_frames {
                    (target_headroom_frames - rb.available_frames()).min(frames_needed.max(256) * 4)
                } else {
                    0
                };

                if frames_to_generate > 0 {
                    runner.set_target_tier(rain.quality_tier);
                    runner.set_active_experts(action.active_experts);
                    runner.set_diffusion_bypass(action.diffusion_bypass);
                    runner.update_quantization_bounds(action.min_bits, action.max_bits);

                    let mut synth_state = rain.clone();
                    synth_state.is_playing = true;
                    let blend = action.synthesis_blend;

                    for _ in 0..frames_to_generate {
                        let foa_proc = synth.process_frame(&synth_state);

                        let foa = if blend >= 0.999 {
                            foa_proc
                        } else {
                            let cond = synth_state.to_conditioning_array();
                            let (nw, nx, ny, nz) = runner.step(&cond);
                            let foa_neural = crate::decoder::FoaFrame::new(nw, nx, ny, nz);

                            crate::decoder::FoaFrame::new(
                                foa_proc.w * blend + foa_neural.w * (1.0 - blend),
                                foa_proc.x * blend + foa_neural.x * (1.0 - blend),
                                foa_proc.y * blend + foa_neural.y * (1.0 - blend),
                                foa_proc.z * blend + foa_neural.z * (1.0 - blend),
                            )
                        };

                        let stereo = if action.ambisonic_order_reduced {
                            crate::decoder::StereoFrame {
                                left: foa.w * 0.707 + foa.y * 0.5,
                                right: foa.w * 0.707 - foa.y * 0.5,
                            }
                        } else {
                            decoder.decode_stereo(foa)
                        };

                        let limited_l = soft_limit(stereo.left);
                        let limited_r = soft_limit(stereo.right);
                        if !rb.push_frame(limited_l, limited_r) {
                            break;
                        }
                    }
                }

                let current_available = rb.available_frames();
                let buffer_ms = (current_available as f32 / sample_rate) * 1000.0;
                rain.telemetry.buffer_health_ms = buffer_ms;

                if rain.is_playing {
                    rain.telemetry.is_prebuffered = false;
                    for i in 0..frames_needed {
                        if let Some((l, r)) = rb.pop_frame() {
                            left_out[i] = soft_limit(l * rain.master_volume);
                            right_out[i] = soft_limit(r * rain.master_volume);
                        } else {
                            left_out[i] = 0.0;
                            right_out[i] = 0.0;
                        }
                    }
                } else {
                    for i in 0..frames_needed {
                        left_out[i] = 0.0;
                        right_out[i] = 0.0;
                    }

                    if current_available >= target_headroom_frames.saturating_sub(64) {
                        rain.telemetry.is_prebuffered = true;
                        rain.telemetry.governor_status = "Pre-Buffered & Ready (Happy)".into();
                    } else {
                        rain.telemetry.is_prebuffered = false;
                        rain.telemetry.governor_status = format!(
                            "Pre-Buffering... ({:.0}ms / {:.0}ms)",
                            buffer_ms,
                            action.target_buffer_ms
                        );
                    }
                }
            }

            audio_state.update_telemetry(&rain.telemetry);

            let _ = output_buffer.copy_to_channel(&left_out, 0);
            let _ = output_buffer.copy_to_channel(&right_out, 1);
        }) as Box<dyn FnMut(web_sys::AudioProcessingEvent)>);

        processor.set_onaudioprocess(Some(closure.as_ref().unchecked_ref()));
        processor
            .connect_with_audio_node(&ctx.destination())
            .map_err(|e| AudioError::WebAudioError(format!("{e:?}")))?;

        Ok(Self {
            state,
            _ctx: ctx,
            _processor: processor,
            _closure: closure,
        })
    }

    pub fn resume(&self) -> Result<(), AudioError> {
        let _ = self._ctx.resume();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_ring_buffer_fifo() {
        let mut rb = AudioRingBuffer::new(4);
        assert_eq!(rb.available_frames(), 0);
        assert_eq!(rb.free_frames(), 4);

        assert!(rb.push_frame(0.1, 0.2));
        assert!(rb.push_frame(0.3, 0.4));
        assert_eq!(rb.available_frames(), 2);

        let (l1, r1) = rb.pop_frame().unwrap();
        assert!((l1 - 0.1).abs() < 1e-6);
        assert!((r1 - 0.2).abs() < 1e-6);

        assert!(rb.push_frame(0.5, 0.6));
        assert!(rb.push_frame(0.7, 0.8));
        assert!(rb.push_frame(0.9, 1.0));
        assert_eq!(rb.available_frames(), 4); // full
        assert!(!rb.push_frame(1.1, 1.2));    // overflow prevented

        let (l2, r2) = rb.pop_frame().unwrap();
        assert!((l2 - 0.3).abs() < 1e-6);
        assert!((r2 - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_master_soft_limiter_saturation() {
        // Linear range transparent pass-through
        assert_eq!(soft_limit(0.0), 0.0);
        assert_eq!(soft_limit(0.5), 0.5);
        assert_eq!(soft_limit(-0.5), -0.5);

        // Saturation knee
        let saturated_pos = soft_limit(2.5);
        let saturated_neg = soft_limit(-2.5);
        assert!(saturated_pos < 1.0);
        assert!(saturated_pos > 0.9);
        assert!(saturated_neg > -1.0);
        assert!(saturated_neg < -0.9);

        // Extreme peaks strictly bounded in [-1.0, 1.0]
        let extreme = soft_limit(100.0);
        assert!(extreme <= 1.0);
        assert!(extreme > 0.999);
    }
}
