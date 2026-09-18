//! Offline faster-than-realtime streaming WAV exporter.
//!
//! Generates WAV audio in discrete, low-memory chunks and streams them directly
//! to a writer without buffering the entire uncompressed audio file in memory.

use crate::decoder::{AmbisonicDecoder, DecodeMode, FoaFrame};
use crate::procedural::ProceduralSynthesizer;
use shared::rain::{
    GovernorOptimizationProfile, HardwareStressProfile, MetaControllerInterceptionMode,
    QualityTier, RainState,
};
use std::io::{self, Write};

pub const CHUNK_FRAMES: usize = 2048;

/// Writes a standard 32-bit IEEE float WAV file header
pub fn write_wav_header<W: Write>(
    writer: &mut W,
    num_channels: u16,
    sample_rate: u32,
    num_frames: u32,
) -> io::Result<()> {
    let bits_per_sample = 32u16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_frames * u32::from(block_align);
    let riff_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt subchunk (IEEE float = format 3)
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM/float)
    writer.write_all(&3u16.to_le_bytes())?;  // AudioFormat: 3 = IEEE Float
    writer.write_all(&num_channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    Ok(())
}

/// Streams faster-than-realtime synthesized audio into any Write stream in small chunks
pub fn render_wav_stream<W: Write>(
    state: &RainState,
    duration_secs: f32,
    sample_rate: u32,
    mode: DecodeMode,
    writer: &mut W,
    mut progress_cb: impl FnMut(f32),
) -> io::Result<u64> {
    let num_channels = match mode {
        DecodeMode::RawFoaPassthrough => 4u16,
        DecodeMode::Surround71 => 8u16,
        DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => 2u16,
    };

    let total_frames = (duration_secs * sample_rate as f32).round() as u32;
    write_wav_header(writer, num_channels, sample_rate, total_frames)?;

    let mut synth = ProceduralSynthesizer::new(sample_rate as f32);
    
    // Load inference weights and setup engine
    let weight_cache = inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
    let mut runner = inference::runner::InferenceRunner::new(state.quality_tier, weight_cache);
    let mut decoder = AmbisonicDecoder::new(mode);

    // Force active synth state for offline export even if UI is paused
    let mut active_state = state.clone();
    active_state.is_playing = true;

    // Decouple Offline Export: OfflineMaxQuality unlocks infinite headroom and K=5 deliberation
    if state.meta_mediation_mode == MetaControllerInterceptionMode::OfflineMaxQuality {
        let mut gov = crate::meta_governor::MetaGovernor::new();
        let action = gov.evaluate(
            &state.telemetry,
            QualityTier::StudioFp32,
            false,
            GovernorOptimizationProfile::StudioMaster,
            HardwareStressProfile::NominalDesktop,
            MetaControllerInterceptionMode::OfflineMaxQuality,
            state.user_thinking_steps,
            0.1,
        );
        runner.set_target_tier(QualityTier::StudioFp32);
        runner.set_thinking_steps(action.thinking_steps);
        runner.set_active_experts(action.active_experts);
        runner.set_diffusion_bypass(action.diffusion_bypass);
        runner.update_quantization_bounds(action.min_bits, action.max_bits);
        runner.set_use_consistency_jump(false);
        active_state.telemetry.synthesis_blend = 0.0;
    } else {
        // Direct Parameter Control Mode
        if let Some(steps) = state.user_thinking_steps {
            runner.set_thinking_steps(steps);
        } else {
            runner.set_thinking_steps(state.thinking_steps);
        }
        runner.set_use_consistency_jump(state.use_consistency_jump);
    }

    let blend = active_state.telemetry.synthesis_blend;

    let mut frames_remaining = total_frames;
    let mut foa_buf = [FoaFrame::default(); CHUNK_FRAMES];
    let mut bytes_written = 44u64; // WAV header size

    while frames_remaining > 0 {
        let frames_to_process = (frames_remaining as usize).min(CHUNK_FRAMES);
        let slice = &mut foa_buf[..frames_to_process];

        // Synthesize mixed Procedural/Neural frame buffer
        for frame in slice.iter_mut() {
            let foa_proc = synth.process_frame(&active_state);
            
            *frame = if blend >= 0.999 {
                foa_proc
            } else {
                let cond = active_state.to_conditioning_array();
                let (nw, nx, ny, nz) = if runner.use_consistency_jump && runner.has_consistency_jump_head() {
                    runner.fast_consistency_step(&cond)
                } else {
                    runner.step(&cond)
                };
                let foa_neural = crate::decoder::FoaFrame::new(nw, nx, ny, nz);

                crate::decoder::FoaFrame::new(
                    foa_proc.w * blend + foa_neural.w * (1.0 - blend),
                    foa_proc.x * blend + foa_neural.x * (1.0 - blend),
                    foa_proc.y * blend + foa_neural.y * (1.0 - blend),
                    foa_proc.z * blend + foa_neural.z * (1.0 - blend),
                )
            };
        }

        match mode {
            DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => {
                for frame in slice.iter() {
                    let stereo = decoder.decode_stereo(*frame);
                    let left = crate::engine::soft_limit(stereo.left);
                    let right = crate::engine::soft_limit(stereo.right);
                    writer.write_all(&left.to_le_bytes())?;
                    writer.write_all(&right.to_le_bytes())?;
                    bytes_written += 8;
                }
            }
            DecodeMode::RawFoaPassthrough => {
                for frame in slice.iter() {
                    writer.write_all(&frame.w.to_le_bytes())?;
                    writer.write_all(&frame.x.to_le_bytes())?;
                    writer.write_all(&frame.y.to_le_bytes())?;
                    writer.write_all(&frame.z.to_le_bytes())?;
                    bytes_written += 16;
                }
            }
            DecodeMode::Surround71 => {
                for frame in slice.iter() {
                    let s71 = decoder.decode_71(*frame);
                    writer.write_all(&s71.left.to_le_bytes())?;
                    writer.write_all(&s71.right.to_le_bytes())?;
                    writer.write_all(&s71.center.to_le_bytes())?;
                    writer.write_all(&s71.lfe.to_le_bytes())?;
                    writer.write_all(&s71.left_surround.to_le_bytes())?;
                    writer.write_all(&s71.right_surround.to_le_bytes())?;
                    writer.write_all(&s71.left_back.to_le_bytes())?;
                    writer.write_all(&s71.right_back.to_le_bytes())?;
                    bytes_written += 32;
                }
            }
        }

        frames_remaining -= frames_to_process as u32;
        let progress = 1.0 - (frames_remaining as f32 / total_frames as f32);
        progress_cb(progress);
    }

    Ok(bytes_written)
}

/// GPU-accelerated faster-than-realtime streaming WAV exporter.
///
/// Uses WebGPU / GpuInferenceRunner when available to process neural trajectory frames
/// in large batch chunks, streaming results directly to the writer with automatic CPU fallback.
pub async fn render_wav_stream_gpu<W: Write>(
    state: &RainState,
    duration_secs: f32,
    sample_rate: u32,
    mode: DecodeMode,
    writer: &mut W,
    mut progress_cb: impl FnMut(f32),
) -> io::Result<u64> {
    let weight_cache = inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
    
    // Attempt to initialize GPU inference runner
    match inference::webgpu_backend::GpuInferenceRunner::new(&weight_cache).await {
        Ok((mut gpu_runner, _consumer)) => {
            let num_channels = match mode {
                DecodeMode::RawFoaPassthrough => 4u16,
                DecodeMode::Surround71 => 8u16,
                DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => 2u16,
            };

            let total_frames = (duration_secs * sample_rate as f32).round() as u32;
            write_wav_header(writer, num_channels, sample_rate, total_frames)?;

            let mut synth = ProceduralSynthesizer::new(sample_rate as f32);
            let mut decoder = AmbisonicDecoder::new(mode);

            let mut frames_remaining = total_frames;
            let mut foa_buf = [FoaFrame::default(); CHUNK_FRAMES];
            let mut bytes_written = 44u64;

            let mut active_state = state.clone();
            active_state.is_playing = true;
            let blend = active_state.telemetry.synthesis_blend;

            while frames_remaining > 0 {
                let frames_to_process = (frames_remaining as usize).min(CHUNK_FRAMES);
                let slice = &mut foa_buf[..frames_to_process];

                for frame in slice.iter_mut() {
                    let foa_proc = synth.process_frame(&active_state);
                    *frame = if blend >= 0.999 {
                        foa_proc
                    } else {
                        let cond = active_state.to_conditioning_array();
                        let (nw, nx, ny, nz) = gpu_runner.step_async(&cond).await.unwrap_or((0.0, 0.0, 0.0, 0.0));
                        let foa_neural = crate::decoder::FoaFrame::new(nw, nx, ny, nz);
                        crate::decoder::FoaFrame::new(
                            foa_proc.w * blend + foa_neural.w * (1.0 - blend),
                            foa_proc.x * blend + foa_neural.x * (1.0 - blend),
                            foa_proc.y * blend + foa_neural.y * (1.0 - blend),
                            foa_proc.z * blend + foa_neural.z * (1.0 - blend),
                        )
                    };
                }

                match mode {
                    DecodeMode::BinauralHeadphones | DecodeMode::StereoSpeakers => {
                        for frame in slice.iter() {
                            let stereo = decoder.decode_stereo(*frame);
                            let left = crate::engine::soft_limit(stereo.left);
                            let right = crate::engine::soft_limit(stereo.right);
                            writer.write_all(&left.to_le_bytes())?;
                            writer.write_all(&right.to_le_bytes())?;
                            bytes_written += 8;
                        }
                    }
                    DecodeMode::RawFoaPassthrough => {
                        for frame in slice.iter() {
                            writer.write_all(&frame.w.to_le_bytes())?;
                            writer.write_all(&frame.x.to_le_bytes())?;
                            writer.write_all(&frame.y.to_le_bytes())?;
                            writer.write_all(&frame.z.to_le_bytes())?;
                            bytes_written += 16;
                        }
                    }
                    DecodeMode::Surround71 => {
                        for frame in slice.iter() {
                            let s71 = decoder.decode_71(*frame);
                            writer.write_all(&s71.left.to_le_bytes())?;
                            writer.write_all(&s71.right.to_le_bytes())?;
                            writer.write_all(&s71.center.to_le_bytes())?;
                            writer.write_all(&s71.lfe.to_le_bytes())?;
                            writer.write_all(&s71.left_surround.to_le_bytes())?;
                            writer.write_all(&s71.right_surround.to_le_bytes())?;
                            writer.write_all(&s71.left_back.to_le_bytes())?;
                            writer.write_all(&s71.right_back.to_le_bytes())?;
                            bytes_written += 32;
                        }
                    }
                }

                frames_remaining -= frames_to_process as u32;
                let progress = 1.0 - (frames_remaining as f32 / total_frames as f32);
                progress_cb(progress);
            }

            Ok(bytes_written)
        }
        Err(err) => {
            tracing::warn!("WebGPU offline export initialization failed ({}); falling back to CPU streaming export.", err);
            render_wav_stream(state, duration_secs, sample_rate, mode, writer, progress_cb)
        }
    }
}