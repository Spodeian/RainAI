//! Acoustic History Buffer & Retro-Refine "Rewind & Upgrade" Engine.
//!
//! Maintains a circular memory-bounded ring buffer of conditioning trajectories
//! and synthesized FOA frames, enabling non-causal retroactive upgrade of past audio
//! into Studio Master quality with 5 deliberation thinking steps and lookahead smoothing.

use crate::decoder::{AmbisonicDecoder, FoaFrame};
use inference::runner::InferenceRunner;
use shared::rain::CONDITION_DIM;

/// Circular history recorder storing past conditioning states and audio frames.
#[derive(Clone, Debug)]
pub struct AcousticHistoryBuffer {
    capacity_frames: usize,
    sample_rate: f32,
    audio_history: Vec<FoaFrame>,
    cond_history: Vec<[f32; CONDITION_DIM]>,
    write_pos: usize,
    total_written: usize,
}

impl AcousticHistoryBuffer {
    /// Allocate history buffer with given maximum duration (e.g. 30.0 seconds)
    pub fn new(max_seconds: f32, sample_rate: f32) -> Self {
        let capacity_frames = (max_seconds * sample_rate).round() as usize;
        Self {
            capacity_frames,
            sample_rate,
            audio_history: vec![FoaFrame::default(); capacity_frames],
            cond_history: vec![[0.0f32; CONDITION_DIM]; capacity_frames],
            write_pos: 0,
            total_written: 0,
        }
    }

    /// Records an audio frame and its corresponding conditioning state.
    #[inline]
    pub fn record_frame(&mut self, cond: &[f32; CONDITION_DIM], foa: FoaFrame) {
        let idx = self.write_pos;
        self.audio_history[idx] = foa;
        self.cond_history[idx] = *cond;
        self.write_pos = (self.write_pos + 1) % self.capacity_frames;
        self.total_written += 1;
    }

    /// Available recorded history in seconds
    #[inline]
    pub fn available_seconds(&self) -> f32 {
        let count = self.total_written.min(self.capacity_frames);
        count as f32 / self.sample_rate.max(1.0)
    }

    /// Number of recorded frames currently stored in the ring buffer
    #[inline]
    pub fn available_frames(&self) -> usize {
        self.total_written.min(self.capacity_frames)
    }

    /// Extracts past seconds of conditioning trajectory and original FOA audio linearly ordered from oldest to newest.
    pub fn extract_window(&self, seconds: f32) -> (Vec<[f32; CONDITION_DIM]>, Vec<FoaFrame>) {
        let frames_wanted = ((seconds * self.sample_rate).round() as usize)
            .min(self.available_frames());
        if frames_wanted == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut cond_out = Vec::with_capacity(frames_wanted);
        let mut audio_out = Vec::with_capacity(frames_wanted);

        let current_count = self.available_frames();
        let start_offset = current_count.saturating_sub(frames_wanted);

        // Circular ring calculation
        let ring_start = if self.total_written >= self.capacity_frames {
            (self.write_pos + start_offset) % self.capacity_frames
        } else {
            start_offset
        };

        for i in 0..frames_wanted {
            let idx = (ring_start + i) % self.capacity_frames;
            cond_out.push(self.cond_history[idx]);
            audio_out.push(self.audio_history[idx]);
        }

        (cond_out, audio_out)
    }

    /// Non-causally re-synthesizes past seconds of audio at maximum Studio Master quality (K=5 thinking steps)
    /// with bidirectional trajectory lookahead smoothing.
    pub fn retro_upgrade_window(
        &self,
        seconds: f32,
        runner: &mut InferenceRunner,
        decoder: &mut AmbisonicDecoder,
    ) -> Vec<f32> {
        let (cond_trajectory, _original_audio) = self.extract_window(seconds);
        if cond_trajectory.is_empty() {
            return Vec::new();
        }

        // Lock maximum deliberation quality for non-realtime offline upgrade
        let original_thinking = runner.thinking_steps;
        let original_jump = runner.use_consistency_jump;
        runner.set_thinking_steps(5);
        runner.set_use_consistency_jump(false);

        let n = cond_trajectory.len();
        let mut upgraded_stereo = Vec::with_capacity(n * 2);

        // Forward-backward bidirectional lookahead smoothing on conditioning trajectory
        let mut smoothed_cond = cond_trajectory.clone();
        if n >= 3 {
            for i in 1..n - 1 {
                for d in 0..32 {
                    // Non-causal 3-point Gaussian kernel smoothing on acoustic control dimensions
                    smoothed_cond[i][d] = 0.25 * cond_trajectory[i - 1][d]
                        + 0.50 * cond_trajectory[i][d]
                        + 0.25 * cond_trajectory[i + 1][d];
                }
            }
        }

        // High-precision neural synthesis pass
        for cond in &smoothed_cond {
            let (w, x, y, z) = runner.step(cond);
            let foa = FoaFrame::new(w, x, y, z);
            let stereo = decoder.decode_stereo(foa);
            upgraded_stereo.push(crate::engine::soft_limit(stereo.left));
            upgraded_stereo.push(crate::engine::soft_limit(stereo.right));
        }

        // Restore original runner state
        runner.set_thinking_steps(original_thinking);
        runner.set_use_consistency_jump(original_jump);

        upgraded_stereo
    }

    /// Exports past seconds of audio upgraded to Studio Master into a standard 32-bit float WAV buffer.
    pub fn retro_upgrade_to_wav_buffer(&self, seconds: f32) -> Result<Vec<u8>, std::io::Error> {
        let weight_cache = inference::weight_loader::WeightLoader::load_embedded_ternary().unwrap_or_default();
        let mut runner = InferenceRunner::new(shared::rain::QualityTier::StudioFp32, weight_cache);
        let mut decoder = AmbisonicDecoder::new(crate::decoder::DecodeMode::BinauralHeadphones);
        let stereo = self.retro_upgrade_window(seconds, &mut runner, &mut decoder);

        if stereo.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "No acoustic history recorded yet to retro-upgrade",
            ));
        }

        let num_frames = (stereo.len() / 2) as u32;
        let mut wav_bytes = Vec::with_capacity(44 + stereo.len() * 4);
        crate::export::write_wav_header(&mut wav_bytes, 2, self.sample_rate as u32, num_frames)?;
        for sample in stereo {
            wav_bytes.extend_from_slice(&sample.to_le_bytes());
        }
        Ok(wav_bytes)
    }
}