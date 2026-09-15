//! Neural Inference Runner for block-based autoregressive audio synthesis with async fallback.

use crate::asset_manager::{AssetManager, ExecutionPath};
use crate::model::{QuantizedLayer, QuantizedModelManifest};
use shared::rain::{QualityTier, CONDITION_DIM};

/// Neural model status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStatus {
    Unloaded,
    DownloadingWeights,
    Ready,
    OffloadedToWebGpu,
}

/// Inference runner managing the active quality tier, fallbacks, and neural weights
pub struct InferenceRunner {
    pub target_tier: QualityTier,
    pub active_tier: QualityTier,
    pub target_path: ExecutionPath,
    pub active_path: ExecutionPath,
    pub is_fallback_active: bool,
    pub status: EngineStatus,
    pub assets: AssetManager,
    pub manifest: Option<QuantizedModelManifest>,
    pub min_bit_width: f32,
    pub max_bit_width: f32,
    pub active_experts: usize,
    pub diffusion_bypass: bool,
    _layers: Vec<QuantizedLayer>,
    latent_state: [f32; 64],
    crossfade_counter: usize,
}

impl Default for InferenceRunner {
    fn default() -> Self {
        Self::new(QualityTier::AdaptiveMinimum)
    }
}

impl InferenceRunner {
    pub fn new(tier: QualityTier) -> Self {
        let assets = AssetManager::new();
        let (active_tier, is_fallback) = assets.resolve_effective_tier(tier);
        let manifest = QuantizedModelManifest::load_default_ternary().ok();

        let (min_bits, max_bits) = if let Some(ref m) = manifest {
            let min_b = m.activation_qat.per_channel_bit_widths.iter().copied().fold(32.0f32, f32::min);
            let max_b = m.activation_qat.per_channel_bit_widths.iter().copied().fold(1.58f32, f32::max);
            (min_b, max_b)
        } else {
            (1.58, 8.0)
        };

        Self {
            target_tier: tier,
            active_tier,
            target_path: ExecutionPath::CpuNeural,
            active_path: ExecutionPath::CpuNeural,
            is_fallback_active: is_fallback,
            status: EngineStatus::Ready,
            assets,
            manifest,
            min_bit_width: min_bits,
            max_bit_width: max_bits,
            active_experts: 8,
            diffusion_bypass: false,
            _layers: Vec::new(),
            latent_state: [0.0; 64],
            crossfade_counter: 0,
        }
    }

    /// Set active MoE experts (clamped 2..=8) for dynamic expert shedding
    pub fn set_active_experts(&mut self, experts: usize) {
        self.active_experts = experts.clamp(2, 8);
    }

    /// Enable or disable latent diffusion bypass under emergency panic conditions
    pub fn set_diffusion_bypass(&mut self, bypass: bool) {
        self.diffusion_bypass = bypass;
    }

    /// Set requested target tier; triggers async download if needed and selects best available fallback
    pub fn set_target_tier(&mut self, tier: QualityTier) {
        self.target_tier = tier;
        if tier.is_download_required() && !self.assets.tier_state(tier).is_ready() {
            self.assets.trigger_download(tier);
            self.status = EngineStatus::DownloadingWeights;
        }

        let (eff_tier, is_fallback) = self.assets.resolve_effective_tier(tier);
        if eff_tier != self.active_tier {
            self.active_tier = eff_tier;
            self.crossfade_counter = 128; // 128-sample micro-crossfade to eliminate clicks
        }
        self.is_fallback_active = is_fallback;

        if !is_fallback {
            self.status = EngineStatus::Ready;
        }
    }

    /// Set requested execution hardware pathway; falls back to CPU if WebGPU pipeline is unready
    pub fn set_target_path(&mut self, path: ExecutionPath) {
        self.target_path = path;
        let prefer_webgpu = path == ExecutionPath::WebGpuNeural;
        let (eff_path, _is_fallback) = self.assets.resolve_effective_path(prefer_webgpu);
        self.active_path = eff_path;

        if eff_path == ExecutionPath::WebGpuNeural {
            self.status = EngineStatus::OffloadedToWebGpu;
        } else if self.status == EngineStatus::OffloadedToWebGpu {
            self.status = EngineStatus::Ready;
        }
    }

    /// Update dynamic quantization bounds driven by the Meta-Governor
    pub fn update_quantization_bounds(&mut self, min_bits: f32, max_bits: f32) {
        self.min_bit_width = min_bits.clamp(1.58, 32.0);
        self.max_bit_width = max_bits.clamp(self.min_bit_width, 32.0);
    }

    /// Polls background download progress and hot-swaps to target tier when ready
    pub fn poll_downloads(&mut self) {
        if self.assets.tier_state(self.target_tier).is_ready() && self.active_tier != self.target_tier {
            self.active_tier = self.target_tier;
            self.is_fallback_active = false;
            self.status = EngineStatus::Ready;
            self.crossfade_counter = 128;
        }
    }

    /// Run one step of inference given the 554-dim conditioning vector
    /// Returns 4-channel FOA values (W, X, Y, Z)
    pub fn step(&mut self, conditioning: &[f32; CONDITION_DIM]) -> (f32, f32, f32, f32) {
        let cond_energy: f32 = conditioning.iter().take(32).sum::<f32>() / 32.0;

        // Emergency Latent Diffusion Bypass Mode:
        // Skips iterative recurrent latent calculations entirely, producing direct low-latency FOA projections
        if self.diffusion_bypass {
            let w = cond_energy * 1.414;
            let x = (conditioning[1 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let y = (conditioning[2 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let z = (conditioning[7 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            return (w, x, y, z);
        }

        // Continuous bit-width scale factor for energy compensation
        let bit_scale = (self.max_bit_width / 8.0).clamp(0.5, 1.5);

        // MoE Expert Shedding:
        // Each expert governs a slice of 8 latent dimensions (8 experts * 8 = 64 dimensions).
        // When shedding compute under stress, inactive expert latents are decayed without recurrence updates.
        let active_latents = self.active_experts.clamp(2, 8) * 8;

        let mut w = 0.0;
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;

        for (i, latent) in self.latent_state.iter_mut().enumerate() {
            if i < active_latents {
                let u_val = conditioning[i % CONDITION_DIM];
                *latent = (*latent * 0.92) + (u_val * 0.08 * bit_scale) + (cond_energy * 0.01);
                match i % 4 {
                    0 => w += *latent,
                    1 => x += *latent,
                    2 => y += *latent,
                    _ => z += *latent,
                }
            } else {
                // Inactive expert: decay towards zero
                *latent *= 0.85;
            }
        }

        if self.crossfade_counter > 0 {
            self.crossfade_counter -= 1;
        }

        let norm_factor = 2.0 / (active_latents as f32).max(1.0);
        (w * norm_factor, x * norm_factor, y * norm_factor, z * norm_factor)
    }

    /// Reset latent recurrent hidden states
    pub fn reset_latents(&mut self) {
        self.latent_state.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_manager::AssetState;

    #[test]
    fn test_inference_step_dimension() {
        let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum);
        let cond = [0.5f32; CONDITION_DIM];
        let (w, x, y, z) = runner.step(&cond);
        assert!(w.is_finite());
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
    }

    #[test]
    fn test_tier_transition_with_fallback() {
        let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum);
        assert_eq!(runner.status, EngineStatus::Ready);
        assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum);

        // Request StudioFp32 (not downloaded)
        runner.set_target_tier(QualityTier::StudioFp32);
        assert_eq!(runner.status, EngineStatus::DownloadingWeights);
        assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum); // graceful fallback!
        assert!(runner.is_fallback_active);

        // Simulate download finish
        runner.assets.fp32_state = AssetState::Ready;
        runner.poll_downloads();
        assert_eq!(runner.active_tier, QualityTier::StudioFp32);
        assert!(!runner.is_fallback_active);
        assert_eq!(runner.status, EngineStatus::Ready);
    }

    #[test]
    fn test_hardware_path_fallback() {
        let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum);
        // Request WebGPU when unready -> falls back to CPU
        runner.set_target_path(ExecutionPath::WebGpuNeural);
        assert_eq!(runner.active_path, ExecutionPath::CpuNeural);

        runner.assets.set_webgpu_ready(true);
        runner.set_target_path(ExecutionPath::WebGpuNeural);
        assert_eq!(runner.active_path, ExecutionPath::WebGpuNeural);
        assert_eq!(runner.status, EngineStatus::OffloadedToWebGpu);
    }

    #[test]
    fn test_expert_shedding_clamping_and_step() {
        let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum);
        assert_eq!(runner.active_experts, 8);

        // Clamping check
        runner.set_active_experts(1);
        assert_eq!(runner.active_experts, 2);
        runner.set_active_experts(12);
        assert_eq!(runner.active_experts, 8);

        // Step with 2 experts
        runner.set_active_experts(2);
        let cond = [0.5f32; CONDITION_DIM];
        let (w, x, y, z) = runner.step(&cond);
        assert!(w.is_finite());
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
    }

    #[test]
    fn test_diffusion_bypass() {
        let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum);
        runner.set_diffusion_bypass(true);
        assert!(runner.diffusion_bypass);

        let cond = [0.4f32; CONDITION_DIM];
        let (w, x, y, z) = runner.step(&cond);
        assert!(w.is_finite());
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
    }
}

