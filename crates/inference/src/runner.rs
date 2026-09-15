//! Neural Inference Runner for block-based autoregressive audio synthesis with async fallback.

use crate::asset_manager::{AssetManager, ExecutionPath};
use crate::kernels;
use crate::model::{PrecisionFormat, QuantizedLayer, QuantizedModelManifest};
use crate::weight_cache_manager::WeightCacheManager;
use crate::weight_loader::WeightCache;
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
#[derive(Clone, Debug)]
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
    pub weight_cache: WeightCache,
    _layers: Vec<QuantizedLayer>,
    latent_state: [f32; 64],
    u_t_buffer: [f32; 64],
    crossfade_counter: usize,
}

impl Default for InferenceRunner {
    fn default() -> Self {
        Self::new(QualityTier::AdaptiveMinimum, WeightCache::default())
    }
}

impl InferenceRunner {
    pub fn new(tier: QualityTier, weight_cache: WeightCache) -> Self {
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
            weight_cache,
            _layers: Vec::new(),
            latent_state: [0.0; 64],
            u_t_buffer: [0.0; 64],
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

        // Emergency Latent Diffusion Bypass Mode
        if self.diffusion_bypass {
            let w = cond_energy * 1.414;
            let x = (conditioning[1 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let y = (conditioning[2 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            let z = (conditioning[7 % CONDITION_DIM] - 0.5) * 2.0 * cond_energy;
            return (w, x, y, z);
        }

        // 1. Conditioning Projection: u_t = W_in @ c_t + b_in
        if let Some(cond_layer) = self.weight_cache.get("encoder.cond_proj.weight") {
            let bias = self.weight_cache.get("encoder.cond_proj.bias");
            let bias_ref = bias.as_ref().map(|b| b.weights.as_slice());
            
            if cond_layer.format == PrecisionFormat::Ternary158 {
                kernels::ternary_matmul_simd_f32(
                    &cond_layer.packed_weights,
                    conditioning,
                    &mut self.u_t_buffer,
                    cond_layer.scale,
                );
                if let Some(b) = bias_ref {
                    for (u, &bias_val) in self.u_t_buffer.iter_mut().zip(b.iter()) {
                        *u += bias_val;
                    }
                }
            } else {
                kernels::dense_projection(
                    conditioning, 
                    &cond_layer.weights, 
                    bias_ref, 
                    &mut self.u_t_buffer
                );
            }
        }

        // Apply Mamba2 SiLU Gating Activation
        kernels::simd_silu_in_place(&mut self.u_t_buffer);

        // 2. Mamba2 Recurrence: s_t = A * s_{t-1} + B * u_t
        if let (Some(a_diag), Some(b_diag)) = (
            self.weight_cache.get("mamba.A_diag.weight"),
            self.weight_cache.get("mamba.B_diag.weight")
        ) {
            kernels::step_recurrence_f32(
                &mut self.latent_state,
                &a_diag.weights,
                &b_diag.weights,
                &self.u_t_buffer,
            );
        }

        // 3. MoE Dispatch & Expert Shedding
        let top_k = 2; // Route to top 2 experts
        let decay_factor = 0.85; // Unselected expert state decay
        
        if let Some(router) = self.weight_cache.get("moe.router.weight") {
            kernels::route_and_decay(
                &mut self.latent_state,
                &router.weights,
                8, // Total experts
                top_k,
                decay_factor,
            );
        }

        // 4. Ambisonic FOA Projection: y_t = W_foa @ s_t
        let mut foa_out = [0.0; 4];
        if let Some(foa_layer) = self.weight_cache.get("decoder.foa_proj.weight") {
            if foa_layer.format == PrecisionFormat::Ternary158 {
                kernels::ternary_matmul_simd_f32(
                    &foa_layer.packed_weights,
                    &self.latent_state,
                    &mut foa_out,
                    foa_layer.scale,
                );
            } else {
                kernels::dense_projection(
                    &self.latent_state,
                    &foa_layer.weights,
                    None, 
                    &mut foa_out
                );
            }
        }

        if self.crossfade_counter > 0 {
            self.crossfade_counter -= 1;
        }

        // Bit-width scaling for energy compensation across tiers
        let bit_scale = (self.max_bit_width / 8.0).clamp(0.5, 1.5);
        
        (
            foa_out[0] * bit_scale, 
            foa_out[1] * bit_scale, 
            foa_out[2] * bit_scale, 
            foa_out[3] * bit_scale
        )
    }

    /// Asynchronously requests a quality tier upgrade and loads weights via IndexedDB/Network
    pub async fn upgrade_tier_async(&mut self, tier: QualityTier) -> Result<(), String> {
        self.set_target_tier(tier);
        
        if tier.is_download_required() && !self.assets.tier_state(tier).is_ready() {
            let new_cache = WeightCacheManager::load_tier(tier).await?;
            self.weight_cache = new_cache;
            
            *self.assets.tier_state_mut(tier) = crate::asset_manager::AssetState::Ready;
            self.active_tier = tier;
            self.is_fallback_active = false;
            self.status = EngineStatus::Ready;
        }
        Ok(())
    }
}
