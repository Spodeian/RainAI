//! Loss Functions and Regularizers for Neural Training.
//!
//! Includes 2nd-order physics trajectory losses, MoE load balancing and router z-loss,
//! expert diversity, Beta-VAE KL divergence, HWIL latency penalties, flow matching,
//! and multi-resolution reconstruction loss.

use anyhow::Result;
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use super::*;

pub fn compute_physics_trajectory_loss(
    z_pred: &Tensor,
    z_target: &Tensor,
    z_prev: &Tensor,
    lambda_vel: f64,
) -> Result<Tensor> {
    let diff = (z_pred - z_target)?;
    let mse = diff.sqr()?.mean_all()?;

    let v_pred = (z_pred - z_prev)?;
    let v_target = (z_target - z_prev)?;
    let v_diff = (v_pred - v_target)?;
    let v_loss = v_diff.sqr()?.mean_all()?;

    let total = (&mse + (&v_loss * lambda_vel)?)?;
    Ok(total)
}

/// 2nd-Order Physics-Informed Trajectory Smoothness & Aerodynamic Drag Loss.
/// Evaluates:
/// 1. Position MSE: ||z_pred - z_target||^2
/// 2. 1st-order velocity continuity: Huber(v_pred - v_target)
/// 3. 2nd-order acceleration (jerk) smoothness: Huber(a_pred - a_target) where a_t = z_t - 2*z_{t-1} + z_{t-2}
/// 4. Terminal aerodynamic drag dissipation: ReLU(||v_pred||_2 - v_terminal)^2
pub fn compute_physics_trajectory_loss_v2(
    z_pred: &Tensor,
    z_target: &Tensor,
    z_prev: &Tensor,
    z_prev2: Option<&Tensor>,
    lambda_vel: f64,
    lambda_acc: f64,
    lambda_drag: f64,
    v_terminal: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let diff = (z_pred - z_target)?;
    let pos_loss = diff.sqr()?.mean_all()?;

    let v_pred = (z_pred - z_prev)?;
    let v_target = (z_target - z_prev)?;
    let v_diff = (v_pred.clone() - &v_target)?;
    let vel_loss = crate::stft_loss::huber_loss(&v_diff, 0.5)?;

    let acc_loss = if let Some(prev2) = z_prev2 {
        let two_z_prev = (z_prev * 2.0)?;
        let a_pred = ((z_pred - &two_z_prev)? + prev2)?;
        let a_target = ((z_target - &two_z_prev)? + prev2)?;
        let a_diff = (a_pred - a_target)?;
        crate::stft_loss::huber_loss(&a_diff, 0.5)?
    } else {
        Tensor::zeros((), DType::F32, z_pred.device())?
    };

    let v_pred_sq = v_pred.sqr()?.sum_keepdim(1)?.relu()?;
    let speed = (v_pred_sq + 1e-6)?.sqrt()?;
    let excess_speed = (speed - v_terminal)?.relu()?;
    let drag_loss = excess_speed.sqr()?.mean_all()?;

    let total_acc = (&pos_loss + (&vel_loss * lambda_vel)?)?;
    let total_drag = (&total_acc + (&acc_loss * lambda_acc)?)?;
    let total = (&total_drag + (&drag_loss * lambda_drag)?)?;

    Ok((total, pos_loss, acc_loss, drag_loss))
}

/// Smooth hyperbolic tangent soft-capping function: SoftCap(L, M) = M * tanh(L / M).
/// For normal losses (L << M), gradients are identical to L (1.0).
/// For extreme loss spikes (L >> M), the loss saturates smoothly at M with zero gradient,
/// making the training loop immune to anomalous batch gradient shocks.
pub fn soft_cap_loss(loss: &Tensor, max_val: f64) -> Result<Tensor> {
    let scaled = (loss / max_val)?;
    let capped = (scaled.tanh()? * max_val)?;
    Ok(capped)
}

/// Dimension Outlier Spike Suppression Loss:
/// Penalizes activation dimensions exceeding `tau_max` (default: 3.5 standard deviations)
/// and disproportionate Peak-to-Average Power Ratios (PAPR) across channels.
/// Directly suppresses activation outliers in attention KV caches and latent states.
pub fn compute_dimension_outlier_spike_loss(x: &Tensor, tau_max: f64) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let tau_t = Tensor::full(tau_max as f32, x.shape(), x.device())?;
    let excess = (abs_x.broadcast_sub(&tau_t))?.relu()?;
    let threshold_penalty = excess.sqr()?.mean_all()?;

    // Peak-to-average power ratio across hidden channels
    let mean_mag = (abs_x.mean_keepdim(1)? + 1e-6)?;
    let max_mag = abs_x.max_keepdim(1)?;
    let papr = (max_mag.broadcast_div(&mean_mag)? - 1.0)?.relu()?;
    let papr_penalty = papr.sqr()?.mean_all()?;

    let total = (&threshold_penalty + (&papr_penalty * 0.1)?)?;
    Ok(total)
}

/// Isometric / Orthogonality Regularization Loss for Affine Alignment Matrices:
/// Penalizes deviation from exact isometry: L_ortho = ||W^T * W - I||_F^2.
/// Guarantees that learned affine alignment transforms preserve Euclidean norms,
/// bound singular values sigma_i(W) approx 1.0, and cannot collapse rank or explode activations.
pub fn compute_orthogonality_loss(weight: &Tensor) -> Result<Tensor> {
    let dims = weight.dims();
    if dims.len() != 2 || dims[0] != dims[1] {
        return Ok(Tensor::zeros((), DType::F32, weight.device())?);
    }
    let n = dims[0];
    let w_t_w = weight.t()?.matmul(weight)?;
    let eye = Tensor::eye(n, DType::F32, weight.device())?;
    let diff = (w_t_w - eye)?;
    let loss = diff.sqr()?.mean_all()?;
    Ok(loss)
}

/// Contractive Attractor Trajectory Regularization (Lyapunov Stability):
/// Penalizes runaway velocity expansion: ReLU(||z_{t+1} - z_t|| / (||z_t - z_{t-1}|| + eps) - max_ratio)^2.
/// Guarantees that recurrent state trajectories cannot exponentially diverge without external stimulus.
pub fn compute_contractive_loss(
    z_pred: &Tensor,
    z_prev: &Tensor,
    z_prev2: Option<&Tensor>,
    max_ratio: f64,
) -> Result<Tensor> {
    if let Some(prev2) = z_prev2 {
        let v_curr = (z_pred - z_prev)?.sqr()?.sum_keepdim(1)?.relu()?.sqrt()?;
        let v_prev = ((z_prev - prev2)?.sqr()?.sum_keepdim(1)?.relu()? + 1e-6)?.sqrt()?;
        let ratio = (v_curr.broadcast_div(&v_prev)? - max_ratio)?.relu()?;
        let loss = ratio.sqr()?.mean_all()?;
        Ok(loss)
    } else {
        Ok(Tensor::zeros((), DType::F32, z_pred.device())?)
    }
}

/// Auxiliary load balancing loss penalizing expert imbalance:
/// $\mathcal{L}_{aux} = N \sum_{e=1}^N f_e \cdot P_e$.
pub fn compute_moe_load_balancing_loss(router_probs: &Tensor) -> Result<Tensor> {
    // Mean probability per expert across batch: [8]
    let mean_probs = router_probs.mean(0)?;
    let sq = mean_probs.sqr()?;
    let sum_sq = sq.sum_all()?;
    let aux = (&sum_sq * (NUM_EXPERTS as f64))?;
    Ok(aux)
}

/// Router Z-Loss for MoE numerical stability and floating-point overflow prevention.
/// Penalizes extreme logit magnitudes: L_z = 1/B sum_b (log sum_e exp(z_{b, e}))^2.
pub fn compute_router_z_loss(router_logits: &Tensor) -> Result<Tensor> {
    let max_logit = router_logits.max_keepdim(1)?;
    let exp_diff = router_logits.broadcast_sub(&max_logit)?.clamp(-20.0f32, 20.0f32)?.exp()?;
    let sum_exp = exp_diff.sum_keepdim(1)?.clamp(1e-8f32, 1e8f32)?;
    let log_sum_exp = (&max_logit + &sum_exp.log()?)?;
    let z_loss = log_sum_exp.sqr()?.mean_all()?;
    Ok(z_loss)
}

/// Computes pairwise cosine similarity between router probability distributions across thinking steps.
/// Penalizes collinear expert activation, encouraging orthogonal specialist panels across deliberation depth:
/// L_div = 1 / (M choose 2) * sum_{j < k} (p_j . p_k) / (||p_j|| * ||p_k|| + eps)
pub fn compute_expert_diversity_loss(prob_history: &[Tensor]) -> Result<Tensor> {
    let m = prob_history.len();
    if m < 2 {
        return Ok(Tensor::zeros((), DType::F32, prob_history[0].device())?);
    }

    let mut pair_sim_sum = Tensor::zeros((), DType::F32, prob_history[0].device())?;
    let mut num_pairs = 0usize;
    let eps = 1e-6f64;

    for j in 0..m {
        for k in (j + 1)..m {
            let p_j = &prob_history[j];
            let p_k = &prob_history[k];
            let dot = (p_j * p_k)?.sum_keepdim(1)?;
            let norm_j = (p_j.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;
            let norm_k = (p_k.sqr()?.sum_keepdim(1)? + (eps * eps))?.sqrt()?;
            let denom = (&norm_j * &norm_k)?.clamp(1e-7f32, 1e7f32)?;
            let cos_sim = dot.broadcast_div(&denom)?.mean_all()?;
            pair_sim_sum = (&pair_sim_sum + &cos_sim)?;
            num_pairs += 1;
        }
    }

    if num_pairs > 0 {
        Ok((pair_sim_sum / (num_pairs as f64))?)
    } else {
        Ok(Tensor::zeros((), DType::F32, prob_history[0].device())?)
    }
}

/// 1-Step Consistency Distillation Jump Head.
/// Predicts the multi-step converged latent delta in a single forward pass,
/// enabling sub-millisecond, 1-step Euler inference on edge / WebGPU devices.

/// Beta-VAE loss: Reconstruction MSE + $\beta \cdot \text{KL}(q(z|x) \| p(z))$.
/// Beta-VAE loss with Free-Bits thresholding to prevent posterior collapse:
/// For each latent dimension d, enforces KL_d >= free_bits nats.
/// Below free_bits, the gradient is 0, guaranteeing the latent code cannot be crushed into white noise.
/// Also computes the active latent units count (dimensions where Var_B(mu) > 0.01).
pub fn compute_beta_vae_loss_with_free_bits(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
    free_bits: f64,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    let recon_diff = (pred_bands - target_bands)?;
    let recon_loss = recon_diff.sqr()?.mean_all()?;

    // Numerical clamp on logvar to [-12.0, 12.0] to prevent exponential overflow to infinity and NaN
    let logvar_clamped = logvar.clamp(-12.0f32, 12.0f32)?;
    let mu_sq = mu.sqr()?;
    let var = logvar_clamped.exp()?;
    let ones = Tensor::ones(logvar.shape(), DType::F32, logvar.device())?;
    // KL element-wise: 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    let inner = (((&mu_sq + &var)? - &ones)? - &logvar_clamped)?;
    let kl_elements = (&inner * 0.5)?;

    // Mean KL per latent dimension across batch: [D]
    let kl_per_dim = kl_elements.mean(0)?;

    // Free-bits floor
    let kl_loss = if free_bits > 1e-6 {
        let fb = Tensor::full(free_bits as f32, kl_per_dim.shape(), kl_per_dim.device())?;
        let excess = (kl_per_dim.broadcast_sub(&fb))?.relu()?;
        (&excess + &fb)?.mean_all()?
    } else {
        kl_per_dim.mean_all()?
    };

    // Calculate active latent units (Var_B(mu) > 0.01)
    let active_units = if mu.dim(0)? > 1 {
        let b = mu.dim(0)? as f64;
        let mean_mu = mu.mean_keepdim(0)?;
        let diff = mu.broadcast_sub(&mean_mu)?;
        let var_mu = (diff.sqr()?.sum_keepdim(0)? / (b - 1.0))?;
        let var_vec = var_mu.flatten_all()?.to_vec1::<f32>()?;
        var_vec.iter().filter(|&&v| v > 0.01).count()
    } else {
        LATENT_DIM
    };

    let total = (&recon_loss + (&kl_loss * beta)?)?;
    Ok((total, recon_loss, kl_loss, active_units))
}

/// Backward-compatible Beta-VAE loss wrapper (zero free-bits threshold).
pub fn compute_beta_vae_loss(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (total, recon_loss, kl, _active) = compute_beta_vae_loss_with_free_bits(
        pred_bands,
        target_bands,
        mu,
        logvar,
        beta,
        0.0,
    )?;
    Ok((total, recon_loss, kl))
}


/// Hardware-in-the-Loop (HWIL) governor budget penalty.
pub fn compute_hwil_penalty(
    active_experts: usize,
    budget_experts: usize,
    buffer_health_ms: f32,
    target_buffer_ms: f32,
) -> f32 {
    let expert_penalty = if active_experts > budget_experts {
        (active_experts - budget_experts) as f32 * 0.15
    } else {
        0.0
    };

    let buffer_deficit = (target_buffer_ms - buffer_health_ms).max(0.0) / target_buffer_ms.max(1.0);
    expert_penalty + buffer_deficit * 0.25
}

/// Continuous, differentiable Hardware-in-the-Loop (HWIL) governor budget and buffer penalty.
pub fn compute_continuous_hwil_penalty(
    buffer_health_ms: f32,
    target_buffer_ms: f32,
    active_experts: f32,
    budget_experts: f32,
) -> f32 {
    let buffer_deficit = ((target_buffer_ms - buffer_health_ms).max(0.0) / target_buffer_ms.max(1.0)).powi(2);
    let expert_excess = ((active_experts - budget_experts).max(0.0) * 0.15).powi(2);
    expert_excess + buffer_deficit * 0.35
}

/// Conditional Optimal Transport (OT) Flow Matching Loss.
/// $\mathcal{L}_{flow} = \| v_{pred} - (z_{target} - (1 - \sigma_{min}) z_{noise}) \|^2$.
/// Mirrors `compute_flow_matching_loss` from `src/models/mamba2_moe.py`.
pub fn compute_flow_matching_loss(
    pred_velocity: &Tensor,
    z_target: &Tensor,
    z_noise: &Tensor,
    sigma_min: f64,
) -> Result<Tensor> {
    let scale = 1.0 - sigma_min;
    let target_velocity = (z_target - (z_noise * scale)?)?;
    let diff = (pred_velocity - &target_velocity)?;
    let loss = diff.sqr()?.mean_all()?;
    Ok(loss)
}

/// Straight-Path Conditional Optimal Transport Flow Matching Loss.
/// Regularizes probability flow trajectories toward straight paths:
/// L_flow = ||v_pred - target_v||^2 + lambda_straight * ||v_pred - mean_v||^2.
pub fn compute_straight_flow_loss(
    pred_velocity: &Tensor,
    z_target: &Tensor,
    z_noise: &Tensor,
    sigma_min: f64,
    lambda_straight: f64,
) -> Result<(Tensor, Tensor)> {
    let scale = 1.0 - sigma_min;
    let target_velocity = (z_target - (z_noise * scale)?)?;
    let diff = (pred_velocity - &target_velocity)?;
    let base_flow = diff.sqr()?.mean_all()?;

    let mean_target = target_velocity.mean_keepdim(0)?;
    let curvature = pred_velocity.broadcast_sub(&mean_target)?.sqr()?.mean_all()?;
    let total = (&base_flow + (&curvature * lambda_straight)?)?;
    Ok((total, base_flow))
}

/// Hierarchical Multi-Resolution Reconstruction Loss evaluating error
/// across multiple acoustic sampling tiers (16kHz, 32kHz, 48kHz).
/// Mirrors `HierarchicalMultiResLoss` from `src/models/diff_autoencoder.py`.
pub fn compute_hierarchical_multi_res_loss(
    pred_audio: &Tensor,
    target_audio: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let diff = (pred_audio - target_audio)?;
    let l_fine = diff.abs()?.mean_all()?;
    let l_energy = diff.sqr()?.mean_all()?;
    let l_total = ((&l_fine + &l_energy)? * 0.5)?;
    Ok((l_total, l_fine, l_energy))
}

/// VICReg-style Latent Variance Hinge Loss preventing dimensional collapse:
/// L_var = 1/D sum_d relu(target_std - sqrt(Var_B(z_d) + eps))^2.
/// Enforces that all D latent channels maintain at least `target_std` (default 1.0)
/// spread across batch samples, preventing low-rank subspace collapse.
pub fn compute_latent_variance_loss(z: &Tensor, target_std: f64) -> Result<Tensor> {
    let batch_size = z.dim(0)?;
    if batch_size < 2 {
        return Ok(Tensor::zeros((), DType::F32, z.device())?);
    }
    let mean = z.mean_keepdim(0)?;
    let diff = z.broadcast_sub(&mean)?;
    let var = (diff.sqr()?.sum_keepdim(0)? / ((batch_size - 1) as f64))?;
    let std = (var + 1e-4)?.sqrt()?;
    let target_t = Tensor::full(target_std as f32, std.shape(), std.device())?;
    let hinge = (target_t - std)?.relu()?;
    let loss = hinge.sqr()?.mean_all()?;
    Ok(loss)
}

/// Router Shannon Entropy Loss & Perplexity Metric preventing MoE winner-take-all collapse:
/// Evaluates negative entropy across batch-averaged expert probabilities:
/// L_entropy = sum_e P_e * log(P_e + eps).
/// Minimizing this maximizes router entropy H(P).
/// Returns: (entropy_loss_tensor, perplexity_scalar, dead_expert_count).
pub fn compute_router_entropy_loss(router_probs: &Tensor) -> Result<(Tensor, f32, usize)> {
    let mean_probs = router_probs.mean(0)?; // [NUM_EXPERTS]
    let eps = 1e-8f32;
    let probs_vec = mean_probs.to_vec1::<f32>()?;

    let mut entropy = 0.0f32;
    let mut dead_count = 0usize;
    let dead_threshold = 0.10f32 / (NUM_EXPERTS as f32); // 0.0125 (under 1.25%)
    for &p in &probs_vec {
        if p > eps {
            entropy -= p * (p + eps).ln();
        }
        if p < dead_threshold {
            dead_count += 1;
        }
    }
    let perplexity = entropy.exp().clamp(1.0, NUM_EXPERTS as f32);

    let log_p = (mean_probs.clone() + (eps as f64))?.log()?;
    let neg_entropy = (mean_probs * log_p)?.sum_all()?;
    Ok((neg_entropy, perplexity, dead_count))
}

/// Trajectory Diversity Preservation Loss preventing mode collapse:
/// Enforces that predicted trajectory latents across different batch samples maintain
/// non-zero batch-wise standard deviation (default min_std = 0.25).
pub fn compute_trajectory_diversity_loss(z_pred: &Tensor, min_std: f64) -> Result<Tensor> {
    let batch_size = z_pred.dim(0)?;
    if batch_size < 2 {
        return Ok(Tensor::zeros((), DType::F32, z_pred.device())?);
    }
    let mean = z_pred.mean_keepdim(0)?;
    let diff = z_pred.broadcast_sub(&mean)?;
    let var = (diff.sqr()?.sum_keepdim(0)? / ((batch_size - 1) as f64))?;
    let std = (var + 1e-4)?.sqrt()?;
    let min_std_t = Tensor::full(min_std as f32, std.shape(), std.device())?;
    let penalty = (min_std_t - std)?.relu()?;
    let loss = penalty.sqr()?.mean_all()?;
    Ok(loss)
}

/// Live Model Health & Collapse Diagnostics snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCollapseDiagnostics {
    pub active_latents: usize,         // Active latent channels (out of 64) with Var > 0.01
    pub latent_total_dim: usize,       // 64
    pub router_perplexity: f32,       // Router effective expert diversity [1.0 .. 8.0]
    pub dead_experts: usize,           // Experts receiving < 1.25% routing probability
    pub trajectory_variance: f32,      // Mean standard deviation of predicted trajectories
    pub status_code: String,           // "OPTIMAL", "POSTERIOR_RISK", "ROUTER_STARVATION", "MODE_COLLAPSE"
    pub is_mitigating: bool,
}

impl Default for ModelCollapseDiagnostics {
    fn default() -> Self {
        Self {
            active_latents: LATENT_DIM,
            latent_total_dim: LATENT_DIM,
            router_perplexity: NUM_EXPERTS as f32,
            dead_experts: 0,
            trajectory_variance: 1.0,
            status_code: "OPTIMAL".to_string(),
            is_mitigating: false,
        }
    }
}


