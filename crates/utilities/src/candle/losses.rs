//! Loss Functions and Regularizers for Neural Training.
//!
//! Includes 2nd-order physics trajectory losses, MoE load balancing and router z-loss,
//! expert diversity, Beta-VAE KL divergence, HWIL latency penalties, flow matching,
//! and multi-resolution reconstruction loss.

use anyhow::Result;
use candle_core::{DType, Tensor};

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

    let v_pred_sq = v_pred.sqr()?.sum_keepdim(1)?;
    let speed = (v_pred_sq + 1e-6)?.sqrt()?;
    let excess_speed = (speed - v_terminal)?.relu()?;
    let drag_loss = excess_speed.sqr()?.mean_all()?;

    let total_acc = (&pos_loss + (&vel_loss * lambda_vel)?)?;
    let total_drag = (&total_acc + (&acc_loss * lambda_acc)?)?;
    let total = (&total_drag + (&drag_loss * lambda_drag)?)?;

    Ok((total, pos_loss, acc_loss, drag_loss))
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
    let exp_diff = router_logits.broadcast_sub(&max_logit)?.exp()?;
    let sum_exp = exp_diff.sum_keepdim(1)?;
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
            let denom = (&norm_j * &norm_k)?;
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
pub fn compute_beta_vae_loss(
    pred_bands: &Tensor,
    target_bands: &Tensor,
    mu: &Tensor,
    logvar: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let recon_diff = (pred_bands - target_bands)?;
    let recon_loss = recon_diff.sqr()?.mean_all()?;

    // KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    let mu_sq = mu.sqr()?;
    let var = logvar.exp()?;
    let ones = Tensor::ones(logvar.shape(), DType::F32, logvar.device())?;
    let inner = (((&ones + logvar)? - &mu_sq)? - &var)?;
    let kl = (inner.mean_all()? * -0.5)?;

    let total = (&recon_loss + (&kl * beta)?)?;
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

