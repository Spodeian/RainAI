//! Real-time allocation-free Mixture of Experts (MoE) dispatch and state decay.

pub const MAX_SUPPORTED_EXPERTS: usize = 16;

#[cfg(not(target_arch = "wasm32"))]
pub fn route_and_decay(
    latent_state: &mut [f32],
    router_weights: &[f32],
    num_experts: usize,
    tau_moe: f32,
    decay_factor: f32,
) {
    let latent_dim = latent_state.len();
    let num_exp = num_experts.min(MAX_SUPPORTED_EXPERTS);
    let mut logits = [0.0f32; MAX_SUPPORTED_EXPERTS];
    let mut max_logit = f32::NEG_INFINITY;

    // 1. Calculate router logits
    for exp in 0..num_exp {
        let offset = exp * latent_dim;
        let w_slice = &router_weights[offset..offset + latent_dim];
        let sum: f32 = w_slice
            .iter()
            .zip(latent_state.iter())
            .map(|(&w, &s)| w * s)
            .sum();
        logits[exp] = sum;
        if sum > max_logit {
            max_logit = sum;
        }
    }

    // 2. Numerically stable, temperature-scaled continuous softmax routing
    let tau = tau_moe.max(0.05);
    let mut sum_exp = 0.0f32;
    let mut exps = [0.0f32; MAX_SUPPORTED_EXPERTS];
    for exp in 0..num_exp {
        let e = ((logits[exp] - max_logit) / tau).exp();
        exps[exp] = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-8);
    let mut weights = [0.0f32; MAX_SUPPORTED_EXPERTS];
    for exp in 0..num_exp {
        weights[exp] = exps[exp] * inv_sum;
    }

    // 3. Continuous Hermite C^1 smoothstep state retention across all experts
    // All experts participate smoothly — zero discrete winner-take-all switching
    let chunk_size = latent_dim / num_experts;
    for exp in 0..num_exp {
        let p = weights[exp];
        let norm_p = (p * num_exp as f32 * 0.5).clamp(0.0, 1.0);
        let smooth_factor = norm_p * norm_p * (3.0 - 2.0 * norm_p); // Hermite C1 smoothstep
        let retention = decay_factor + smooth_factor * (1.0 - decay_factor);

        let start = exp * chunk_size;
        for val in &mut latent_state[start..start + chunk_size] {
            *val *= retention;
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn route_and_decay(
    latent_state: &mut [f32],
    router_weights: &[f32],
    num_experts: usize,
    tau_moe: f32,
    decay_factor: f32,
) {
    use std::arch::wasm32::*;

    let latent_dim = latent_state.len();
    let num_exp = num_experts.min(MAX_SUPPORTED_EXPERTS);
    let mut logits = [0.0f32; MAX_SUPPORTED_EXPERTS];
    let mut max_logit = f32::NEG_INFINITY;

    // 1. Vectorized logit dot products
    let iters_4 = latent_dim / 4;
    for exp in 0..num_exp {
        let offset = exp * latent_dim;
        let mut acc = f32x4_splat(0.0);

        unsafe {
            for j in 0..iters_4 {
                let j_offset = j * 4;
                let w = v128_load(router_weights.as_ptr().add(offset + j_offset) as *const v128);
                let s = v128_load(latent_state.as_ptr().add(j_offset) as *const v128);
                acc = f32x4_add(acc, f32x4_mul(w, s));
            }
        }

        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        for j in (iters_4 * 4)..latent_dim {
            sum += router_weights[offset + j] * latent_state[j];
        }
        logits[exp] = sum;
        if sum > max_logit {
            max_logit = sum;
        }
    }

    // 2. Numerically stable, temperature-scaled continuous softmax
    let tau = tau_moe.max(0.05);
    let mut sum_exp = 0.0f32;
    let mut exps = [0.0f32; MAX_SUPPORTED_EXPERTS];
    for exp in 0..num_exp {
        let e = ((logits[exp] - max_logit) / tau).exp();
        exps[exp] = e;
        sum_exp += e;
    }

    let inv_sum = 1.0 / sum_exp.max(1e-8);
    let mut weights = [0.0f32; MAX_SUPPORTED_EXPERTS];
    for exp in 0..num_exp {
        weights[exp] = exps[exp] * inv_sum;
    }

    // 3. Continuous Hermite C^1 smoothstep state retention with SIMD
    let chunk_size = latent_dim / num_experts;

    for exp in 0..num_exp {
        let p = weights[exp];
        let norm_p = (p * num_exp as f32 * 0.5).clamp(0.0, 1.0);
        let smooth_factor = norm_p * norm_p * (3.0 - 2.0 * norm_p);
        let retention = decay_factor + smooth_factor * (1.0 - decay_factor);

        let retention_v = f32x4_splat(retention);
        let start = exp * chunk_size;
        let slice = &mut latent_state[start..start + chunk_size];
        let chunk_iters = slice.len() / 4;

        unsafe {
            for c in 0..chunk_iters {
                let ptr = slice.as_mut_ptr().add(c * 4) as *mut v128;
                let v = v128_load(ptr as *const v128);
                v128_store(ptr, f32x4_mul(v, retention_v));
            }
        }

        for val in &mut slice[(chunk_iters * 4)..] {
            *val *= retention;
        }
    }
}
