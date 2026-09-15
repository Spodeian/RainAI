//! Real-time allocation-free Mixture of Experts (MoE) dispatch and state decay.

pub const MAX_SUPPORTED_EXPERTS: usize = 16;

#[cfg(not(target_arch = "wasm32"))]
pub fn route_and_decay(
    latent_state: &mut [f32],
    router_weights: &[f32],
    num_experts: usize,
    top_k: usize,
    decay_factor: f32,
) {
    let latent_dim = latent_state.len();
    let num_exp = num_experts.min(MAX_SUPPORTED_EXPERTS);
    let mut logits = [0.0f32; MAX_SUPPORTED_EXPERTS];

    // 1. Calculate router logits on stack
    for exp in 0..num_exp {
        let offset = exp * latent_dim;
        let w_slice = &router_weights[offset..offset + latent_dim];
        logits[exp] = w_slice
            .iter()
            .zip(latent_state.iter())
            .map(|(&w, &s)| w * s)
            .sum();
    }

    // 2. Allocation-free partial top-K selection (O(K * E), typically 2 * 8 = 16 ops)
    let k = top_k.min(num_exp);
    let mut selected_indices = [usize::MAX; 4];

    for rank in 0..k.min(4) {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0;
        for exp in 0..num_exp {
            if !selected_indices[..rank].contains(&exp) && logits[exp] > best_val {
                best_val = logits[exp];
                best_idx = exp;
            }
        }
        selected_indices[rank] = best_idx;
    }

    // 3. Decay latents of unselected experts
    let chunk_size = latent_dim / num_experts;
    for exp in 0..num_exp {
        if !selected_indices[..k.min(4)].contains(&exp) {
            let start = exp * chunk_size;
            for val in &mut latent_state[start..start + chunk_size] {
                *val *= decay_factor;
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn route_and_decay(
    latent_state: &mut [f32],
    router_weights: &[f32],
    num_experts: usize,
    top_k: usize,
    decay_factor: f32,
) {
    use std::arch::wasm32::*;

    let latent_dim = latent_state.len();
    let num_exp = num_experts.min(MAX_SUPPORTED_EXPERTS);
    let mut logits = [0.0f32; MAX_SUPPORTED_EXPERTS];

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
    }

    // 2. Allocation-free partial top-K selection
    let k = top_k.min(num_exp);
    let mut selected_indices = [usize::MAX; 4];

    for rank in 0..k.min(4) {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0;
        for exp in 0..num_exp {
            if !selected_indices[..rank].contains(&exp) && logits[exp] > best_val {
                best_val = logits[exp];
                best_idx = exp;
            }
        }
        selected_indices[rank] = best_idx;
    }

    // 3. SIMD-vectorized state decay for unselected experts
    let chunk_size = latent_dim / num_experts;
    let decay_v = f32x4_splat(decay_factor);

    for exp in 0..num_exp {
        if !selected_indices[..k.min(4)].contains(&exp) {
            let start = exp * chunk_size;
            let slice = &mut latent_state[start..start + chunk_size];
            let chunk_iters = slice.len() / 4;

            unsafe {
                for c in 0..chunk_iters {
                    let ptr = slice.as_mut_ptr().add(c * 4) as *mut v128;
                    let v = v128_load(ptr as *const v128);
                    v128_store(ptr, f32x4_mul(v, decay_v));
                }
            }

            for val in &mut slice[(chunk_iters * 4)..] {
                *val *= decay_factor;
            }
        }
    }
}
