//! Fast matrix multiplication for Bfloat16 (BF16) weights via 1-cycle bit-shift upcast.

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn bf16_matmul_simd_f32(
    raw_weights: &[u16],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    assert!(
        raw_weights.len() >= output.len() * in_dim,
        "BF16 weight slice dimensions insufficient for projection"
    );

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &raw_weights[w_offset..w_offset + in_dim];
        let mut sum = 0.0f32;

        let iters_4 = in_dim / 4;
        for j in 0..iters_4 {
            let idx = j * 4;
            let w0 = f32::from_bits((w_row[idx] as u32) << 16);
            let w1 = f32::from_bits((w_row[idx + 1] as u32) << 16);
            let w2 = f32::from_bits((w_row[idx + 2] as u32) << 16);
            let w3 = f32::from_bits((w_row[idx + 3] as u32) << 16);

            sum += w0 * activations[idx]
                + w1 * activations[idx + 1]
                + w2 * activations[idx + 2]
                + w3 * activations[idx + 3];
        }

        for j in (iters_4 * 4)..in_dim {
            let w = f32::from_bits((w_row[j] as u32) << 16);
            sum += w * activations[j];
        }

        *out_val = sum * scale;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn bf16_matmul_simd_f32(
    raw_weights: &[u16],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    use std::arch::wasm32::*;

    let in_dim = activations.len();
    assert!(
        raw_weights.len() >= output.len() * in_dim,
        "BF16 weight slice dimensions insufficient for projection"
    );
    let iters_4 = in_dim / 4;

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &raw_weights[w_offset..w_offset + in_dim];
        let mut acc = f32x4_splat(0.0);

        unsafe {
            for j in 0..iters_4 {
                let idx = j * 4;
                let w0 = f32::from_bits((w_row[idx] as u32) << 16);
                let w1 = f32::from_bits((w_row[idx + 1] as u32) << 16);
                let w2 = f32::from_bits((w_row[idx + 2] as u32) << 16);
                let w3 = f32::from_bits((w_row[idx + 3] as u32) << 16);
                let w_v = f32x4(w0, w1, w2, w3);

                let act_v = v128_load(activations.as_ptr().add(idx) as *const v128);
                acc = f32x4_add(acc, f32x4_mul(w_v, act_v));
            }
        }

        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        for j in (iters_4 * 4)..in_dim {
            let w = f32::from_bits((w_row[j] as u32) << 16);
            sum += w * activations[j];
        }

        *out_val = sum * scale;
    }
}
