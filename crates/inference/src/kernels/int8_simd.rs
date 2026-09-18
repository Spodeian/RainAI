//! Vectorized matrix multiplication for 8-bit signed integer (INT8) weights.

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn int8_matmul_simd_f32(
    weights: &[i8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let in_dim = activations.len();
    assert!(
        weights.len() >= output.len() * in_dim,
        "INT8 weight slice dimensions insufficient for projection"
    );
    let inv_127 = scale / 127.0;

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &weights[w_offset..w_offset + in_dim];
        let mut sum = 0.0f32;

        let iters_4 = in_dim / 4;
        for j in 0..iters_4 {
            let idx = j * 4;
            sum += (w_row[idx] as f32) * activations[idx]
                + (w_row[idx + 1] as f32) * activations[idx + 1]
                + (w_row[idx + 2] as f32) * activations[idx + 2]
                + (w_row[idx + 3] as f32) * activations[idx + 3];
        }

        for j in (iters_4 * 4)..in_dim {
            sum += (w_row[j] as f32) * activations[j];
        }

        *out_val = sum * inv_127;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn int8_matmul_simd_f32(
    weights: &[i8],
    activations: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    use std::arch::wasm32::*;

    let in_dim = activations.len();
    assert!(
        weights.len() >= output.len() * in_dim,
        "INT8 weight slice dimensions insufficient for projection"
    );
    let inv_127 = scale / 127.0;
    let iters_4 = in_dim / 4;

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &weights[w_offset..w_offset + in_dim];
        let mut acc = f32x4_splat(0.0);

        unsafe {
            for j in 0..iters_4 {
                let idx = j * 4;
                let w_v = f32x4(
                    w_row[idx] as f32,
                    w_row[idx + 1] as f32,
                    w_row[idx + 2] as f32,
                    w_row[idx + 3] as f32,
                );
                let act_v = v128_load(activations.as_ptr().add(idx) as *const v128);
                acc = f32x4_add(acc, f32x4_mul(w_v, act_v));
            }
        }

        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        for j in (iters_4 * 4)..in_dim {
            sum += (w_row[j] as f32) * activations[j];
        }

        *out_val = sum * inv_127;
    }
}
