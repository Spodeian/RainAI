//! Cache-aligned dense projection matrix multiplication for input and FOA output layers.

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn dense_projection(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let in_dim = input.len();
    assert!(
        weights.len() >= output.len() * in_dim,
        "Weight matrix dimensions insufficient for projection"
    );

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let w_row = &weights[w_offset..w_offset + in_dim];
        let dot: f32 = w_row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum();
        *out_val = bias.map_or(0.0, |b| b[i]) + dot;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn dense_projection(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use std::arch::wasm32::*;

    let in_dim = input.len();
    assert!(
        weights.len() >= output.len() * in_dim,
        "Weight matrix dimensions insufficient for projection"
    );

    let iters_8 = in_dim / 8;
    let has_remainder_4 = (in_dim % 8) >= 4;

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * in_dim;
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);

        unsafe {
            // Dual accumulator unrolling (8 floats per step)
            for j in 0..iters_8 {
                let j_offset = j * 8;
                let w0 = v128_load(weights.as_ptr().add(w_offset + j_offset) as *const v128);
                let x0 = v128_load(input.as_ptr().add(j_offset) as *const v128);
                let w1 = v128_load(weights.as_ptr().add(w_offset + j_offset + 4) as *const v128);
                let x1 = v128_load(input.as_ptr().add(j_offset + 4) as *const v128);

                acc0 = f32x4_add(acc0, f32x4_mul(w0, x0));
                acc1 = f32x4_add(acc1, f32x4_mul(w1, x1));
            }

            acc0 = f32x4_add(acc0, acc1);

            let rem_8_offset = iters_8 * 8;
            if has_remainder_4 {
                let w = v128_load(weights.as_ptr().add(w_offset + rem_8_offset) as *const v128);
                let x = v128_load(input.as_ptr().add(rem_8_offset) as *const v128);
                acc0 = f32x4_add(acc0, f32x4_mul(w, x));
            }
        }

        let mut total = f32x4_extract_lane::<0>(acc0)
            + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0)
            + f32x4_extract_lane::<3>(acc0);

        // Process final 1 to 3 elements
        let processed = (iters_8 * 8) + if has_remainder_4 { 4 } else { 0 };
        for j in processed..in_dim {
            total += weights[w_offset + j] * input[j];
        }

        *out_val = bias.map_or(0.0, |b| b[i]) + total;
    }
}
