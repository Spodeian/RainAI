//! Fused unpack-and-accumulate matrix multiplication for 2-bit packed ternary weights.

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn ternary_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    gamma: f32,
) {
    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(4);

    for (i, out_val) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        let w_offset = i * bytes_per_row;
        let row_bytes = &packed_weights[w_offset..w_offset + bytes_per_row];

        for (b_idx, &byte) in row_bytes.iter().enumerate() {
            let act_offset = b_idx * 4;
            
            for bit_shift in 0..4 {
                let act_idx = act_offset + bit_shift;
                if act_idx >= in_dim {
                    break;
                }
                
                let code = (byte >> (bit_shift * 2)) & 0x03;
                let w = match code {
                    0x01 => 1.0,
                    0x03 => -1.0,
                    _ => 0.0,
                };
                sum += w * activations[act_idx];
            }
        }
        *out_val = sum * gamma;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn ternary_matmul_simd_f32(
    packed_weights: &[u8],
    activations: &[f32],
    output: &mut [f32],
    gamma: f32,
) {
    use std::arch::wasm32::*;

    let in_dim = activations.len();
    let bytes_per_row = in_dim.div_ceil(4);
    let _gamma_v = f32x4_splat(gamma);

    for (i, out_val) in output.iter_mut().enumerate() {
        let w_offset = i * bytes_per_row;
        let mut sum_v = f32x4_splat(0.0);

        // Process 4 weights (1 byte) and 4 floats (16 bytes) per iteration
        let iters = in_dim / 4;
        unsafe {
            for j in 0..iters {
                let byte = packed_weights[w_offset + j];
                let act_v = v128_load(activations.as_ptr().add(j * 4) as *const v128);
                
                // Manually unroll the byte extraction to avoid branching in the hot loop
                let w0 = match byte & 0x03 { 0x01 => 1.0, 0x03 => -1.0, _ => 0.0 };
                let w1 = match (byte >> 2) & 0x03 { 0x01 => 1.0, 0x03 => -1.0, _ => 0.0 };
                let w2 = match (byte >> 4) & 0x03 { 0x01 => 1.0, 0x03 => -1.0, _ => 0.0 };
                let w3 = match (byte >> 6) & 0x03 { 0x01 => 1.0, 0x03 => -1.0, _ => 0.0 };
                
                let w_v = f32x4(w0, w1, w2, w3);
                
                sum_v = f32x4_add(sum_v, f32x4_mul(w_v, act_v));
            }
        }

        let mut total = f32x4_extract_lane::<0>(sum_v)
            + f32x4_extract_lane::<1>(sum_v)
            + f32x4_extract_lane::<2>(sum_v)
            + f32x4_extract_lane::<3>(sum_v);

        // Handle remainder for dimensions not divisible by 4
        if in_dim % 4 != 0 {
            let byte = packed_weights[w_offset + iters];
            for bit_shift in 0..(in_dim % 4) {
                let act_idx = iters * 4 + bit_shift;
                let code = (byte >> (bit_shift * 2)) & 0x03;
                let w = match code { 0x01 => 1.0, 0x03 => -1.0, _ => 0.0 };
                total += w * activations[act_idx];
            }
        }

        *out_val = total * gamma;
    }
}
