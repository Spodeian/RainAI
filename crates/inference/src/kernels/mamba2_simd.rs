//! Vectorized Mamba2 State-Space Recurrence Kernel: s_t = A * s_{t-1} + B * u_t.

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn step_recurrence_f32(
    latent_state: &mut [f32],
    a_diag: &[f32],
    b_diag: &[f32],
    u_t: &[f32],
) {
    let len = latent_state.len();
    assert!(
        a_diag.len() >= len && b_diag.len() >= len && u_t.len() >= len,
        "Mamba2 recurrence buffer dimension mismatch"
    );

    // Idiomatic zip allows LLVM to prove bounds and emit AVX2/AVX-512 FMA instructions
    for (((s, &a), &b), &u) in latent_state
        .iter_mut()
        .zip(a_diag.iter())
        .zip(b_diag.iter())
        .zip(u_t.iter())
    {
        *s = a * *s + b * u;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub fn step_recurrence_f32(
    latent_state: &mut [f32],
    a_diag: &[f32],
    b_diag: &[f32],
    u_t: &[f32],
) {
    use std::arch::wasm32::*;

    let len = latent_state.len();
    assert!(
        a_diag.len() >= len && b_diag.len() >= len && u_t.len() >= len,
        "Mamba2 recurrence buffer dimension mismatch"
    );

    // Dual-register unrolling (8 floats / 2x v128 per iteration)
    let iters_8 = len / 8;
    for i in 0..iters_8 {
        let offset = i * 8;
        unsafe {
            let s0 = v128_load(latent_state.as_ptr().add(offset) as *const v128);
            let s1 = v128_load(latent_state.as_ptr().add(offset + 4) as *const v128);
            let a0 = v128_load(a_diag.as_ptr().add(offset) as *const v128);
            let a1 = v128_load(a_diag.as_ptr().add(offset + 4) as *const v128);
            let b0 = v128_load(b_diag.as_ptr().add(offset) as *const v128);
            let b1 = v128_load(b_diag.as_ptr().add(offset + 4) as *const v128);
            let u0 = v128_load(u_t.as_ptr().add(offset) as *const v128);
            let u1 = v128_load(u_t.as_ptr().add(offset + 4) as *const v128);

            let next_s0 = f32x4_add(f32x4_mul(a0, s0), f32x4_mul(b0, u0));
            let next_s1 = f32x4_add(f32x4_mul(a1, s1), f32x4_mul(b1, u1));

            v128_store(latent_state.as_mut_ptr().add(offset) as *mut v128, next_s0);
            v128_store(latent_state.as_mut_ptr().add(offset + 4) as *mut v128, next_s1);
        }
    }

    // Handle remaining 4-lane chunk
    let rem_4_offset = iters_8 * 8;
    if len - rem_4_offset >= 4 {
        unsafe {
            let s = v128_load(latent_state.as_ptr().add(rem_4_offset) as *const v128);
            let a = v128_load(a_diag.as_ptr().add(rem_4_offset) as *const v128);
            let b = v128_load(b_diag.as_ptr().add(rem_4_offset) as *const v128);
            let u = v128_load(u_t.as_ptr().add(rem_4_offset) as *const v128);

            let next_s = f32x4_add(f32x4_mul(a, s), f32x4_mul(b, u));
            v128_store(latent_state.as_mut_ptr().add(rem_4_offset) as *mut v128, next_s);
        }
    }

    // Scalar cleanup
    for i in ((len / 4) * 4)..len {
        latent_state[i] = a_diag[i] * latent_state[i] + b_diag[i] * u_t[i];
    }
}
