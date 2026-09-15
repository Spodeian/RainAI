//! SIMD-vectorized rational approximations for non-linear activations.

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn simd_silu_in_place(x: &mut [f32]) {
    for val in x.iter_mut() {
        // Fast algebraic SiLU: x * 0.5 * (1 + (x / (1 + |x|)))
        let sig = 0.5 * (1.0 + (*val / (1.0 + val.abs())));
        *val *= sig;
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub fn simd_silu_in_place(x: &mut [f32]) {
    use std::arch::wasm32::*;
    
    let len = x.len();
    let iters = len / 4;
    
    let half_v = f32x4_splat(0.5);
    let one_v = f32x4_splat(1.0);
    
    unsafe {
        for i in 0..iters {
            let ptr = x.as_mut_ptr().add(i * 4) as *mut v128;
            let val = v128_load(ptr as *const v128);
            
            let abs_val = f32x4_abs(val);
            let denom = f32x4_add(one_v, abs_val);
            let frac = f32x4_div(val, denom);
            let sig = f32x4_mul(half_v, f32x4_add(one_v, frac));
            
            let result = f32x4_mul(val, sig);
            v128_store(ptr, result);
        }
    }
    
    for i in (iters * 4)..len {
        let val = x[i];
        let sig = 0.5 * (1.0 + (val / (1.0 + val.abs())));
        x[i] = val * sig;
    }
}
