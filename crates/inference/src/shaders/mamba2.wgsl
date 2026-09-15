const LATENT_DIM: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> latent_state: array<f32>;
@group(0) @binding(1) var<storage, read> a_diag: array<f32>;
@group(0) @binding(2) var<storage, read> b_diag: array<f32>;
@group(0) @binding(3) var<storage, read> u_t: array<f32>;

@compute @workgroup_size(64)
fn step_recurrence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx < LATENT_DIM {
        let s_prev = latent_state[idx];
        let a = a_diag[idx];
        let b = b_diag[idx];
        let u = u_t[idx];
        
        // Fused multiply-accumulate: (b * u) + (a * s_prev)
        latent_state[idx] = fma(b, u, a * s_prev);
    }
}
