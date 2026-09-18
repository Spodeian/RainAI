// Mamba-2 State Space Duality (SSD) Parallel Associative Recurrence Kernel
// Optimized for WebGPU compute shader pipeline (workgroup size: 64)

struct MambaParams {
    d_model: u32,
    d_state: u32,
    decay_scale: f32,
    momentum_alpha: f32,
};

@group(0) @binding(0) var<uniform> params: MambaParams;
@group(0) @binding(1) var<storage, read> u_input: array<f32>;
@group(0) @binding(2) var<storage, read> a_diag: array<f32>;
@group(0) @binding(3) var<storage, read> b_diag: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_state: array<f32>;
@group(0) @binding(5) var<storage, read_write> y_output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.d_model) {
        return;
    }

    let u_val = u_input[idx];
    let a_val = a_diag[idx] * params.decay_scale;
    let b_val = b_diag[idx];

    // Read previous state
    let prev_s = s_state[idx];

    // Compute updated state: s_next = exp(-exp(A_log)) * s_prev + B * u
    let s_next = a_val * prev_s + b_val * u_val;

    // Apply SSM momentum damping
    let alpha = params.momentum_alpha;
    let s_damped = alpha * prev_s + (1.0 - alpha) * s_next;

    // Write back updated state and output
    s_state[idx] = s_damped;
    y_output[idx] = s_damped;
}
