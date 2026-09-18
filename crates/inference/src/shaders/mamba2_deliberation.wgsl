// mamba2_deliberation.wgsl - In-VRAM Multi-Step Latent Deliberation Thinking Loop
// Unrolls K iterative recurrence thinking steps (K in [1, 5]) inside a single GPU kernel.
// Keeps latent states in fast GPU LDS / registers without CPU-GPU synchronization between steps.

const LATENT_DIM: u32 = 64u;

struct DeliberationParams {
    thinking_steps: u32,  // Number of deliberation passes (1 to 5)
    decay_rate: f32,      // Recurrent state decay scaling
    step_size: f32,       // Homotopy integration step size
    reserved: u32,
}

@group(0) @binding(0) var<storage, read_write> latent_state: array<f32>; // 64-element latent state s_t
@group(0) @binding(1) var<storage, read> a_diag: array<f32>;             // 64-element diagonal state transition A
@group(0) @binding(2) var<storage, read> b_diag: array<f32>;             // 64-element input projection B
@group(0) @binding(3) var<storage, read> u_t: array<f32>;                // 64-element driving input u_t
@group(0) @binding(4) var<uniform> params: DeliberationParams;

var<workgroup> shared_state: array<f32, 64>;

@compute @workgroup_size(64)
fn deliberate_recurrence(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = local_id.x;
    if idx >= LATENT_DIM {
        return;
    }

    // Load initial state into workgroup LDS
    shared_state[idx] = latent_state[idx];
    workgroupBarrier();

    let a = a_diag[idx] * params.decay_rate;
    let b = b_diag[idx] * params.step_size;
    let u = u_t[idx];

    let steps = clamp(params.thinking_steps, 1u, 5u);

    // Unroll K iterative latent thinking steps in fast on-chip memory
    for (var k = 0u; k < steps; k = k + 1u) {
        let s_curr = shared_state[idx];
        let s_next = fma(b, u, a * s_curr);
        
        workgroupBarrier();
        shared_state[idx] = s_next;
        workgroupBarrier();
    }

    // Write final refined latent state back to storage buffer
    latent_state[idx] = shared_state[idx];
}