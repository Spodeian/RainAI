struct MoeParams {
    num_experts: u32,
    latent_dim: u32,
    decay_factor: f32,
    padding: u32, // 16-byte uniform alignment requirement
}

@group(0) @binding(0) var<storage, read_write> latent_state: array<f32>;
@group(0) @binding(1) var<storage, read> router_weights: array<f32>;
@group(0) @binding(2) var<uniform> params: MoeParams;

var<workgroup> shared_top1: u32;
var<workgroup> shared_top2: u32;

@compute @workgroup_size(64)
fn route_and_decay(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;

    // Leader thread computes logits and determines top-2 routing indices
    if tid == 0u {
        var logits: array<f32, 8>;
        for (var e = 0u; e < params.num_experts; e = e + 1u) {
            var sum = 0.0;
            let offset = e * params.latent_dim;
            for (var i = 0u; i < params.latent_dim; i = i + 1u) {
                sum = fma(router_weights[offset + i], latent_state[i], sum);
            }
            logits[e] = sum;
        }

        var max1_val = -3.402823e+38;
        var max1_idx = 0u;
        var max2_val = -3.402823e+38;
        var max2_idx = 0u;

        for (var e = 0u; e < params.num_experts; e = e + 1u) {
            let val = logits[e];
            if val > max1_val {
                max2_val = max1_val;
                max2_idx = max1_idx;
                max1_val = val;
                max1_idx = e;
            } else if val > max2_val {
                max2_val = val;
                max2_idx = e;
            }
        }

        shared_top1 = max1_idx;
        shared_top2 = max2_idx;
    }

    // Synchronize workgroup execution
    workgroupBarrier();

    // Parallel expert latent decay across all 64 SIMD lanes
    if tid < params.latent_dim {
        let chunk_size = params.latent_dim / params.num_experts;
        let expert_idx = tid / chunk_size;

        if expert_idx != shared_top1 && expert_idx != shared_top2 {
            latent_state[tid] = latent_state[tid] * params.decay_factor;
        }
    }
}
