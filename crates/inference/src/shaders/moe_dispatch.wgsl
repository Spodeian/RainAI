struct MoeParams {
    num_experts: u32,
    latent_dim: u32,
    decay_factor: f32,
    tau_moe: f32, // Continuous temperature for smooth softmax routing (default 0.75)
}

@group(0) @binding(0) var<storage, read_write> latent_state: array<f32>;
@group(0) @binding(1) var<storage, read> router_weights: array<f32>;
@group(0) @binding(2) var<uniform> params: MoeParams;

// Workgroup shared memory for continuous expert routing probabilities
var<workgroup> shared_weights: array<f32, 8>;

@compute @workgroup_size(64)
fn route_and_decay(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;

    // Leader thread computes continuous routing logits and temperature-scaled smooth softmax
    if tid == 0u {
        var logits: array<f32, 8>;
        var max_logit = -3.402823e+38;

        for (var e = 0u; e < params.num_experts; e = e + 1u) {
            var sum = 0.0;
            let offset = e * params.latent_dim;
            for (var i = 0u; i < params.latent_dim; i = i + 1u) {
                sum = fma(router_weights[offset + i], latent_state[i], sum);
            }
            logits[e] = sum;
            if sum > max_logit {
                max_logit = sum;
            }
        }

        // Temperature scaling with numerical stability clamp
        let tau = max(params.tau_moe, 0.05);
        var sum_exp = 0.0;
        var exps: array<f32, 8>;

        for (var e = 0u; e < params.num_experts; e = e + 1u) {
            let exp_val = exp((logits[e] - max_logit) / tau);
            exps[e] = exp_val;
            sum_exp = sum_exp + exp_val;
        }

        let inv_sum = 1.0 / max(sum_exp, 1e-8);
        for (var e = 0u; e < params.num_experts; e = e + 1u) {
            shared_weights[e] = exps[e] * inv_sum;
        }
    }

    // Synchronize workgroup execution so all threads see shared_weights
    workgroupBarrier();

    // Parallel smooth expert latent modulation across all SIMD lanes
    if tid < params.latent_dim {
        let chunk_size = params.latent_dim / params.num_experts;
        let expert_idx = tid / chunk_size;

        // Continuous smooth retention mapping:
        // Experts with higher routing probability maintain full state (retention -> 1.0),
        // while low-affinity experts smoothly decay towards decay_factor without discrete switching chatter.
        let p = shared_weights[expert_idx];
        let norm_p = clamp(p * f32(params.num_experts) * 0.5, 0.0, 1.0);
        let smooth_factor = norm_p * norm_p * (3.0 - 2.0 * norm_p); // Hermite C1 smoothstep
        let retention = mix(params.decay_factor, 1.0, smooth_factor);

        latent_state[tid] = latent_state[tid] * retention;
    }
}

