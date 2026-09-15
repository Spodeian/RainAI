struct LayerParams {
    in_dim: u32,
    out_dim: u32,
    has_bias: u32,
    padding: u32, // 16-byte uniform alignment requirement
}

@group(0) @binding(0) var<storage, read> input_vec: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(3) var<storage, read> bias_vec: array<f32>;
@group(0) @binding(4) var<uniform> params: LayerParams;

@compute @workgroup_size(64)
fn dense_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if out_idx < params.out_dim {
        var sum = 0.0;
        let w_offset = out_idx * params.in_dim;
        
        for (var i = 0u; i < params.in_dim; i = i + 1u) {
            sum = fma(weights[w_offset + i], input_vec[i], sum);
        }
        
        if params.has_bias != 0u {
            sum = sum + bias_vec[out_idx];
        }
        
        output_vec[out_idx] = sum;
    }
}
