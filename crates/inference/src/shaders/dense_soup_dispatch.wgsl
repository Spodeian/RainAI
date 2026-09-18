// Dense Soup Single-Pass Matrix-Vector Dispatch Shader
// Evaluates W_soup @ x + bias without sparse MoE routing bubbles.

struct DenseSoupParams {
    in_dim: u32,
    out_dim: u32,
    scale: f32,
    has_bias: u32,
};

@group(0) @binding(0) var<uniform> params: DenseSoupParams;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read> soup_weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias_vec: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.out_dim) {
        return;
    }

    var acc: f32 = 0.0;
    let weight_offset = row * params.in_dim;
    let vec4_chunks: u32 = params.in_dim / 4u;

    // Vectorized 4-wide SIMD dot product
    for (var c: u32 = 0u; c < vec4_chunks; c = c + 1u) {
        let col = c * 4u;
        let w = vec4<f32>(
            soup_weights[weight_offset + col],
            soup_weights[weight_offset + col + 1u],
            soup_weights[weight_offset + col + 2u],
            soup_weights[weight_offset + col + 3u]
        );
        let x = vec4<f32>(
            input_vec[col],
            input_vec[col + 1u],
            input_vec[col + 2u],
            input_vec[col + 3u]
        );
        acc = acc + dot(w, x);
    }

    // Remainder scalar cleanup
    for (var col: u32 = vec4_chunks * 4u; col < params.in_dim; col = col + 1u) {
        let w = soup_weights[weight_offset + col];
        let x = input_vec[col];
        acc = acc + w * x;
    }

    acc = acc * params.scale;

    if (params.has_bias != 0u) {
        acc = acc + bias_vec[row];
    }

    output_vec[row] = acc;
}
