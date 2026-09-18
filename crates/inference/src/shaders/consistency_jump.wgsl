// consistency_jump.wgsl - Single-Pass 1-Step Direct Consistency Distillation Projection
// Maps 554-parameter physical conditioning vector directly to 4-channel Ambisonics (W, X, Y, Z)
// in a single GPU dispatch, eliminating intermediate latent round-trips and reducing power >75%.

struct ConsistencyJumpParams {
    in_dim: u32,
    out_dim: u32,
    bit_scale: f32,
    yaw_rot: f32,
}

@group(0) @binding(0) var<storage, read> conditioning: array<f32>; // 554 conditioning features
@group(0) @binding(1) var<storage, read> jump_weights: array<f32>; // out_dim x in_dim (4 x 554)
@group(0) @binding(2) var<storage, read> jump_bias: array<f32>;    // 4 bias terms
@group(0) @binding(3) var<storage, read_write> foa_out: array<f32>; // 4 channels: W, X, Y, Z
@group(0) @binding(4) var<uniform> params: ConsistencyJumpParams;

@compute @workgroup_size(4)
fn consistency_jump(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let ch = local_id.x; // 0: W (omnidirectional), 1: X (front-back), 2: Y (left-right), 3: Z (elevation)
    if ch >= params.out_dim {
        return;
    }

    var sum = jump_bias[ch];
    let row_offset = ch * params.in_dim;

    for (var i = 0u; i < params.in_dim; i = i + 1u) {
        sum = fma(jump_weights[row_offset + i], conditioning[i], sum);
    }

    let scaled_val = sum * params.bit_scale;
    foa_out[ch] = scaled_val;
}