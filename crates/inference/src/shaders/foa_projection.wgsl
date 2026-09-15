struct FoaParams {
    latent_dim: u32,
    bit_scale: f32,
    padding: vec2<u32>, // 16-byte uniform alignment requirement
}

@group(0) @binding(0) var<storage, read> latent_state: array<f32>;
@group(0) @binding(1) var<storage, read> foa_weights: array<f32>; // 4 x 64 matrix
@group(0) @binding(2) var<storage, read_write> foa_out: array<f32>; // 4 channels: W, X, Y, Z
@group(0) @binding(3) var<uniform> params: FoaParams;

@compute @workgroup_size(4)
fn project_foa(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let ch = local_id.x; // 0: W, 1: X, 2: Y, 3: Z
    if ch < 4u {
        var sum = 0.0;
        let offset = ch * params.latent_dim;
        for (var i = 0u; i < params.latent_dim; i = i + 1u) {
            sum = fma(foa_weights[offset + i], latent_state[i], sum);
        }
        foa_out[ch] = sum * params.bit_scale;
    }
}
