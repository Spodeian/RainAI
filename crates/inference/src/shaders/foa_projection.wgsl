struct FoaParams {
    latent_dim: u32,
    bit_scale: f32,
    yaw_rad: f32,
    pitch_rad: f32,
}

@group(0) @binding(0) var<storage, read> latent_state: array<f32>;
@group(0) @binding(1) var<storage, read> foa_weights: array<f32>; // 4 x 64 matrix
@group(0) @binding(2) var<storage, read_write> foa_out: array<f32>; // 4 channels: W, X, Y, Z
@group(0) @binding(3) var<uniform> params: FoaParams;

var<workgroup> raw_foa: array<f32, 4>;

@compute @workgroup_size(4)
fn project_foa(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let ch = local_id.x; // 0: W, 1: X, 2: Y, 3: Z
    if ch < 4u {
        var sum = 0.0;
        let offset = ch * params.latent_dim;
        for (var i = 0u; i < params.latent_dim; i = i + 1u) {
            sum = fma(foa_weights[offset + i], latent_state[i], sum);
        }
        raw_foa[ch] = sum * params.bit_scale;
    }
    workgroupBarrier();

    if ch == 0u {
        foa_out[0] = raw_foa[0]; // W (omni)
    } else if ch == 1u {
        let cos_y = cos(params.yaw_rad);
        let sin_y = sin(params.yaw_rad);
        foa_out[1] = raw_foa[1] * cos_y - raw_foa[2] * sin_y;
    } else if ch == 2u {
        let cos_y = cos(params.yaw_rad);
        let sin_y = sin(params.yaw_rad);
        foa_out[2] = raw_foa[1] * sin_y + raw_foa[2] * cos_y;
    } else if ch == 3u {
        foa_out[3] = raw_foa[3]; // Z
    }
}