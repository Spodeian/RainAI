// binaural_convolver.wgsl - Massive Parallel 32-Tap Ambisonic Binaural Convolver
// Synthesizes binaural stereo (L/R) from 4-channel FOA buffers (W, X, Y, Z)
// using time-domain FIR convolution with symmetrical HRIR filter taps in VRAM.

const FILTER_TAPS: u32 = 32u;

struct ConvolverParams {
    num_frames: u32,       // Number of audio frames in buffer (e.g. 256, 512, 1024, 2048)
    sample_rate: f32,      // e.g. 48000.0
    head_shadow_gain: f32, // Attenuation for contralateral channel
    padding: u32,
}

// Bindings:
// 0: FOA Input Audio (Interleaved or planar 4 x num_frames: W, X, Y, Z)
// 1: HRIR Filters: 4 channels * 2 ears (L/R) * 32 taps = 256 floats
// 2: Stereo Output Audio: 2 x num_frames (L0, R0, L1, R1, ...)
// 3: ConvolverParams uniform
@group(0) @binding(0) var<storage, read> foa_in: array<f32>;
@group(0) @binding(1) var<storage, read> hrir_taps: array<f32>;
@group(0) @binding(2) var<storage, read_write> stereo_out: array<f32>;
@group(0) @binding(3) var<uniform> params: ConvolverParams;

@compute @workgroup_size(64)
fn convolve_binaural(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frame_idx = global_id.x;
    if frame_idx >= params.num_frames {
        return;
    }

    var left_acc: f32 = 0.0;
    var right_acc: f32 = 0.0;

    // Convolve with 32-tap HRIR across all 4 FOA channels:
    // ch 0: W (pressure), ch 1: X (front/back), ch 2: Y (left/right), ch 3: Z (elevation)
    for (var ch = 0u; ch < 4u; ch = ch + 1u) {
        let hrir_left_base = (ch * 2u + 0u) * FILTER_TAPS;
        let hrir_right_base = (ch * 2u + 1u) * FILTER_TAPS;
        let ch_offset = ch * params.num_frames;

        let max_tap = min(FILTER_TAPS, frame_idx + 1u);
        for (var tap = 0u; tap < max_tap; tap = tap + 1u) {
            let sample_val = foa_in[ch_offset + (frame_idx - tap)];
            left_acc = fma(sample_val, hrir_taps[hrir_left_base + tap], left_acc);
            right_acc = fma(sample_val, hrir_taps[hrir_right_base + tap], right_acc);
        }
    }

    // Write interleaved stereo L/R
    let out_idx = frame_idx * 2u;
    stereo_out[out_idx + 0u] = left_acc;
    stereo_out[out_idx + 1u] = right_acc;
}