// Multi-Head Latent Attention (MLA) WebGPU Compute Kernel
// Low-rank KV compression & decompression with scaled dot-product attention
// Optimized for workgroup size 64

struct MlaParams {
    d_model: u32,
    d_compress: u32,
    num_heads: u32,
    seq_len: u32,
};

@group(0) @binding(0) var<uniform> params: MlaParams;
@group(0) @binding(1) var<storage, read> input_q: array<f32>;
@group(0) @binding(2) var<storage, read> compressed_kv: array<f32>;
@group(0) @binding(3) var<storage, read> w_uk: array<f32>;
@group(0) @binding(4) var<storage, read> w_uv: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_vec: array<f32>;

var<workgroup> shared_scores: array<f32, 64>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let d_model = params.d_model;
    let d_compress = params.d_compress;
    let seq_len = params.seq_len;

    let head_dim = d_model / params.num_heads;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Compute attention scores between query and uncompressed key for token tid
    if (tid < seq_len) {
        var score: f32 = 0.0;
        let kv_offset = tid * d_compress;

        for (var d: u32 = 0u; d < d_model; d = d + 1u) {
            var k_d: f32 = 0.0;
            let uk_offset = d * d_compress;
            for (var c: u32 = 0u; c < d_compress; c = c + 1u) {
                k_d = fma(w_uk[uk_offset + c], compressed_kv[kv_offset + c], k_d);
            }
            score = fma(input_q[d], k_d, score);
        }

        shared_scores[tid] = score * scale;
    } else {
        shared_scores[tid] = -3.402823e+38;
    }

    workgroupBarrier();

    // Softmax reduction across workgroup
    if (tid == 0u) {
        var max_val = shared_scores[0];
        for (var i: u32 = 1u; i < seq_len; i = i + 1u) {
            max_val = max(max_val, shared_scores[i]);
        }
        shared_max = max_val;

        var exp_sum = 0.0;
        for (var i: u32 = 0u; i < seq_len; i = i + 1u) {
            let exp_score = exp(shared_scores[i] - max_val);
            shared_scores[i] = exp_score;
            exp_sum = exp_sum + exp_score;
        }
        shared_sum = max(exp_sum, 1e-8);
    }

    workgroupBarrier();

    // Normalize attention weights
    if (tid < seq_len) {
        shared_scores[tid] = shared_scores[tid] / shared_sum;
    }

    workgroupBarrier();

    // Weighted accumulation of uncompressed V into output_vec
    if (tid < d_model) {
        var out_d: f32 = 0.0;
        let uv_offset = tid * d_compress;

        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            let alpha = shared_scores[t];
            let kv_offset = t * d_compress;

            var v_t: f32 = 0.0;
            for (var c: u32 = 0u; c < d_compress; c = c + 1u) {
                v_t = fma(w_uv[uv_offset + c], compressed_kv[kv_offset + c], v_t);
            }
            out_d = fma(alpha, v_t, out_d);
        }

        output_vec[tid] = out_d;
    }
}
