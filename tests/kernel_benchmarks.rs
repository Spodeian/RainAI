//! Centralized workspace-level benchmarking suite for RainAI inference kernels.

use criterion::{criterion_group, criterion_main, Criterion};
use inference::kernels::{dense_projection, step_recurrence_f32, ternary_matmul_simd_f32};

fn bench_workspace_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("RainAI Workspace Kernels");

    let in_dim = 554;
    let out_dim = 64;
    let input = vec![0.5f32; in_dim];
    let weights_f32 = vec![0.1f32; out_dim * in_dim];
    let mut output_f32 = vec![0.0f32; out_dim];

    group.bench_function("dense_projection_f32", |b| {
        b.iter(|| {
            dense_projection(&input, &weights_f32, None, &mut output_f32);
        })
    });

    let latent_dim = 64;
    let mut latent_state = vec![0.1f32; latent_dim];
    let a_diag = vec![0.95f32; latent_dim];
    let b_diag = vec![0.05f32; latent_dim];
    let u_t = vec![0.2f32; latent_dim];

    group.bench_function("mamba2_step_recurrence_f32", |b| {
        b.iter(|| {
            step_recurrence_f32(&mut latent_state, &a_diag, &b_diag, &u_t);
        })
    });

    let packed_weights = vec![0x55u8; (out_dim * in_dim + 3) / 4];
    let mut ternary_output = vec![0.0f32; out_dim];
    group.bench_function("ternary_matmul_simd_f32", |b| {
        b.iter(|| {
            ternary_matmul_simd_f32(&packed_weights, &input, &mut ternary_output, 0.125);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_workspace_kernels);
criterion_main!(benches);
