//! Native Rust Golden Reference Vector Generator for RainAI Runtime.
//!
//! Generates exact Box-Cox inversion values, deterministic parameter tensors,
//! and recurrent multi-step traces for Mamba2-MoE and Ambisonic projection validation.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

pub const CONDITION_DIM: usize = 554;
pub const LATENT_DIM: usize = 64;
pub const NUM_EXPERTS: usize = 8;
pub const EXPERTS_TOP_K: usize = 2;
pub const NUM_STEPS: usize = 32;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BoxCoxReference {
    pub y: f64,
    pub lambda: f64,
    pub exact_x: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoaReference {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TraceStep {
    pub step: usize,
    pub conditioning_sample_sum: f64,
    pub latent_state_norm: f64,
    pub top_2_experts: Vec<usize>,
    pub top_2_weights: Vec<f64>,
    pub foa: FoaReference,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GoldenVectorsPayload {
    pub description: String,
    pub latent_dim: usize,
    pub condition_dim: usize,
    pub num_experts: usize,
    pub box_cox_reference: Vec<BoxCoxReference>,
    pub trace: Vec<TraceStep>,
}

pub fn inv_box_cox(y: f64, lambda: f64) -> f64 {
    let sign = if y < 0.0 {
        -1.0
    } else if y > 0.0 {
        1.0
    } else {
        0.0
    };
    let abs_y = y.abs();
    if (lambda - 1.0).abs() < 1e-6 {
        sign * (abs_y.exp() - 1.0)
    } else {
        let p = 1.0 - lambda;
        sign * ((1.0 + abs_y * p).powf(1.0 / p) - 1.0)
    }
}

pub fn generate_box_cox_table() -> Vec<BoxCoxReference> {
    let eval_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.25, 0.5),
        (-0.75, 0.5),
        (0.1, 1.0),
        (-0.5, 1.0),
        (0.4, 0.8),
        (-0.4, 0.8),
        (0.9, 0.2),
        (-0.9, 0.2),
        (1.0, 0.5),
        (-1.0, 0.5),
    ];

    eval_points
        .into_iter()
        .map(|(y, lambda)| {
            let exact_x = inv_box_cox(y, lambda);
            BoxCoxReference { y, lambda, exact_x }
        })
        .collect()
}

pub fn run_simulation() -> GoldenVectorsPayload {
    let box_cox_reference = generate_box_cox_table();

    // 1. Build deterministic parameter vectors/matrices
    let mut a_diag = vec![0.0f32; LATENT_DIM];
    let mut b_diag = vec![0.0f32; LATENT_DIM];
    for i in 0..LATENT_DIM {
        let frac = i as f32 / (LATENT_DIM - 1) as f32;
        a_diag[i] = 0.90 + 0.05 * (4.0 * PI * frac).cos();
        b_diag[i] = 0.10 * (2.0 * PI * frac).sin();
    }

    let mut w_in = vec![0.0f32; LATENT_DIM * CONDITION_DIM];
    for i in 0..LATENT_DIM {
        for j in 0..CONDITION_DIM {
            w_in[i * CONDITION_DIM + j] = 0.01 * ((i as f32) * (j as f32)).sin();
        }
    }
    let b_in = vec![0.005f32; LATENT_DIM];

    let mut w_gate = vec![0.0f32; NUM_EXPERTS * LATENT_DIM];
    for exp in 0..NUM_EXPERTS {
        for lat in 0..LATENT_DIM {
            w_gate[exp * LATENT_DIM + lat] = 0.02 * ((exp as f32) * (lat as f32)).cos();
        }
    }

    let mut w_foa = vec![0.0f32; 4 * LATENT_DIM];
    let scale = 1.0 / (LATENT_DIM as f32 / 4.0).sqrt();
    for i in 0..LATENT_DIM {
        let ch = i % 4;
        let val = if ch == 0 {
            scale * 1.0
        } else {
            scale * ((i + 1) as f32 * 0.5).sin()
        };
        w_foa[ch * LATENT_DIM + i] = val;
    }

    // 2. Step simulation
    let mut latent_state = vec![0.0f32; LATENT_DIM];
    let mut trace = Vec::with_capacity(NUM_STEPS);

    for t in 0..NUM_STEPS {
        let t_f = t as f32;
        let mut conditioning = vec![0.0f32; CONDITION_DIM];
        for k in 0..CONDITION_DIM {
            conditioning[k] = 0.5 + 0.3 * (k as f32 * 0.05 + t_f * 0.1).sin();
        }

        // u_t = w_in @ conditioning + b_in
        let mut u_t = b_in.clone();
        for i in 0..LATENT_DIM {
            let row_offset = i * CONDITION_DIM;
            let mut dot = 0.0f32;
            for j in 0..CONDITION_DIM {
                dot += w_in[row_offset + j] * conditioning[j];
            }
            u_t[i] += dot;
        }

        // s_t = a_diag * s_{t-1} + b_diag * u_t
        for i in 0..LATENT_DIM {
            latent_state[i] = a_diag[i] * latent_state[i] + b_diag[i] * u_t[i];
        }

        // router_logits = w_gate @ latent_state
        let mut router_logits = [0.0f32; NUM_EXPERTS];
        for exp in 0..NUM_EXPERTS {
            let row_offset = exp * LATENT_DIM;
            let mut dot = 0.0f32;
            for i in 0..LATENT_DIM {
                dot += w_gate[row_offset + i] * latent_state[i];
            }
            router_logits[exp] = dot;
        }

        // Top-2 expert selection
        let mut indexed_logits: Vec<(usize, f32)> = router_logits
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_2_indices: Vec<usize> = indexed_logits.iter().take(EXPERTS_TOP_K).map(|x| x.0).collect();
        let max_logit = indexed_logits[0].1;
        let exp0 = (indexed_logits[0].1 - max_logit).exp();
        let exp1 = (indexed_logits[1].1 - max_logit).exp();
        let sum_exp = exp0 + exp1;
        let top_2_weights = vec![(exp0 / sum_exp) as f64, (exp1 / sum_exp) as f64];

        // Expert modulation
        let expert_boost = 1.0 + 0.05 * (top_2_indices[0] as f32);
        for i in 0..LATENT_DIM {
            latent_state[i] *= expert_boost;
        }

        // Project to 4-channel Ambisonics (FOA)
        let mut foa_out = [0.0f32; 4];
        for ch in 0..4 {
            let row_offset = ch * LATENT_DIM;
            let mut dot = 0.0f32;
            for i in 0..LATENT_DIM {
                dot += w_foa[row_offset + i] * latent_state[i];
            }
            foa_out[ch] = dot;
        }

        let cond_sum: f64 = conditioning.iter().map(|&x| x as f64).sum();
        let state_norm: f64 = latent_state.iter().map(|&x| (x * x) as f64).sum::<f64>().sqrt();

        trace.push(TraceStep {
            step: t,
            conditioning_sample_sum: cond_sum,
            latent_state_norm: state_norm,
            top_2_experts: top_2_indices,
            top_2_weights,
            foa: FoaReference {
                w: foa_out[0] as f64,
                x: foa_out[1] as f64,
                y: foa_out[2] as f64,
                z: foa_out[3] as f64,
            },
        });
    }

    GoldenVectorsPayload {
        description: "Deterministic SymPy & NumPy Golden Reference Trace for RainAI Inference Runtime".to_string(),
        latent_dim: LATENT_DIM,
        condition_dim: CONDITION_DIM,
        num_experts: NUM_EXPERTS,
        box_cox_reference,
        trace,
    }
}

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// In-process Golden Reference Vector generator pipeline.
pub fn run_golden_vectors_pipeline(log_tx: Option<Sender<String>>) -> Result<usize> {
    let emit_log = |msg: String| {
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    emit_log("[*] Generating Golden Reference Vectors trace in-process...".to_string());
    let payload = run_simulation();

    let target_dir = PathBuf::from("crates/inference/data/golden_vectors");
    fs::create_dir_all(&target_dir).context("Failed creating golden vectors directory")?;
    let target_file = target_dir.join("baseline_step_trace.json");

    let file = File::create(&target_file).context("Failed creating output file")?;
    serde_json::to_writer_pretty(file, &payload).context("Failed serializing JSON payload")?;

    let count = payload.trace.len();
    emit_log(format!("[+] Successfully generated {} golden steps to {:?}", count, target_file));
    Ok(count)
}
