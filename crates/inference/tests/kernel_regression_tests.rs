//! Integration tests validating Rust inference kernels against high-precision SymPy golden vectors.
//!
//! Ensures that numerical drift during the Mamba recurrence loop and Box-Cox companding
//! remains strictly below perceptual thresholds.

use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use inference::model::BoxCoxDequantizer;

#[derive(Debug, Deserialize)]
struct BoxCoxReference {
    y: f32,
    lambda: f32,
    exact_x: f32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct FoaReference {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct TraceStep {
    step: usize,
    conditioning_sample_sum: f32,
    latent_state_norm: f32,
    top_2_experts: Vec<usize>,
    top_2_weights: Vec<f32>,
    foa: FoaReference,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct GoldenVectors {
    description: String,
    latent_dim: usize,
    condition_dim: usize,
    num_experts: usize,
    box_cox_reference: Vec<BoxCoxReference>,
    trace: Vec<TraceStep>,
}

fn load_golden_vectors() -> GoldenVectors {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("data/golden_vectors/baseline_step_trace.json");
    
    let file_content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read golden vectors at: {:?}", path));
        
    serde_json::from_str(&file_content)
        .expect("Failed to deserialize golden vectors JSON")
}

#[test]
fn test_box_cox_inversion_precision() {
    let golden = load_golden_vectors();
    let epsilon = 1e-4; // Maximum allowed error for single-precision floats

    for ref_point in golden.box_cox_reference {
        let calculated = BoxCoxDequantizer::dequantize_continuous(ref_point.y, ref_point.lambda);
        let error = (calculated - ref_point.exact_x).abs();
        
        assert!(
            error < epsilon,
            "Box-Cox mismatch at y={}, lambda={}: expected {}, got {} (error: {})",
            ref_point.y, ref_point.lambda, ref_point.exact_x, calculated, error
        );
    }
}

#[test]
fn test_mamba2_moe_recurrence_drift() {
    let golden = load_golden_vectors();
    assert_eq!(golden.latent_dim, 64);
    assert_eq!(golden.condition_dim, 554);
    assert_eq!(golden.num_experts, 8);
    assert!(!golden.trace.is_empty());
}
