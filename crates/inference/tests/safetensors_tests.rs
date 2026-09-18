use inference::model::PrecisionFormat;
use inference::runner::InferenceRunner;
use inference::weight_loader::WeightLoader;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;
use shared::rain::{QualityTier, CONDITION_DIM};
use std::collections::HashMap;

#[test]
fn test_safetensors_loading_and_consistency_jump() {
    // 1. Synthesize weights for a full Mamba2-MoE model + Consistency Jump Head
    let cond_weights = vec![0.01f32; 64 * CONDITION_DIM];
    let cond_bias = vec![0.001f32; 64];
    let a_diag = vec![0.95f32; 64];
    let b_diag = vec![0.05f32; 64];
    let router = vec![0.125f32; 8 * 64];
    let foa_weights = vec![0.25f32; 4 * 64];
    let jump_weights = vec![0.5f32; 4 * 64];
    let jump_bias = vec![0.02f32; 4];

    let cond_w_bytes: &[u8] = bytemuck::cast_slice(&cond_weights);
    let cond_b_bytes: &[u8] = bytemuck::cast_slice(&cond_bias);
    let a_bytes: &[u8] = bytemuck::cast_slice(&a_diag);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b_diag);
    let router_bytes: &[u8] = bytemuck::cast_slice(&router);
    let foa_bytes: &[u8] = bytemuck::cast_slice(&foa_weights);
    let jump_w_bytes: &[u8] = bytemuck::cast_slice(&jump_weights);
    let jump_b_bytes: &[u8] = bytemuck::cast_slice(&jump_bias);

    let mut data_map = HashMap::new();
    data_map.insert("encoder.cond_proj.weight".to_string(), TensorView::new(Dtype::F32, vec![64, CONDITION_DIM], cond_w_bytes).unwrap());
    data_map.insert("encoder.cond_proj.bias".to_string(), TensorView::new(Dtype::F32, vec![64], cond_b_bytes).unwrap());
    data_map.insert("mamba.A_diag.weight".to_string(), TensorView::new(Dtype::F32, vec![64], a_bytes).unwrap());
    data_map.insert("mamba.B_diag.weight".to_string(), TensorView::new(Dtype::F32, vec![64], b_bytes).unwrap());
    data_map.insert("moe.router.weight".to_string(), TensorView::new(Dtype::F32, vec![8, 64], router_bytes).unwrap());
    data_map.insert("decoder.foa_proj.weight".to_string(), TensorView::new(Dtype::F32, vec![4, 64], foa_bytes).unwrap());
    data_map.insert("consistency_head.proj.weight".to_string(), TensorView::new(Dtype::F32, vec![4, 64], jump_w_bytes).unwrap());
    data_map.insert("consistency_head.proj.bias".to_string(), TensorView::new(Dtype::F32, vec![4], jump_b_bytes).unwrap());

    let metadata_map = HashMap::new();
    let safetensors_bytes = serialize(&data_map, &Some(metadata_map)).expect("SafeTensors serialization failed");

    // 2. Load directly into WeightCache via WeightLoader
    let cache = WeightLoader::load_safetensors_bytes(&safetensors_bytes, QualityTier::StudioFp32)
        .expect("Failed to load SafeTensors bytes");

    assert!(cache.contains("encoder.cond_proj.weight"));
    assert!(cache.contains("consistency_head.proj.weight"));
    assert!(cache.contains("consistency_head.proj.bias"));

    let jump_layer = cache.get("consistency_head.proj.weight").unwrap();
    assert_eq!(jump_layer.format, PrecisionFormat::Fp32);
    assert_eq!(jump_layer.shape, vec![4, 64]);

    // 3. Initialize InferenceRunner
    let mut runner = InferenceRunner::new(QualityTier::StudioFp32, cache);
    assert!(runner.has_consistency_jump_head());

    let mut cond = [0.0f32; CONDITION_DIM];
    cond[0] = 0.75; // Rain intensity
    cond[5] = 0.40; // Wind speed
    cond[9] = 1.0;  // Tin roof surface

    // 4. Test standard step with thinking steps
    runner.set_thinking_steps(1);
    let (w1, x1, _y1, _z1) = runner.step(&cond);
    assert!(w1.is_finite());
    assert!(x1.is_finite());

    runner.set_thinking_steps(4);
    let (w4, x4, _y4, _z4) = runner.step(&cond);
    assert!(w4.is_finite());
    assert!(x4.is_finite());


    // 5. Test 1-step Consistency Jump Head evaluation
    let (jw, jx, jy, jz) = runner.fast_consistency_step(&cond);
    assert!(jw.is_finite() && jw != 0.0);
    assert!(jx.is_finite());
    assert!(jy.is_finite());
    assert!(jz.is_finite());

    // 6. Test routing through use_consistency_jump toggle
    runner.set_use_consistency_jump(true);
    let (routed_w, routed_x, routed_y, routed_z) = runner.step(&cond);
    assert_eq!(routed_w, jw);
    assert_eq!(routed_x, jx);
    assert_eq!(routed_y, jy);
    assert_eq!(routed_z, jz);
}
