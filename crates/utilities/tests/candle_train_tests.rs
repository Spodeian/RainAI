//! Automated Integration Tests for Native Rust Candle Training Engine.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use utilities::candle_train::{
    compute_beta_vae_loss, compute_expert_diversity_loss, compute_flow_matching_loss,
    compute_hierarchical_multi_res_loss, compute_moe_load_balancing_loss,
    compute_physics_trajectory_loss, generate_batch, run_candle_training_pipeline,
    CandleAffineAlignment, CandleConsistencyHead, CandleInvasiveMetaController,
    CandleJambaSelfAttention, CandleLatentAttention, CandleEngramBank, CandleLearnedQuantizer,
    CandleMamba2MoE, CandleMambaSSDBlock,
    CandleManifestDataset, CandleSpatialVae, CandleTrainConfig, TrainingPhase, LATENT_DIM,
};


#[test]
fn test_spatial_vae_forward_and_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let vae = CandleSpatialVae::new(vs).expect("Failed building CandleSpatialVae");

    let batch = generate_batch(2, &device, 0.0).expect("Failed generating batch");
    let (pred_bands, pred_foa, mu, logvar) = vae
        .forward(&batch.audio_features, &batch.conditioning)
        .expect("Forward pass failed");

    assert_eq!(pred_bands.dims(), &[2, 16]);
    assert_eq!(pred_foa.dims(), &[2, 4]);
    assert_eq!(mu.dims(), &[2, LATENT_DIM]);
    assert_eq!(logvar.dims(), &[2, LATENT_DIM]);

    let (loss, recon, kl) =
        compute_beta_vae_loss(&pred_bands, &batch.target_bands, &mu, &logvar, 0.01)
            .expect("Loss computation failed");

    let loss_val: f32 = loss.to_scalar().expect("Failed converting loss to scalar");
    let recon_val: f32 = recon.to_scalar().expect("Failed converting recon to scalar");
    let kl_val: f32 = kl.to_scalar().expect("Failed converting kl to scalar");

    assert!(loss_val.is_finite(), "Loss is not finite: {loss_val}");
    assert!(recon_val >= 0.0, "Recon loss must be non-negative: {recon_val}");
    assert!(kl_val.is_finite(), "KL divergence is not finite: {kl_val}");

    // Verify autograd backward computation succeeds
    let _grads = loss.backward().expect("Backward pass failed");
}

#[test]
fn test_mamba2_moe_forward_and_physics_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mamba = CandleMamba2MoE::new(vs).expect("Failed building CandleMamba2MoE");

    let batch = generate_batch(2, &device, 0.15).expect("Failed generating batch");
    let h_prev = Tensor::zeros((2, 128), DType::F32, &device).expect("Failed creating h_prev");

    let (z_pred, next_h, router_probs) = mamba
        .forward(&batch.z_prev, &batch.conditioning, &h_prev)
        .expect("Mamba forward pass failed");

    assert_eq!(z_pred.dims(), &[2, LATENT_DIM]);
    assert_eq!(next_h.dims(), &[2, 128]);
    assert_eq!(router_probs.dims(), &[2, 8]);

    let traj_loss = compute_physics_trajectory_loss(&z_pred, &batch.z_target, &batch.z_prev, 0.1)
        .expect("Trajectory loss failed");
    let aux_loss = compute_moe_load_balancing_loss(&router_probs)
        .expect("MoE aux loss failed");

    let total_loss = (&traj_loss + (&aux_loss * 0.05).unwrap()).unwrap();
    let loss_val: f32 = total_loss.to_scalar().expect("Failed scalar conversion");

    assert!(loss_val.is_finite(), "Mamba loss is not finite: {loss_val}");
    assert!(loss_val > 0.0, "Mamba loss must be strictly positive");

    let _grads = total_loss.backward().expect("Mamba backward pass failed");
}

#[test]
fn test_candle_pipeline_runner() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_candle_test_{}", rand::random::<u64>()));
    
    let config = CandleTrainConfig {
        phases: vec![TrainingPhase::All],
        vae_epochs: 1,
        mamba_epochs: 1,
        batch_size: 2,
        max_batches: 2,
        learning_rate: 1e-3,
        accumulation_steps: 1,
        cfg_dropout: 0.1,
        lambda_vel: 0.1,
        lambda_acc: 0.05,
        lambda_drag: 0.02,
        lambda_z: 1e-3,
        lambda_doa: 0.1,
        lambda_diff: 0.05,
        lambda_straight: 0.05,
        lambda_div: 0.05,
        gamma_tabu: 1.0,
        enable_distillation: true,
        lambda_distill: 0.1,
        so3_aug_prob: 0.3,
        surface_mixup_prob: 0.25,
        thinking_curriculum: true,
        stochastic_jitter_sigma: 0.01,
        beta_kl: 0.001,
        use_real_data: false,
        use_flow_matching: true,
        tau_moe: 0.75,
        max_thinking_steps: 3,
        eps_thinking_halt: 0.02,
        max_grad_norm: 1.0,
        warmup_steps: 1,
        val_ratio: 0.2,
        stft_mode: utilities::stft_loss::StftLossMode::Combined,
        stft_weight: 0.2,
        mfp_depth: 3,
        mfp_decay: 0.5,
        lambda_soup_deficit: 0.1,
        enable_latent_caching: false,
        continuous_refinement: false,
        output_dir: temp_dir.clone(),
        device: "cpu".to_string(),
    };


    run_candle_training_pipeline(&config).expect("Pipeline execution failed");

    let vae_path = temp_dir.join("spatial_vae.safetensors");
    let vae_best_path = temp_dir.join("spatial_vae_best.safetensors");
    let mamba_path = temp_dir.join("mamba2_moe.safetensors");
    let mamba_best_path = temp_dir.join("mamba2_moe_best.safetensors");
    let mamba_fast_path = temp_dir.join("mamba2_moe_fast.safetensors");
    let mamba_soup_path = temp_dir.join("mamba2_dense_soup.safetensors");
    let manifest_path = temp_dir.join("candle_manifest.json");

    assert!(vae_path.exists(), "Spatial VAE SafeTensors was not created");
    assert!(vae_best_path.exists(), "Spatial VAE Best SafeTensors was not created");
    assert!(mamba_path.exists(), "Mamba-2 MoE SafeTensors was not created");
    assert!(mamba_best_path.exists(), "Mamba-2 MoE Best SafeTensors was not created");
    assert!(mamba_fast_path.exists(), "Mamba-2 MoE Fast SafeTensors was not created");
    assert!(mamba_soup_path.exists(), "Mamba-2 Dense Soup SafeTensors was not created");
    assert!(manifest_path.exists(), "Candle manifest was not created");


    // Verify manifest contains new training telemetry fields
    let manifest_content = std::fs::read_to_string(&manifest_path).expect("Failed reading manifest");
    let v: serde_json::Value = serde_json::from_str(&manifest_content).expect("Invalid JSON in manifest");
    assert_eq!(v["tau_moe"], 0.75);
    assert_eq!(v["max_thinking_steps"], 3);
    assert_eq!(v["enable_distillation"], true);
    assert_eq!(v["lambda_div"], 0.05);
    assert_eq!(v["gamma_tabu"], 1.0);
    assert_eq!(v["scheduler"], "CosineAnnealingWithWarmup");

    // Cleanup
    let _ = std::fs::remove_dir_all(temp_dir);
}

#[test]
fn test_candle_affine_alignment_and_quantizer() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let affine = CandleAffineAlignment::new(LATENT_DIM, vs.pp("affine"))
        .expect("Failed creating CandleAffineAlignment");
    let quantizer = CandleLearnedQuantizer::new(LATENT_DIM, 6.0, vs.pp("quantizer"))
        .expect("Failed creating CandleLearnedQuantizer");

    let z_raw = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).expect("Failed generating z");
    let z_align = affine.forward(&z_raw).expect("Affine forward failed");
    assert_eq!(z_align.dims(), &[2, LATENT_DIM]);

    let z_q = quantizer.forward(&z_align, 0.1).expect("Quantizer forward failed");
    assert_eq!(z_q.dims(), &[2, LATENT_DIM]);

    let loss = z_q.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("Backward pass failed");
}

#[test]
fn test_candle_mamba_ssd_block() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let ssd = CandleMambaSSDBlock::new(128, 64, vs.pp("ssd")).expect("Failed building CandleMambaSSDBlock");

    let x = Tensor::randn(0.0f32, 1.0f32, (2, 128), &device).expect("Failed creating input");
    let h_prev = Tensor::zeros((2, 128, 64), DType::F32, &device).expect("Failed creating h_prev");

    let (y, h_next) = ssd.forward(&x, &h_prev).expect("SSD forward failed");
    assert_eq!(y.dims(), &[2, 128]);
    assert_eq!(h_next.dims(), &[2, 128, 64]);

    let loss = y.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("SSD backward pass failed");
}

#[test]
fn test_candle_jamba_self_attention() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let attn = CandleJambaSelfAttention::new(128, vs.pp("attn"))
        .expect("Failed building CandleJambaSelfAttention");

    let x = Tensor::randn(0.0f32, 1.0f32, (2, 128), &device).expect("Failed creating input");
    let out = attn.forward(&x).expect("Attention forward failed");
    assert_eq!(out.dims(), &[2, 128]);

    let loss = out.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("Attention backward pass failed");
}

#[test]
fn test_candle_invasive_meta_controller() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let meta = CandleInvasiveMetaController::new(8, 16, vs.pp("meta"))
        .expect("Failed building CandleInvasiveMetaController");

    let moe_logits = Tensor::randn(0.0f32, 1.0f32, (2, 8), &device).expect("Failed creating logits");
    let telemetry = Tensor::randn(0.5f32, 0.2f32, (2, 4), &device).expect("Failed creating telemetry");
    let user_weights = Tensor::randn(0.5f32, 0.1f32, (2, 3), &device).expect("Failed creating user weights");
    let quality_scores = Tensor::randn(0.9f32, 0.05f32, (2, 2), &device).expect("Failed creating quality");
    let slice_level = Tensor::randn(0.75f32, 0.1f32, (2, 1), &device).expect("Failed creating slice");
    let telem_state = Tensor::zeros((2, 32, 16), DType::F32, &device).expect("Failed creating telem state");

    let out = meta.forward(&moe_logits, &telemetry, &user_weights, &quality_scores, &slice_level, &telem_state)
        .expect("MetaController forward failed");

    assert_eq!(out.expert_mask.dims(), &[2, 8]);
    assert_eq!(out.tau_moe.dims(), &[2, 1]);
    assert_eq!(out.ambisonic_order.dims(), &[2, 1]);
    assert_eq!(out.diffusion_bypass.dims(), &[2, 1]);
    assert_eq!(out.synthesis_blend.dims(), &[2, 1]);
    assert_eq!(out.stress.dims(), &[2, 1]);
    assert_eq!(out.pre_generated_steps.dims(), &[2, 1]);
    assert_eq!(out.next_telem_state.dims(), &[2, 32, 16]);

    let rec_steps = out.recommended_steps();
    assert!((1..=5).contains(&rec_steps), "Recommended steps must be in 1..=5: {rec_steps}");

    let loss = out.expert_mask.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("MetaController backward pass failed");
}

#[test]
fn test_flow_matching_and_hierarchical_loss() {
    let device = Device::Cpu;
    let pred_v = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).expect("Failed creating pred_v");
    let z_target = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).expect("Failed creating z_target");
    let z_noise = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).expect("Failed creating z_noise");

    let flow_loss = compute_flow_matching_loss(&pred_v, &z_target, &z_noise, 1e-4)
        .expect("Flow matching loss failed");
    let flow_val: f32 = flow_loss.to_scalar().expect("Failed converting to scalar");
    assert!(flow_val.is_finite());
    assert!(flow_val >= 0.0);

    let pred_audio = Tensor::randn(0.0f32, 0.5f32, (2, 4800), &device).expect("Failed creating pred_audio");
    let target_audio = Tensor::randn(0.0f32, 0.5f32, (2, 4800), &device).expect("Failed creating target_audio");

    let (l_total, l_fine, l_energy) = compute_hierarchical_multi_res_loss(&pred_audio, &target_audio)
        .expect("Hierarchical loss failed");
    let total_val: f32 = l_total.to_scalar().expect("Failed converting to scalar");
    let fine_val: f32 = l_fine.to_scalar().expect("Failed converting to scalar");
    let energy_val: f32 = l_energy.to_scalar().expect("Failed converting to scalar");

    assert!(total_val.is_finite());
    assert!(fine_val >= 0.0);
    assert!(energy_val >= 0.0);
}

#[test]
fn test_candle_manifest_dataset() {
    let manifest_path = std::path::Path::new("data/processed/manifest.json");
    if manifest_path.exists() {
        let dataset = CandleManifestDataset::load_from_manifest(manifest_path)
            .expect("Failed loading real manifest dataset");
        assert!(!dataset.entries.is_empty(), "Manifest entries must not be empty");

        let device = Device::Cpu;
        let batch = dataset.sample_batch(4, &device, 0.1)
            .expect("Failed sampling batch from real manifest dataset");

        assert_eq!(batch.audio_features.dims(), &[4, LATENT_DIM]);
        assert_eq!(batch.conditioning.dims(), &[4, 554]);
        assert_eq!(batch.target_bands.dims(), &[4, 16]);
        assert_eq!(batch.z_target.dims(), &[4, LATENT_DIM]);

        // Test dataset split
        let (train_ds, val_ds) = dataset.split(0.2);
        assert!(!train_ds.entries.is_empty());
        assert!(!val_ds.entries.is_empty());
    }
}

#[test]
fn test_smooth_softmax_routing() {
    let device = Device::Cpu;
    // Logits: row 0 concentrated (expert 0 dominates), row 1 uniform
    let logits = Tensor::from_slice(
        &[
            10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        (2, 8),
        &device,
    ).expect("Failed creating test logits");

    let (smooth_weights, weights_as_mask, mean_eff) =
        CandleMamba2MoE::route_smooth_softmax(&logits, 0.75)
            .expect("Smooth routing failed");

    assert_eq!(smooth_weights.dims(), &[2, 8]);
    assert_eq!(weights_as_mask.dims(), &[2, 8]);

    let w_vec = smooth_weights.to_vec2::<f32>().unwrap();

    // ALL weights must be non-zero — no discrete zeroing
    for (row, row_weights) in w_vec.iter().enumerate() {
        for (e, &w) in row_weights.iter().enumerate() {
            assert!(w > 0.0, "Row {row} expert {e} weight must be > 0 (got {w}) — smooth routing must never zero-out experts");
        }
    }

    // Weights must sum to ~1.0 per row
    let row0_sum: f32 = w_vec[0].iter().sum();
    let row1_sum: f32 = w_vec[1].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "Row 0 must sum to 1.0, got {row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "Row 1 must sum to 1.0, got {row1_sum}");

    // Concentrated logits (row 0) must have higher weight on expert 0 than uniform (row 1)
    assert!(
        w_vec[0][0] > w_vec[1][0],
        "Concentrated row must weight expert 0 higher than uniform row: {:.4} vs {:.4}",
        w_vec[0][0], w_vec[1][0]
    );

    // Uniform logits (row 1) should produce near-uniform weights (1/8 = 0.125)
    for &w in &w_vec[1] {
        assert!((w - 0.125).abs() < 0.01, "Uniform logits must produce ~uniform weights, got {w}");
    }

    // Mean effective count: exp(H). For uniform dist over 8, H = ln(8) ≈ 2.079, exp(H) ≈ 8.
    // For concentrated (row 0), exp(H) ≈ 1 (nearly all mass on one expert).
    // Mean should be between 1 and 8.
    let eff: f32 = mean_eff.to_scalar().unwrap();
    assert!(eff > 1.0 && eff <= 8.0, "Mean effective count must be in (1, 8], got {eff}");
}

#[test]
fn test_candle_thinking_block() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let thinking = utilities::candle_train::CandleThinkingBlock::new(
        LATENT_DIM,
        554,
        vs.pp("thinking"),
    ).expect("Failed building CandleThinkingBlock");

    let z_init = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).unwrap();
    let cond = Tensor::randn(0.0f32, 0.5f32, (2, 554), &device).unwrap();

    let (z_refined, steps_taken, halting_probs) = thinking
        .forward_thinking(&z_init, &cond, 4, 0.001)
        .expect("Thinking pass failed");

    assert_eq!(z_refined.dims(), &[2, LATENT_DIM]);
    assert!((1..=4).contains(&steps_taken));
    assert_eq!(halting_probs.len(), steps_taken);

    // Verify autograd backward computation through thinking loop
    let loss = z_refined.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("Thinking backward failed");
}

#[test]
fn test_cosine_scheduler_and_grad_clipping() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let align = CandleAffineAlignment::new(16, vs.pp("align")).unwrap();

    let mut scheduler = utilities::candle_train::CosineAnnealingWithWarmup::new(
        1e-3,
        1e-6,
        10,
        100,
    );

    // Step 1: linear warmup
    let lr_1 = scheduler.step();
    assert!(lr_1 > 0.0 && lr_1 <= 1e-4);

    // Step 10: end of warmup
    for _ in 2..=10 {
        scheduler.step();
    }
    assert!((scheduler.step() - 1e-3).abs() < 1e-4);

    // Test gradient clipping
    let x = Tensor::randn(0.0f32, 1.0f32, (2, 16), &device).unwrap();
    let y = align.forward(&x).unwrap();
    // Huge scale to create exploding gradients
    let huge_loss = (y.sqr().unwrap().mean_all().unwrap() * 1000.0).unwrap();
    let mut grads = huge_loss.backward().unwrap();

    let initial_norm = utilities::candle_train::clip_grad_norm_varmap(&varmap, &mut grads, 1.0)
        .expect("Gradient clipping failed");
    assert!(initial_norm > 1.0, "Initial norm should be large before clipping");
}

#[test]
fn test_stft_loss_modes() {
    let device = Device::Cpu;
    let stft = utilities::stft_loss::MultiResolutionStftLoss::default();

    // 1. Envelope mode
    let pred_bands = Tensor::full(0.5f32, (2, 16), &device).unwrap();
    let target_bands = Tensor::full(0.5f32, (2, 16), &device).unwrap();

    let loss_zero = stft.evaluate_loss(
        utilities::stft_loss::StftLossMode::Envelope16,
        &pred_bands,
        &target_bands,
        None,
        None,
        &device,
    ).expect("STFT envelope evaluation failed");

    let val_zero: f32 = loss_zero.to_scalar().unwrap();
    assert!(val_zero.abs() < 1e-4, "Identity reconstruction must produce near-zero envelope loss: {val_zero}");

    // 2. Waveform mode
    let pred_audio = vec![0.1f32; 4096];
    let target_audio = vec![0.1f32; 4096];
    let (wf_loss, sc, mag) = stft.evaluate_waveform_loss(&pred_audio, &target_audio);
    assert!(wf_loss < 1e-4, "Identity audio frames must yield zero waveform loss: {wf_loss}");
    assert!(sc < 1e-4);
    assert!(mag < 1e-4);

    // 3. Mel mode
    let target_diff = Tensor::full(0.8f32, (2, 16), &device).unwrap();
    let loss_mel = stft.evaluate_loss(
        utilities::stft_loss::StftLossMode::MelSpectral,
        &pred_bands,
        &target_diff,
        None,
        None,
        &device,
    ).expect("Mel spectral loss evaluation failed");
    let val_mel: f32 = loss_mel.to_scalar().unwrap();
    assert!(val_mel > 0.0, "Dissimilar spectra must produce positive loss");
}

#[test]
fn test_acoustic_doa_and_diffuseness_loss() {
    let device = Device::Cpu;
    
    // Batch of 2, 4 FOA channels [W, X, Y, Z]
    // Directional downward soundfield
    let target_foa = Tensor::from_slice(
        &[0.8f32, 0.2, 0.1, -0.6,
          0.9f32, -0.3, 0.4, -0.5],
        (2, 4),
        &device,
    ).unwrap();

    // 1. Exact identity should give near-zero DOA error and diffuseness loss
    let (_ident_doa, doa_err) = utilities::stft_loss::compute_acoustic_intensity_and_doa_loss(
        &target_foa,
        &target_foa,
        0.5,
    ).expect("DOA computation failed");
    let ident_diff = utilities::stft_loss::compute_soundfield_diffuseness_loss(
        &target_foa,
        &target_foa,
        0.5,
    ).expect("Diffuseness computation failed");

    let doa_err_val: f32 = doa_err.to_scalar().unwrap();
    let diff_val: f32 = ident_diff.to_scalar().unwrap();
    assert!(doa_err_val.abs() < 1e-4, "Identity DOA error should be ~0: {doa_err_val}");
    assert!(diff_val.abs() < 1e-4, "Identity diffuseness loss should be ~0: {diff_val}");

    // 2. Opposing directional soundfield should yield bounded positive loss
    let opp_foa = Tensor::from_slice(
        &[0.8f32, -0.2, -0.1, 0.6,
          0.9f32, 0.3, -0.4, 0.5],
        (2, 4),
        &device,
    ).unwrap();
    let (opp_doa, opp_err) = utilities::stft_loss::compute_acoustic_intensity_and_doa_loss(
        &opp_foa,
        &target_foa,
        0.5,
    ).unwrap();
    let opp_doa_val: f32 = opp_doa.to_scalar().unwrap();
    let opp_err_val: f32 = opp_err.to_scalar().unwrap();
    assert!(opp_doa_val > 0.0, "Opposing DOA loss must be strictly positive");
    assert!(opp_err_val > 0.0 && opp_err_val.is_finite());

    // 3. Autograd backward pass check
    let _grads = opp_doa.backward().expect("DOA backward pass failed");
}

#[test]
fn test_physics_trajectory_v2_acceleration_and_drag() {
    let device = Device::Cpu;

    let z_prev2 = Tensor::full(0.9f32, (2, 64), &device).unwrap();
    let z_prev = Tensor::full(1.0f32, (2, 64), &device).unwrap();
    let z_target = Tensor::full(1.1f32, (2, 64), &device).unwrap(); // constant velocity v = 0.1, a = 0.0, speed = sqrt(64 * 0.01) = 0.8

    // Constant velocity trajectory: z_pred = 1.1 (v = 0.1, a = 0.0, speed = 0.8 < v_terminal 2.5)
    let z_pred_ideal = Tensor::full(1.1f32, (2, 64), &device).unwrap();
    let (_loss_ideal, pos_ideal, acc_ideal, drag_ideal) = utilities::candle_train::compute_physics_trajectory_loss_v2(
        &z_pred_ideal,
        &z_target,
        &z_prev,
        Some(&z_prev2),
        0.1,
        0.05,
        0.02,
        2.5,
    ).unwrap();

    let pos_val: f32 = pos_ideal.to_scalar().unwrap();
    let acc_val: f32 = acc_ideal.to_scalar().unwrap();
    let drag_val: f32 = drag_ideal.to_scalar().unwrap();
    assert!(pos_val.abs() < 1e-5, "Position error must be zero");
    assert!(acc_val.abs() < 1e-5, "Zero acceleration error must be zero");
    assert!(drag_val.abs() < 1e-5, "Velocity below terminal must incur zero drag");

    // Extreme velocity exceeding terminal velocity v_terminal = 2.5
    // v = 10.0 - 1.0 = 9.0 >> 2.5
    let z_pred_extreme = Tensor::full(10.0f32, (2, 64), &device).unwrap();
    let (loss_ext, _, _, drag_ext) = utilities::candle_train::compute_physics_trajectory_loss_v2(
        &z_pred_extreme,
        &z_target,
        &z_prev,
        Some(&z_prev2),
        0.1,
        0.05,
        0.02,
        2.5,
    ).unwrap();

    let drag_ext_val: f32 = drag_ext.to_scalar().unwrap();
    assert!(drag_ext_val > 10.0, "Drag barrier must strongly penalize excessive speeds: {drag_ext_val}");

    // Backward pass check
    let _grads = loss_ext.backward().expect("Physics trajectory backward pass failed");
}

#[test]
fn test_moe_router_z_loss() {
    let device = Device::Cpu;

    // Small logits -> small Z-loss
    let small_logits = Tensor::zeros((2, 8), DType::F32, &device).unwrap();
    let z_loss_small = utilities::candle_train::compute_router_z_loss(&small_logits).unwrap();
    let z_val_small: f32 = z_loss_small.to_scalar().unwrap();
    assert!(z_val_small.is_finite());

    // Exploding logits (e.g. 50.0) -> high Z-loss
    let large_logits = Tensor::full(30.0f32, (2, 8), &device).unwrap();
    let z_loss_large = utilities::candle_train::compute_router_z_loss(&large_logits).unwrap();
    let z_val_large: f32 = z_loss_large.to_scalar().unwrap();
    assert!(z_val_large > z_val_small, "Large logits must produce higher Z-loss penalty");

    // Autograd backward pass
    let _grads = z_loss_large.backward().expect("Router Z-loss backward pass failed");
}

#[test]
fn test_straight_path_flow_matching_loss() {
    let device = Device::Cpu;

    let z_noise = Tensor::zeros((4, 64), DType::F32, &device).unwrap();
    let z_target = Tensor::full(1.0f32, (4, 64), &device).unwrap();

    // Straight vector field matching the average direction
    let pred_straight = Tensor::full(1.0f32, (4, 64), &device).unwrap();
    let (loss_straight, base_straight) = utilities::candle_train::compute_straight_flow_loss(
        &pred_straight,
        &z_target,
        &z_noise,
        1e-4,
        0.1,
    ).unwrap();

    let straight_val: f32 = loss_straight.to_scalar().unwrap();
    let base_val: f32 = base_straight.to_scalar().unwrap();
    assert!(base_val.abs() < 1e-4);
    assert!(straight_val.abs() < 1e-4);

    // Curved vector field with variance across batch
    let pred_curved = Tensor::from_slice(
        &vec![2.0f32; 64 * 2].into_iter().chain(vec![-2.0f32; 64 * 2]).collect::<Vec<_>>(),
        (4, 64),
        &device,
    ).unwrap();

    let (loss_curved, _) = utilities::candle_train::compute_straight_flow_loss(
        &pred_curved,
        &z_target,
        &z_noise,
        1e-4,
        0.5,
    ).unwrap();

    let curved_val: f32 = loss_curved.to_scalar().unwrap();
    assert!(curved_val > straight_val, "Curved flow field must incur higher curvature loss");
}

#[test]
fn test_spectral_flux_transient_loss() {
    // 3 time frames, 4 frequency bins
    let target_mag = vec![
        vec![0.1f32, 0.1, 0.1, 0.1],
        vec![0.8f32, 0.9, 0.7, 0.8], // sharp onset transient
        vec![0.4f32, 0.4, 0.4, 0.4],
    ];

    // Perfect match
    let loss_ident = utilities::stft_loss::compute_spectral_flux_loss(&target_mag, &target_mag, 0.5);
    assert!(loss_ident < 1e-5, "Identity spectral flux loss must be zero: {loss_ident}");

    // Missed transient (flat prediction)
    let pred_flat = vec![
        vec![0.1f32, 0.1, 0.1, 0.1],
        vec![0.1f32, 0.1, 0.1, 0.1], // missed the droplet strike!
        vec![0.1f32, 0.1, 0.1, 0.1],
    ];
    let loss_missed = utilities::stft_loss::compute_spectral_flux_loss(&pred_flat, &target_mag, 0.5);
    assert!(loss_missed > 0.1, "Missed droplet onset transient must produce significant flux penalty: {loss_missed}");
}

#[test]
fn test_huber_loss_properties() {
    let device = Device::Cpu;

    // Small diff (<= delta 0.5): quadratic 0.5 * x^2
    let small_diff = Tensor::from_slice(&[0.2f32, -0.2f32], (2,), &device).unwrap();
    let l_small = utilities::stft_loss::huber_loss(&small_diff, 0.5).unwrap();
    let val_small: f32 = l_small.to_scalar().unwrap();
    let expected_small = 0.5 * 0.2 * 0.2;
    assert!((val_small - expected_small).abs() < 1e-5);

    // Large diff (> delta 0.5): linear delta * |x| - 0.5 * delta^2
    let large_diff = Tensor::from_slice(&[2.0f32, -2.0f32], (2,), &device).unwrap();
    let l_large = utilities::stft_loss::huber_loss(&large_diff, 0.5).unwrap();
    let val_large: f32 = l_large.to_scalar().unwrap();
    let expected_large = 0.5 * 2.0 - 0.5 * 0.5 * 0.5;
    assert!((val_large - expected_large).abs() < 1e-5);
}

#[test]
fn test_so3_ambisonic_rotation_energy_invariance() {
    let device = Device::Cpu;

    // FOA batch [B=2, C=4] (W, X, Y, Z)
    let foa = Tensor::from_slice(
        &[
            1.0f32, 0.4, -0.3, 0.5,
            0.8f32, -0.6, 0.2, 0.1,
        ],
        (2, 4),
        &device,
    ).unwrap();

    let angles = (0.75f32, -0.42f32, 1.15f32);
    let rotated = utilities::stft_loss::apply_so3_foa_rotation(&foa, angles).unwrap();

    // 1. Omnidirectional pressure W must be strictly invariant
    let orig_w = foa.narrow(1, 0, 1).unwrap().to_vec2::<f32>().unwrap();
    let rot_w = rotated.narrow(1, 0, 1).unwrap().to_vec2::<f32>().unwrap();
    for b in 0..2 {
        assert!((orig_w[b][0] - rot_w[b][0]).abs() < 1e-6, "W channel must be invariant under rotation");
    }

    // 2. Total velocity magnitude X^2 + Y^2 + Z^2 must be strictly preserved
    let orig_xyz_energy = foa.narrow(1, 1, 3).unwrap().sqr().unwrap().sum_keepdim(1).unwrap().to_vec2::<f32>().unwrap();
    let rot_xyz_energy = rotated.narrow(1, 1, 3).unwrap().sqr().unwrap().sum_keepdim(1).unwrap().to_vec2::<f32>().unwrap();
    for b in 0..2 {
        assert!(
            (orig_xyz_energy[b][0] - rot_xyz_energy[b][0]).abs() < 1e-5,
            "Directional acoustic velocity energy must be strictly preserved: orig={}, rot={}",
            orig_xyz_energy[b][0],
            rot_xyz_energy[b][0],
        );
    }

    // 3. Identity angles (0, 0, 0) should reproduce identical tensor
    let ident_rot = utilities::stft_loss::apply_so3_foa_rotation(&foa, (0.0, 0.0, 0.0)).unwrap();
    let diff = (&ident_rot - &foa).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-6, "Identity rotation produced difference: {diff}");
}

#[test]
fn test_temporal_tabu_and_expert_diversity_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mamba = CandleMamba2MoE::new(vs).unwrap();

    let z_prev = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).unwrap();
    let cond = Tensor::randn(0.0f32, 1.0f32, (2, 554), &device).unwrap();
    let h_prev = Tensor::zeros((2, 128), DType::F32, &device).unwrap();

    // Prior activation history heavily using expert 0
    let mut prior_usage = vec![0.0f32; 2 * 8];
    prior_usage[0] = 3.0; // Batch 0, expert 0 used 3 times
    prior_usage[8] = 3.0; // Batch 1, expert 0 used 3 times
    let prior_tensor = Tensor::from_slice(&prior_usage, (2, 8), &device).unwrap();

    // Without tabu penalty (gamma = 0.0)
    let (_, _, probs_no_tabu, _, _, _) = mamba
        .forward_smooth_tabu(&z_prev, &cond, &h_prev, 0.75, Some(&prior_tensor), 0.0)
        .unwrap();

    // With strong tabu penalty (gamma = 5.0)
    let (_, _, probs_with_tabu, _, _, _) = mamba
        .forward_smooth_tabu(&z_prev, &cond, &h_prev, 0.75, Some(&prior_tensor), 5.0)
        .unwrap();

    let p0_no = probs_no_tabu.get(0).unwrap().get(0).unwrap().to_scalar::<f32>().unwrap();
    let p0_tabu = probs_with_tabu.get(0).unwrap().get(0).unwrap().to_scalar::<f32>().unwrap();
    assert!(
        p0_tabu < p0_no,
        "Tabu penalty must suppress repeatedly activated expert 0: p0_no={p0_no}, p0_tabu={p0_tabu}"
    );

    // Test compute_expert_diversity_loss:
    // Case 1: Collinear distributions across steps -> diversity loss near 1.0
    let collinear_step0 = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (1, 8), &device).unwrap();
    let collinear_step1 = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (1, 8), &device).unwrap();
    let div_loss_collinear = compute_expert_diversity_loss(&[collinear_step0, collinear_step1]).unwrap();
    let col_val: f32 = div_loss_collinear.to_scalar().unwrap();
    assert!((col_val - 1.0).abs() < 1e-4, "Collinear distributions must have similarity ~1.0: {col_val}");

    // Case 2: Orthogonal distributions across steps -> diversity loss near 0.0
    let ortho_step0 = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (1, 8), &device).unwrap();
    let ortho_step1 = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (1, 8), &device).unwrap();
    let div_loss_ortho = compute_expert_diversity_loss(&[ortho_step0, ortho_step1]).unwrap();
    let ortho_val: f32 = div_loss_ortho.to_scalar().unwrap();
    assert!(ortho_val < 1e-4, "Orthogonal distributions must have similarity ~0.0: {ortho_val}");
}

#[test]
fn test_consistency_jump_head_distillation() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let head = CandleConsistencyHead::new(LATENT_DIM + 554, LATENT_DIM, vs.pp("consistency"))
        .expect("Failed creating CandleConsistencyHead");

    let z_0 = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).unwrap();
    let cond = Tensor::randn(0.0f32, 1.0f32, (2, 554), &device).unwrap();
    let z_converged = Tensor::randn(0.0f32, 1.0f32, (2, LATENT_DIM), &device).unwrap();

    let z_fast = head.forward(&z_0, &cond).expect("Consistency head forward failed");
    assert_eq!(z_fast.dims(), &[2, LATENT_DIM]);

    let distill_loss = head.compute_distill_loss(&z_fast, &z_converged, 0.5)
        .expect("Distill loss computation failed");
    let loss_val: f32 = distill_loss.to_scalar().unwrap();

    assert!(loss_val.is_finite(), "Distill loss is not finite: {loss_val}");
    assert!(loss_val > 0.0, "Distill loss must be positive: {loss_val}");

    // Backward pass
    let _grads = distill_loss.backward().expect("Consistency head backward failed");
}

#[test]
fn test_mla_latent_attention() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mla = CandleLatentAttention::new(128, vs.pp("mla"))
        .expect("Failed creating CandleLatentAttention");

    let x = Tensor::randn(0.0f32, 1.0f32, (2, 128), &device).unwrap();
    let out = mla.forward(&x).expect("MLA forward failed");
    assert_eq!(out.dims(), &[2, 128]);

    let loss = out.sqr().unwrap().mean_all().unwrap();
    let _grads = loss.backward().expect("MLA backward failed");
}

#[test]
fn test_engram_bank_deterministic() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let bank = CandleEngramBank::new(1024, 64, vs.pp("engram"))
        .expect("Failed creating CandleEngramBank");

    let x = Tensor::randn(0.0f32, 1.0f32, (2, 64), &device).unwrap();
    let out1 = bank.forward(&x).expect("Engram forward 1 failed");
    let out2 = bank.forward(&x).expect("Engram forward 2 failed");

    let diff = (&out1 - &out2).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff < 1e-6, "Engram bank lookup must be deterministic, got diff={diff}");
    assert_eq!(out1.dims(), &[2, 64]);
}

#[test]
fn test_mamba2_dense_soup_collapse() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mamba = CandleMamba2MoE::new(vs).expect("Failed building CandleMamba2MoE");

    // Dynamic router -> soup coefficients
    let router_probs = Tensor::from_slice(&[0.3f32, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], (1, 8), &device).unwrap();
    let soup_alpha = mamba.router_to_soup_coefficients(&router_probs).expect("Failed calculating soup alpha");
    assert_eq!(soup_alpha.dims(), &[1, 8]);

    // Collapse to single dense Mamba expert
    let alpha_slice = [0.25f32, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10];
    let dense_expert = mamba.collapse_to_dense_soup(&alpha_slice).expect("Failed collapsing dense soup");

    let x = Tensor::randn(0.0f32, 1.0f32, (1, 128), &device).unwrap();
    let h_prev = Tensor::zeros((1, 128), DType::F32, &device).unwrap();
    let (y, h_next) = dense_expert.forward(&x, &h_prev).expect("Dense expert forward failed");

    assert_eq!(y.dims(), &[1, 128]);
    assert_eq!(h_next.dims(), &[1, 128]);
}

#[test]
fn test_multi_frame_prediction_heads() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mamba = CandleMamba2MoE::new(vs).expect("Failed building CandleMamba2MoE");
    let fused = Tensor::randn(0.0f32, 1.0f32, (2, 128), &device).unwrap();

    let (z1, z2, z3) = mamba.predict_multi_frame(&fused).expect("Multi-frame prediction failed");
    assert_eq!(z1.dims(), &[2, LATENT_DIM]);
    assert_eq!(z2.dims(), &[2, LATENT_DIM]);
    assert_eq!(z3.dims(), &[2, LATENT_DIM]);
}

#[test]
fn test_autopilot_hardware_probe() {
    use utilities::autopilot::HardwareProfile;

    let profile = HardwareProfile::probe();
    assert!(profile.cpu_cores >= 1, "Must detect at least 1 CPU core");
    assert!(profile.total_ram_gb > 0.1, "Total RAM must be positive");
    assert!(profile.recommended_batch_size >= 1, "Batch size must be at least 1");
    assert!(profile.recommended_accumulation_steps >= 1, "Accumulation steps must be at least 1");
    assert!(profile.recommended_thinking_steps >= 1, "Thinking steps must be at least 1");
}

#[test]
fn test_autopilot_surface_entropy_and_quotas() {
    use std::collections::HashMap;
    use utilities::autopilot::{CANONICAL_SURFACES, SurfaceEntropyAuditor};

    // 1. Perfectly balanced distribution
    let mut balanced = HashMap::new();
    for &surf in &CANONICAL_SURFACES {
        balanced.insert(surf.to_string(), 10);
    }
    let (entropy_bal, quotas_bal) = SurfaceEntropyAuditor::audit(&balanced);
    assert!((entropy_bal - 1.0).abs() < 1e-4, "Equal counts must yield Shannon entropy ~1.0, got {entropy_bal}");
    for q in quotas_bal {
        assert_eq!(q.deficit_count, 0, "Balanced quotas must have zero deficit");
    }

    // 2. Heavily skewed distribution (only 1 surface)
    let mut skewed = HashMap::new();
    skewed.insert("pavement".to_string(), 90);
    let (entropy_skewed, quotas_skewed) = SurfaceEntropyAuditor::audit(&skewed);
    assert!(entropy_skewed < 0.1, "Single-surface corpus must yield near-zero entropy, got {entropy_skewed}");
    let deficit_count = quotas_skewed.iter().filter(|q| q.deficit_count > 0).count();
    assert_eq!(deficit_count, 8, "Expected 8 surfaces with deficit");
}

#[test]
fn test_autopilot_convergence_tracker() {
    use utilities::autopilot::AutoPilotConvergenceTracker;

    let mut tracker = AutoPilotConvergenceTracker::new(5, 0.008, 3);

    // Rapid progress: no plateau
    assert!(!tracker.record_loss(1.00));
    assert!(!tracker.record_loss(0.80));
    assert!(!tracker.record_loss(0.60));
    assert!(!tracker.record_loss(0.40));

    // Stagnant progress: relative delta < 0.8%
    assert!(!tracker.record_loss(0.399));
    assert!(!tracker.record_loss(0.398));
    let plateaued = tracker.record_loss(0.3975);
    assert!(plateaued, "Convergence tracker must signal plateau after 3 stagnant steps");
}

#[test]
fn test_autopilot_dense_soup_tracker() {
    use utilities::autopilot::DenseSoupTracker;

    let mut tracker = DenseSoupTracker::default();
    let initial_lambda = tracker.lambda_soup;

    // High deficit should scale up lambda
    let boosted = tracker.update(0.10, 0.50);
    assert!(boosted > initial_lambda, "High soup deficit must boost lambda_soup");

    // Low deficit should gradually decay towards baseline
    let mut current = boosted;
    for _ in 0..10 {
        current = tracker.update(0.10, 0.11);
    }
    assert!(current < boosted, "Low soup deficit must decay lambda_soup towards base");
}

