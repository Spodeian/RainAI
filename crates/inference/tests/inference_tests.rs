use inference::weight_loader::{WeightLoader, WeightCache, LoadedLayer, WeightBuffer};
use inference::runner::{InferenceRunner, EngineStatus};
use inference::model::{QuantizedModelManifest, PrecisionFormat};
use inference::asset_manager::{AssetManager, AssetState};
use shared::rain::{QualityTier, CONDITION_DIM};

#[test]
fn test_ternary_2bit_pack_unpack_roundtrip() {
    let original = vec![0i8, 1, -1, 1, 0, 0, -1, -1, 1];
    let packed = WeightLoader::pack_ternary_2bit(&original);
    assert_eq!(packed.len(), 3);

    let unpacked = WeightLoader::unpack_ternary_2bit(&packed, original.len());
    assert_eq!(original, unpacked);
}

#[test]
fn test_weight_cache_operations() {
    let mut cache = WeightCache::new();
    let packed = vec![0x05; (64usize * 554).div_ceil(4)];
    let layer = LoadedLayer {
        name: "encoder.cond_proj.weight".to_string(),
        shape: vec![64, 554],
        scale: 0.125,
        format: PrecisionFormat::Ternary158,
        weights: vec![],
        packed_weights: packed.clone(),
        buffer: WeightBuffer::Ternary2Bit {
            packed,
            gamma: 0.125,
        },
    };

    cache.insert(layer);
    assert!(cache.contains("encoder.cond_proj.weight"));

    let retrieved = cache.get("encoder.cond_proj.weight").expect("Layer must exist");
    assert_eq!(retrieved.num_elements(), 64 * 554);
    assert!(retrieved.weights.is_empty());
    assert!(!retrieved.packed_weights.is_empty());
}

#[test]
fn test_inference_step_dimension() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);
    let cond = [0.5f32; CONDITION_DIM];
    let (w, x, y, z) = runner.step(&cond);
    assert!(w.is_finite());
    assert!(x.is_finite());
    assert!(y.is_finite());
    assert!(z.is_finite());
}

#[test]
fn test_tier_transition_with_fallback() {
    let weight_cache = WeightCache::default();
    let mut runner = InferenceRunner::new(QualityTier::AdaptiveMinimum, weight_cache);
    assert_eq!(runner.status, EngineStatus::Ready);
    assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum);

    runner.set_target_tier(QualityTier::StudioFp32);
    assert_eq!(runner.status, EngineStatus::DownloadingWeights);
    assert_eq!(runner.active_tier, QualityTier::AdaptiveMinimum);
    assert!(runner.is_fallback_active);

    runner.assets.fp32_state = AssetState::Ready;
    runner.poll_downloads();
    assert_eq!(runner.active_tier, QualityTier::StudioFp32);
    assert!(!runner.is_fallback_active);
    assert_eq!(runner.status, EngineStatus::Ready);
}

#[test]
fn test_load_default_ternary_manifest() {
    let manifest = QuantizedModelManifest::load_default_ternary()
        .expect("Embedded ternary manifest must parse cleanly");
    assert_eq!(manifest.tier, "ternary_1_58bit");
    assert_eq!(manifest.activation_qat.per_channel_bit_widths.len(), 64);
    assert!(manifest.layers.contains_key("encoder.cond_proj.weight"));
}

#[test]
fn test_fallback_chain_when_unready() {
    let mgr = AssetManager::new();
    let (eff, is_fallback) = mgr.resolve_effective_tier(QualityTier::StudioFp32);
    assert_eq!(eff, QualityTier::AdaptiveMinimum);
    assert!(is_fallback);
}

#[test]
fn test_geometric_midpoint_boundaries_and_pareto_allocation() {
    use inference::model::LayerRole;

    // 1. Invariant: Geometric level midpoints
    assert_eq!(PrecisionFormat::from_continuous_bit_width(0.2), PrecisionFormat::Pruned);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(1.0), PrecisionFormat::Ternary158);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(1.80), PrecisionFormat::Ternary158);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(1.81), PrecisionFormat::Int2);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(2.58), PrecisionFormat::Int2);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(2.60), PrecisionFormat::Posit8);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(3.58), PrecisionFormat::Posit8);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(3.60), PrecisionFormat::Int4);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(4.58), PrecisionFormat::Int4);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(4.60), PrecisionFormat::Int5);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(5.58), PrecisionFormat::Int5);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(5.60), PrecisionFormat::Int8);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(8.0), PrecisionFormat::Bf16);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(12.0), PrecisionFormat::Fp16);
    assert_eq!(PrecisionFormat::from_continuous_bit_width(18.0), PrecisionFormat::Fp32);

    // 2. Invariant: Natural per-layer Pareto allocation
    // SSM Recurrence: Retains Bf16 / Fp32 for recurrent stability
    assert_eq!(LayerRole::MambaStateSpaceRecurrence.down_quantization_fallback(8.0), PrecisionFormat::Bf16);
    assert_eq!(LayerRole::MambaStateSpaceRecurrence.down_quantization_fallback(5.0), PrecisionFormat::Int5);

    // Latent Bottleneck: Prioritizes Posit tapered precision around 0 dBFS
    assert_eq!(LayerRole::LatentBottleneck.down_quantization_fallback(6.0), PrecisionFormat::Posit16);
    assert_eq!(LayerRole::LatentBottleneck.down_quantization_fallback(3.0), PrecisionFormat::Posit8);

    // Ambisonic / Filters: Preserves unitary 3D rotation phase
    assert_eq!(LayerRole::AmbisonicRotation.down_quantization_fallback(12.0), PrecisionFormat::Fp32);
    assert_eq!(LayerRole::AmbisonicRotation.down_quantization_fallback(6.0), PrecisionFormat::Fp16);

    // Projections: Scales smoothly down to coarse integer / ternary
    assert_eq!(LayerRole::DenseProjection.down_quantization_fallback(1.5), PrecisionFormat::Ternary158);
}

#[test]
fn test_simd_kernels_posit_int8_bf16() {
    use inference::kernels;

    let activations = vec![1.0f32, -0.5, 2.0, 0.25];
    let mut out_posit = [0.0f32; 1];
    let mut out_int8 = [0.0f32; 1];
    let mut out_bf16 = [0.0f32; 1];

    // Posit8 kernel test
    let posit_raw = vec![0x40u8, 0x40, 0x40, 0x40]; // Some non-zero posit bytes
    kernels::posit8_matmul_simd_f32(&posit_raw, &activations, &mut out_posit, 1.0);
    assert!(out_posit[0].is_finite());

    // INT8 kernel test
    let int8_weights = vec![127i8, -64, 32, -16];
    kernels::int8_matmul_simd_f32(&int8_weights, &activations, &mut out_int8, 1.0);
    assert!(out_int8[0].is_finite());

    // BF16 kernel test
    let bf16_weights = vec![0x3F80u16, 0x3F80, 0x3F80, 0x3F80]; // 1.0 in BF16 is 0x3F80
    kernels::bf16_matmul_simd_f32(&bf16_weights, &activations, &mut out_bf16, 1.0);
    assert!(out_bf16[0].is_finite());
    // Sum = 1.0*1.0 + 1.0*(-0.5) + 1.0*2.0 + 1.0*0.25 = 2.75
    assert!((out_bf16[0] - 2.75).abs() < 1e-3);
}

