use inference::weight_loader::{WeightLoader, WeightCache, LoadedLayer};
use inference::runner::{InferenceRunner, EngineStatus};
use inference::model::{BoxCoxDequantizer, QuantizedModelManifest, PrecisionFormat, LayerRole, apply_equal_power_crossfade};
use inference::asset_manager::{AssetManager, AssetState, ExecutionPath};
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
    let layer = LoadedLayer {
        name: "encoder.cond_proj.weight".to_string(),
        shape: vec![64, 554],
        scale: 0.125,
        format: PrecisionFormat::Ternary158,
        weights: vec![],
        packed_weights: vec![0x05; (64 * 554 + 3) / 4],
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
    let mut mgr = AssetManager::new();
    let (eff, is_fallback) = mgr.resolve_effective_tier(QualityTier::StudioFp32);
    assert_eq!(eff, QualityTier::AdaptiveMinimum);
    assert!(is_fallback);
}
