use shared::{
    ItemCollection, Priority, export_to_csv, export_to_json, import_from_csv, import_from_json,
};

#[test]
fn test_item_collection_defaults_and_operations() {
    let mut collection = ItemCollection::default();
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);
    assert_eq!(collection.completion_ratio(), 0.0);

    let id = collection.add("New Task", "Task Description", Priority::High);
    assert_eq!(collection.total_count(), 4);

    collection.toggle(id);
    assert_eq!(collection.completed_count(), 1);
    assert_eq!(collection.completion_ratio(), 0.25);

    let removed = collection.remove(id);
    assert!(removed);
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);
}

#[test]
fn test_json_roundtrip() {
    let collection = ItemCollection::default();
    let json = export_to_json(&collection).expect("Failed to export JSON");
    assert!(json.contains("Explore Cross-Platform State Persistence"));

    let imported = import_from_json(&json).expect("Failed to import JSON");
    assert_eq!(imported.items.len(), collection.items.len());
    assert_eq!(imported.items[0].title, collection.items[0].title);
}

#[test]
fn test_csv_roundtrip() {
    let collection = ItemCollection::default();
    let csv = export_to_csv(&collection);
    assert!(csv.starts_with("id,title,description,priority,completed"));

    let imported = import_from_csv(&csv).expect("Failed to import CSV");
    assert_eq!(imported.items.len(), collection.items.len());
    assert_eq!(imported.items[0].title, collection.items[0].title);
    assert_eq!(imported.items[0].priority, collection.items[0].priority);
}

#[test]
fn test_clear_completed() {
    let mut collection = ItemCollection::default();
    collection.toggle(1);
    assert_eq!(collection.completed_count(), 1);

    collection.clear_completed();
    assert_eq!(collection.total_count(), 2);
    assert_eq!(collection.completed_count(), 0);
}

#[test]
fn test_compressed_bson_roundtrip() {
    let mut collection = ItemCollection::default();
    collection.add(
        "BSON Persistence Item",
        "State compressed with zlib",
        Priority::High,
    );

    let bytes = shared::export_to_compressed_bson(&collection).expect("BSON export failed");
    assert!(!bytes.is_empty());

    let restored = shared::import_from_compressed_bson(&bytes).expect("BSON import failed");
    assert_eq!(restored.total_count(), 4);
    assert_eq!(restored.items[3].title, "BSON Persistence Item");
}

#[test]
fn test_state_backward_compatibility() {
    let legacy_json = r#"{"config":{},"collection":{"items":[]}}"#;
    let app_state: Result<shared::AppState, _> = serde_json::from_str(legacy_json);
    assert!(
        app_state.is_ok(),
        "Should parse legacy state without failure"
    );
}

#[test]
fn test_rain_state_conditioning_vector_and_drift() {
    use shared::{RainState, QualityTier};
    let mut rain = RainState::default();
    assert_eq!(rain.quality_tier, QualityTier::AdaptiveMinimum);
    assert!(!rain.quality_tier.is_download_required());
    assert!(!QualityTier::Ternary158.is_download_required());
    assert!(QualityTier::HighInt16.is_download_required());
    assert!(QualityTier::StudioFp32.is_download_required());

    // Verify NoiseColor switching to Brown
    use shared::NoiseColor;
    rain.noise_color = NoiseColor::Brown;
    assert_eq!(rain.noise_color.spectral_decay_power(), 1.0);
    assert!(rain.noise_color.short_label().contains("Brown"));

    // Verify 554 conditioning dimension
    let u = rain.to_conditioning_vector();
    assert_eq!(u.len(), 554, "Conditioning vector must match exact 554 model dimension");

    // Verify surface partition of unity
    let normalized = rain.surfaces.normalized();
    let sum: f32 = normalized.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "Surface partition of unity must sum to 1.0");

    // Verify procedural drift advances
    let prev_t = rain.drift_time;
    rain.step_procedural_drift(0.5);
    assert!(rain.drift_time > prev_t);
}
