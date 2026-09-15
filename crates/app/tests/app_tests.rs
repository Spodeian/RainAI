use app::TemplateApp;
use eframe::{App, Storage};
use shared::{AppState, Priority};
use std::collections::HashMap;

#[derive(Default)]
struct MockStorage {
    data: HashMap<String, String>,
}

impl Storage for MockStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.data.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.data.insert(key.to_owned(), value);
    }
    fn remove_string(&mut self, key: &str) {
        self.data.remove(key);
    }
    fn flush(&mut self) {}
}

#[test]
fn test_template_app_initialization() {
    let app = TemplateApp::default();
    assert_eq!(app.state.collection.total_count(), 3);
    assert_eq!(app.state.collection.completed_count(), 0);
}

#[test]
fn test_template_app_save_and_load() {
    let mut storage = MockStorage::default();
    let mut app = TemplateApp::default();

    // Add item and toggle
    let new_id = app.state.collection.add(
        "Persistent Task",
        "Should be saved in storage",
        Priority::High,
    );
    app.state.collection.toggle(new_id);

    // Save
    app.save(&mut storage);

    // Verify storage contents
    let serialized = storage
        .get_string(eframe::APP_KEY)
        .expect("App key must exist in storage");
    assert!(serialized.contains("Persistent Task"));

    // Simulate reload from storage
    let loaded_state: AppState = eframe::get_value(&storage, eframe::APP_KEY).unwrap();
    assert_eq!(loaded_state.collection.total_count(), 4);
    assert_eq!(loaded_state.collection.completed_count(), 1);
}

#[test]
fn test_template_app_export_dialog() {
    let mut app = TemplateApp::default();
    app.open_export_dialog(app::ExportFormat::Json);

    assert!(app.show_export_dialog.is_some());
    assert!(
        app.export_text_buffer
            .contains("Explore Cross-Platform State Persistence")
    );
}

#[test]
fn test_template_app_load_multi_tier_json_and_ron() {
    use app::storage_manager::{
        DEDICATED_STORAGE_KEY, deserialize_app_state, load_state_multi_tier,
    };

    let mut original_state = AppState::default();
    original_state
        .collection
        .add("Task A", "Description A", Priority::High);

    // 1. JSON roundtrip
    let json_str = serde_json::to_string(&original_state).unwrap();
    let from_json = deserialize_app_state(&json_str).expect("JSON should deserialize");
    assert_eq!(from_json.collection.total_count(), 4);

    // 2. RON roundtrip
    let ron_str = ron::to_string(&original_state).unwrap();
    let from_ron = deserialize_app_state(&ron_str).expect("RON should deserialize");
    assert_eq!(from_ron.collection.total_count(), 4);

    // 3. Load from dedicated key in storage (JSON)
    let mut storage_dedicated = MockStorage::default();
    storage_dedicated.set_string(DEDICATED_STORAGE_KEY, json_str.clone());
    let loaded_dedicated =
        load_state_multi_tier(Some(&storage_dedicated)).expect("Should load from dedicated key");
    assert_eq!(loaded_dedicated.collection.total_count(), 4);

    // 4. Load from legacy app key in storage (JSON)
    let mut storage_json_app = MockStorage::default();
    storage_json_app.set_string(eframe::APP_KEY, json_str);
    let loaded_json =
        load_state_multi_tier(Some(&storage_json_app)).expect("Should load from app key JSON");
    assert_eq!(loaded_json.collection.total_count(), 4);

    // 5. Load from legacy app key in storage (RON)
    let mut storage_ron_app = MockStorage::default();
    storage_ron_app.set_string(eframe::APP_KEY, ron_str);
    let loaded_ron =
        load_state_multi_tier(Some(&storage_ron_app)).expect("Should load from app key RON");
    assert_eq!(loaded_ron.collection.total_count(), 4);
}

#[test]
fn test_template_app_save_populates_both_keys() {
    use app::storage_manager::{DEDICATED_STORAGE_KEY, load_state_multi_tier};

    let mut storage = MockStorage::default();
    let mut app = TemplateApp::default();
    app.state.collection.add(
        "New Saved Item",
        "Active Persistence Test",
        Priority::Medium,
    );

    app.save(&mut storage);

    // Both eframe::APP_KEY (RON) and DEDICATED_STORAGE_KEY (JSON) should be populated
    assert!(storage.get_string(eframe::APP_KEY).is_some());
    assert!(storage.get_string(DEDICATED_STORAGE_KEY).is_some());

    let loaded = load_state_multi_tier(Some(&storage)).expect("Should restore state successfully");
    assert_eq!(loaded.collection.total_count(), 4);
}

#[test]
fn test_template_app_audio_prebuffering_and_radar_defaults() {
    let mut app = TemplateApp::default();
    assert_eq!(app.rain_view.dragged_source, None);
    assert!(!app.rain_view.enable_gpu_radar);
    assert_eq!(app.rain_view.export_duration, 30.0);

    // Verify ensure_audio_engine initializes audio engine without requiring is_playing = true
    app.ensure_audio_engine();
    assert!(app.audio_state.is_some(), "Audio state must be initialized for pre-buffering");

    app.sync_audio_engine();
    assert!(app.audio_state.is_some());
}

