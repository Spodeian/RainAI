use app::TemplateApp;
use shared::{AppState, Priority, ItemCollection, export_to_json, import_from_json, export_to_csv, import_from_csv};

#[test]
fn test_template_app_initialization() {
    let app = TemplateApp::default();
    assert_eq!(app.state.collection.total_count(), 3);
    assert_eq!(app.state.collection.completed_count(), 0);
}

#[test]
fn test_item_collection_defaults_and_operations() {
    let mut collection = ItemCollection::default();
    assert_eq!(collection.total_count(), 3);
    assert_eq!(collection.completed_count(), 0);

    let id = collection.add("New Task", "Task Description", Priority::High);
    assert_eq!(collection.total_count(), 4);

    collection.toggle(id);
    assert_eq!(collection.completed_count(), 1);
}

#[test]
fn test_json_roundtrip() {
    let collection = ItemCollection::default();
    let json = export_to_json(&collection).expect("Failed to export JSON");
    let imported = import_from_json(&json).expect("Failed to import JSON");
    assert_eq!(imported.items.len(), collection.items.len());
}
