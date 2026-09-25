//! Import and Export handlers for CSV and JSON persistence interchange.

use crate::models::{Item, ItemCollection, Priority};
pub use spodeian_export::{escape_csv_field, parse_csv_line};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataTransferError {
    #[error("JSON serialization error: {0}")]
    JsonSerialize(#[from] serde_json::Error),
    #[error("Export error: {0}")]
    Export(#[from] spodeian_export::ExportError),
    #[error("Failed to parse CSV: {0}")]
    CsvParse(String),
    #[error("Invalid priority '{0}', expected Low, Medium, or High")]
    InvalidPriority(String),
}

/// Serializes the entire collection to formatted JSON.
pub fn export_to_json(collection: &ItemCollection) -> Result<String, DataTransferError> {
    spodeian_export::export_to_json(collection).map_err(DataTransferError::Export)
}

/// Deserializes a JSON string into an `ItemCollection`.
pub fn import_from_json(json_str: &str) -> Result<ItemCollection, DataTransferError> {
    spodeian_export::import_from_json(json_str).map_err(DataTransferError::Export)
}

/// Serializes the collection to CSV format.
pub fn export_to_csv(collection: &ItemCollection) -> String {
    let mut csv = String::from("id,title,description,priority,completed\n");
    for item in &collection.items {
        let escaped_title = escape_csv_field(&item.title);
        let escaped_desc = escape_csv_field(&item.description);
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            item.id,
            escaped_title,
            escaped_desc,
            item.priority.label(),
            item.completed
        ));
    }
    csv
}

/// Imports items from a CSV string, appending them or creating a fresh collection.
pub fn import_from_csv(csv_str: &str) -> Result<ItemCollection, DataTransferError> {
    let mut items = Vec::new();
    let mut max_id: u64 = 0;

    for (line_idx, line) in csv_str.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || (line_idx == 0 && trimmed.to_lowercase().starts_with("id,")) {
            continue;
        }

        let fields = parse_csv_line(trimmed);
        if fields.len() < 5 {
            return Err(DataTransferError::CsvParse(format!(
                "Line {}: Expected 5 columns (id,title,description,priority,completed), found {}",
                line_idx + 1,
                fields.len()
            )));
        }

        let id = fields[0].parse::<u64>().unwrap_or_else(|_| max_id + 1);
        let title = fields[1].clone();
        let description = fields[2].clone();
        let priority = match fields[3].to_lowercase().as_str() {
            "low" => Priority::Low,
            "medium" => Priority::Medium,
            "high" => Priority::High,
            other => return Err(DataTransferError::InvalidPriority(other.to_string())),
        };
        let completed = fields[4].eq_ignore_ascii_case("true") || fields[4] == "1";

        if id > max_id {
            max_id = id;
        }

        items.push(Item {
            id,
            title,
            description,
            priority,
            completed,
        });
    }

    Ok(ItemCollection {
        items,
        next_id: max_id + 1,
    })
}

/// Exports the collection into compressed BSON binary bytes (Zlib-compressed BSON).
pub fn export_to_compressed_bson(collection: &ItemCollection) -> Result<Vec<u8>, String> {
    spodeian_export::export_to_compressed_bson(collection).map_err(|e| e.to_string())
}

/// Imports and restores an ItemCollection from a compressed (or raw) BSON slice.
pub fn import_from_compressed_bson(bytes: &[u8]) -> Result<ItemCollection, String> {
    spodeian_export::import_from_compressed_bson(bytes).map_err(|e| e.to_string())
}
