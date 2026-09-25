//! Shared Domain Specifications, State Management, and Parameter Models for RainAI.
//!
//! Provides common definitions for physical rain parameters, surface material categories,
//! Ambisonic decoding modes, hardware telemetry metrics, and audio export settings across
//! native desktop, web browser, and audio synthesis runtimes.
//!
//! # Architecture & Modules
//!
//! - [`rain`]: Core physical domain model representing 9 distinct surface profiles, drop size
//!   distributions, wind coupling vectors, hardware governor telemetry, and 554-dimensional conditioning vectors.
//! - [`preset`]: Built-in factory presets (e.g., Attic Tin Roof, Amazon Canopy, Concrete Courtyard,
//!   Cabin Lake Pier) with instant serialization and interpolation curves.
//! - [`export`]: Lossless audio render configurations supporting sample rates up to 96 kHz, bit depths
//!   (16, 24, 32-bit float), and multi-channel Ambisonic layouts.
//! - [`models`]: General collection storage, undo/redo state trackers, and item persistence containers.

use serde::{Deserialize, Serialize};

pub mod conditioning;
pub mod export;
pub mod models;
pub mod paths;
pub mod preset;
pub mod rain;
pub mod surface;

pub use conditioning::*;
pub use export::*;
pub use models::*;
pub use paths::*;
pub use preset::*;
pub use rain::*;
pub use surface::*;

pub use spodeian_tokens::ThemeMode;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct AppConfig {
    pub theme: ThemeMode,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            theme: ThemeMode::Dark,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AppState {
    #[serde(default)]
    pub config: AppConfig,
    #[serde(default)]
    pub collection: ItemCollection,
    #[serde(default)]
    pub rain: RainState,
}

impl AppState {
    /// Resets app domain data while preserving configuration preferences like theme.
    pub fn reset_data(&mut self) {
        self.collection.reset();
    }
}
