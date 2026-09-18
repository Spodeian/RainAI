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

pub mod export;
pub mod models;
pub mod preset;
pub mod rain;

pub use export::*;
pub use models::*;
pub use preset::*;
pub use rain::*;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThemeMode {
    Light,
    #[default]
    Dark,
    HighContrastDark,
    HighContrastLight,
}

impl ThemeMode {
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::HighContrastDark,
            Self::HighContrastDark => Self::HighContrastLight,
            Self::HighContrastLight => Self::Dark,
        }
    }

    #[must_use]
    pub fn is_dark(self) -> bool {
        matches!(self, Self::Dark | Self::HighContrastDark)
    }

    #[must_use]
    pub fn is_high_contrast(self) -> bool {
        matches!(self, Self::HighContrastDark | Self::HighContrastLight)
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Warm Light",
            Self::HighContrastDark => "HC Dark",
            Self::HighContrastLight => "HC Light",
        }
    }

    #[must_use]
    pub fn icon(self) -> &'static str {
        match self {
            Self::Dark => "🌙",
            Self::Light => "☀️",
            Self::HighContrastDark => "⬛",
            Self::HighContrastLight => "⬜",
        }
    }
}

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
