//! Shared business logic, domain models, and configuration.

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
