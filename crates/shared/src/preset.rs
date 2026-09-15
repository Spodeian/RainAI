//! Curated weather soundscapes and preset serialization for RainAI.

use crate::rain::{
    BaseWeather, NoiseColor, QualityTier, RainState, SideSounds, SurfaceMixture, WindParameters,
};
use serde::{Deserialize, Serialize};

/// A named weather and acoustic preset
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WeatherPreset {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub state: RainState,
}

impl WeatherPreset {
    pub fn new(name: impl Into<String>, description: impl Into<String>, state: RainState) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            tags: Vec::new(),
            state,
        }
    }

    pub fn with_tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Serialize preset to pretty JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize preset from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Compress preset to base64 string for URL sharing
    pub fn to_shareable_url_hash(&self) -> Result<String, String> {
        let json = self.to_json().map_err(|e| e.to_string())?;
        let compressed = miniz_oxide::deflate::compress_to_vec(json.as_bytes(), 6);
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            compressed,
        ))
    }

    /// Decode preset from base64 string
    pub fn from_shareable_url_hash(encoded: &str) -> Result<Self, String> {
        let compressed = base64::Engine::decode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            encoded,
        )
        .map_err(|e| format!("Base64 decode error: {e}"))?;
        let decompressed = miniz_oxide::inflate::decompress_to_vec(&compressed)
            .map_err(|e| format!("Decompress error: {e:?}"))?;
        let json_str = std::str::from_utf8(&decompressed)
            .map_err(|e| format!("UTF-8 error: {e}"))?;
        Self::from_json(json_str).map_err(|e| e.to_string())
    }

    /// Default curated presets spanning diverse environments
    pub fn builtins() -> Vec<Self> {
        vec![
            Self::new(
                "Gentle Meadow Drizzle",
                "Light soothing droplets falling on broad leaves and pine needles with gentle breeze and distant birds.",
                RainState {
                    is_playing: true,
                    master_volume: 0.8,
                    quality_tier: QualityTier::AdaptiveMinimum,
                    noise_color: NoiseColor::Pink,
                    evolve_enabled: true,
                    evolve_speed: 0.2,
                    drift_time: 0.0,
                    weather: BaseWeather {
                        intensity: 0.25,
                        runoff: 0.2,
                        temperature: 0.65,
                        humidity: 0.8,
                        pitch_angle: 0.05,
                        distance: 0.4,
                        enclosure: 0.0,
                    },
                    surfaces: SurfaceMixture {
                        tin: 0.0,
                        leaves_broad: 0.45,
                        pine_needles: 0.35,
                        pavement: 0.0,
                        water_deep: 0.05,
                        puddle_shallow: 0.15,
                        canvas_tent: 0.0,
                        glass_window: 0.0,
                        wood_deck: 0.0,
                    },
                    wind: WindParameters {
                        speed: 0.15,
                        gustiness: 0.2,
                        turbulence: 0.1,
                        howl: 0.05,
                    },
                    side_sounds: SideSounds {
                        insect_density: 0.3,
                        insect_proximity: 0.7,
                        insect_azimuth: -0.4,
                        bird_activity: 0.2,
                        bird_proximity: 0.85,
                        bird_elevation: 0.4,
                        fireplace_intensity: 0.0,
                        fireplace_crackle_rate: 0.0,
                        fireplace_azimuth: 0.0,
                        fireplace_elevation: 0.0,
                        thunder_proximity: 0.0,
                        thunder_rumble_length: 0.0,
                        thunder_azimuth: 0.0,
                        thunder_elevation: 0.0,
                        traffic_distance: 0.0,
                        traffic_wetness: 0.0,
                        traffic_azimuth_start: 0.0,
                        traffic_azimuth_end: 0.0,
                    },
                    ..Default::default()
                },
            )
            .with_tags(&["relaxing", "nature", "gentle", "study"]),

            Self::new(
                "Thunderstorm on Tin Roof",
                "Energetic downpour striking corrugated tin sheets with close roaring thunder and whipping wind.",
                RainState {
                    is_playing: true,
                    master_volume: 0.9,
                    quality_tier: QualityTier::AdaptiveMinimum,
                    noise_color: NoiseColor::White,
                    evolve_enabled: true,
                    evolve_speed: 0.35,
                    drift_time: 0.0,
                    weather: BaseWeather {
                        intensity: 0.95,
                        runoff: 0.8,
                        temperature: 0.45,
                        humidity: 0.95,
                        pitch_angle: 0.35,
                        distance: 0.15,
                        enclosure: 0.2,
                    },
                    surfaces: SurfaceMixture {
                        tin: 0.75,
                        leaves_broad: 0.05,
                        pine_needles: 0.0,
                        pavement: 0.1,
                        water_deep: 0.0,
                        puddle_shallow: 0.1,
                        canvas_tent: 0.0,
                        glass_window: 0.0,
                        wood_deck: 0.0,
                    },
                    wind: WindParameters {
                        speed: 0.8,
                        gustiness: 0.85,
                        turbulence: 0.7,
                        howl: 0.6,
                    },
                    side_sounds: SideSounds {
                        insect_density: 0.0,
                        insect_proximity: 0.0,
                        insect_azimuth: 0.0,
                        bird_activity: 0.0,
                        bird_proximity: 0.0,
                        bird_elevation: 0.0,
                        fireplace_intensity: 0.0,
                        fireplace_crackle_rate: 0.0,
                        fireplace_azimuth: 0.0,
                        fireplace_elevation: 0.0,
                        thunder_proximity: 0.85,
                        thunder_rumble_length: 0.9,
                        thunder_azimuth: 0.3,
                        thunder_elevation: 0.75,
                        traffic_distance: 0.0,
                        traffic_wetness: 0.0,
                        traffic_azimuth_start: 0.0,
                        traffic_azimuth_end: 0.0,
                    },
                    ..Default::default()
                },
            )
            .with_tags(&["storm", "heavy", "thunder", "intense"]),

            Self::new(
                "Cozy Cabin Fireplace",
                "Gentle rain drumming against timber walls and glass windows while embers crackle in the hearth.",
                RainState {
                    is_playing: true,
                    master_volume: 0.85,
                    quality_tier: QualityTier::AdaptiveMinimum,
                    noise_color: NoiseColor::Pink,
                    evolve_enabled: true,
                    evolve_speed: 0.15,
                    drift_time: 0.0,
                    weather: BaseWeather {
                        intensity: 0.4,
                        runoff: 0.3,
                        temperature: 0.5,
                        humidity: 0.7,
                        pitch_angle: 0.1,
                        distance: 0.5,
                        enclosure: 0.75,
                    },
                    surfaces: SurfaceMixture {
                        tin: 0.0,
                        leaves_broad: 0.0,
                        pine_needles: 0.1,
                        pavement: 0.0,
                        water_deep: 0.0,
                        puddle_shallow: 0.0,
                        canvas_tent: 0.0,
                        glass_window: 0.45,
                        wood_deck: 0.45,
                    },
                    wind: WindParameters {
                        speed: 0.25,
                        gustiness: 0.3,
                        turbulence: 0.2,
                        howl: 0.15,
                    },
                    side_sounds: SideSounds {
                        insect_density: 0.0,
                        insect_proximity: 0.0,
                        insect_azimuth: 0.0,
                        bird_activity: 0.0,
                        bird_proximity: 0.0,
                        bird_elevation: 0.0,
                        fireplace_intensity: 0.75,
                        fireplace_crackle_rate: 0.55,
                        fireplace_azimuth: -0.35,
                        fireplace_elevation: -0.1,
                        thunder_proximity: 0.1,
                        thunder_rumble_length: 0.4,
                        thunder_azimuth: 0.8,
                        thunder_elevation: 0.5,
                        traffic_distance: 0.0,
                        traffic_wetness: 0.0,
                        traffic_azimuth_start: 0.0,
                        traffic_azimuth_end: 0.0,
                    },
                    ..Default::default()
                },
            )
            .with_tags(&["cozy", "cabin", "fireplace", "sleep"]),

            Self::new(
                "Urban Street & Wet Tires",
                "Rain falling on asphalt and gutters with the characteristic hiss of cars passing over drenched asphalt.",
                RainState {
                    is_playing: true,
                    master_volume: 0.8,
                    quality_tier: QualityTier::AdaptiveMinimum,
                    noise_color: NoiseColor::White,
                    evolve_enabled: true,
                    evolve_speed: 0.25,
                    drift_time: 0.0,
                    weather: BaseWeather {
                        intensity: 0.55,
                        runoff: 0.7,
                        temperature: 0.55,
                        humidity: 0.85,
                        pitch_angle: 0.15,
                        distance: 0.35,
                        enclosure: 0.1,
                    },
                    surfaces: SurfaceMixture {
                        tin: 0.05,
                        leaves_broad: 0.0,
                        pine_needles: 0.0,
                        pavement: 0.65,
                        water_deep: 0.0,
                        puddle_shallow: 0.3,
                        canvas_tent: 0.0,
                        glass_window: 0.0,
                        wood_deck: 0.0,
                    },
                    wind: WindParameters {
                        speed: 0.35,
                        gustiness: 0.4,
                        turbulence: 0.3,
                        howl: 0.2,
                    },
                    side_sounds: SideSounds {
                        insect_density: 0.0,
                        insect_proximity: 0.0,
                        insect_azimuth: 0.0,
                        bird_activity: 0.0,
                        bird_proximity: 0.0,
                        bird_elevation: 0.0,
                        fireplace_intensity: 0.0,
                        fireplace_crackle_rate: 0.0,
                        fireplace_azimuth: 0.0,
                        fireplace_elevation: 0.0,
                        thunder_proximity: 0.0,
                        thunder_rumble_length: 0.0,
                        thunder_azimuth: 0.0,
                        thunder_elevation: 0.0,
                        traffic_distance: 0.75,
                        traffic_wetness: 0.85,
                        traffic_azimuth_start: -0.85,
                        traffic_azimuth_end: 0.85,
                    },
                    ..Default::default()
                },
            )
            .with_tags(&["city", "pavement", "traffic", "focus"]),

            Self::new(
                "Deep Brown Noise Meditation",
                "Deep, warm, low-frequency Brownian rainfall optimized for deep sleep, insomnia relief, and noise masking.",
                RainState {
                    is_playing: true,
                    master_volume: 0.85,
                    quality_tier: QualityTier::AdaptiveMinimum,
                    noise_color: NoiseColor::Brown,
                    evolve_enabled: false,
                    evolve_speed: 0.0,
                    drift_time: 0.0,
                    weather: BaseWeather {
                        intensity: 0.6,
                        runoff: 0.3,
                        temperature: 0.6,
                        humidity: 0.9,
                        pitch_angle: 0.0,
                        distance: 0.5,
                        enclosure: 0.3,
                    },
                    surfaces: SurfaceMixture {
                        tin: 0.0,
                        leaves_broad: 0.3,
                        pine_needles: 0.3,
                        pavement: 0.0,
                        water_deep: 0.3,
                        puddle_shallow: 0.1,
                        canvas_tent: 0.0,
                        glass_window: 0.0,
                        wood_deck: 0.0,
                    },
                    wind: WindParameters {
                        speed: 0.1,
                        gustiness: 0.15,
                        turbulence: 0.1,
                        howl: 0.0,
                    },
                    side_sounds: SideSounds::default(),
                    ..Default::default()
                },
            )
            .with_tags(&["brown noise", "sleep", "meditation", "warm"]),
        ]
    }
}
