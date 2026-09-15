//! RainAI domain models, Macro-to-Micro conditioning mapping, and audio synthesis state.

use serde::{Deserialize, Serialize};

pub const CONDITION_DIM: usize = 554;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QualityTier {
    Ternary158,
    #[default]
    AdaptiveMinimum,
    HighInt16,
    StudioFp32,
}

impl QualityTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Ternary158 => "Ternary 1.58-Bit (Ultra-Fast / Add-Only)",
            Self::AdaptiveMinimum => "Adaptive QAT (Fast Startup)",
            Self::HighInt16 => "High Quality (INT16 On-Demand)",
            Self::StudioFp32 => "Studio Master (FP32 On-Demand)",
        }
    }

    pub fn download_size_label(self) -> &'static str {
        match self {
            Self::Ternary158 => "~1.1 MB (Included / Instant)",
            Self::AdaptiveMinimum => "~2.4 MB (Included)",
            Self::HighInt16 => "~9.8 MB (Download)",
            Self::StudioFp32 => "~38.4 MB (Download)",
        }
    }

    pub fn is_download_required(self) -> bool {
        match self {
            Self::Ternary158 | Self::AdaptiveMinimum => false,
            Self::HighInt16 | Self::StudioFp32 => true,
        }
    }
}

/// Continuous mixture over 9 physical materials
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SurfaceMixture {
    pub tin: f32,
    pub leaves_broad: f32,
    pub pine_needles: f32,
    pub pavement: f32,
    pub water_deep: f32,
    pub puddle_shallow: f32,
    pub canvas_tent: f32,
    pub glass_window: f32,
    pub wood_deck: f32,
}

impl Default for SurfaceMixture {
    fn default() -> Self {
        Self {
            tin: 0.1,
            leaves_broad: 0.2,
            pine_needles: 0.1,
            pavement: 0.15,
            water_deep: 0.05,
            puddle_shallow: 0.15,
            canvas_tent: 0.05,
            glass_window: 0.1,
            wood_deck: 0.1,
        }
    }
}

impl SurfaceMixture {
    /// Normalizes the surface values using a partition of unity (sum to 1.0)
    pub fn normalized(&self) -> [f32; 9] {
        let raw = [
            self.tin.max(0.0),
            self.leaves_broad.max(0.0),
            self.pine_needles.max(0.0),
            self.pavement.max(0.0),
            self.water_deep.max(0.0),
            self.puddle_shallow.max(0.0),
            self.canvas_tent.max(0.0),
            self.glass_window.max(0.0),
            self.wood_deck.max(0.0),
        ];
        let sum: f32 = raw.iter().sum();
        if sum <= 1e-6 {
            [1.0 / 9.0; 9]
        } else {
            let mut out = [0.0; 9];
            for i in 0..9 {
                out[i] = raw[i] / sum;
            }
            out
        }
    }

    pub fn set_preset_forest(&mut self) {
        self.tin = 0.0;
        self.leaves_broad = 0.45;
        self.pine_needles = 0.35;
        self.pavement = 0.0;
        self.water_deep = 0.05;
        self.puddle_shallow = 0.1;
        self.canvas_tent = 0.0;
        self.glass_window = 0.0;
        self.wood_deck = 0.05;
    }

    pub fn set_preset_urban(&mut self) {
        self.tin = 0.25;
        self.leaves_broad = 0.05;
        self.pine_needles = 0.0;
        self.pavement = 0.40;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.15;
        self.canvas_tent = 0.0;
        self.glass_window = 0.15;
        self.wood_deck = 0.0;
    }

    pub fn set_preset_tent(&mut self) {
        self.tin = 0.0;
        self.leaves_broad = 0.15;
        self.pine_needles = 0.15;
        self.pavement = 0.0;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.1;
        self.canvas_tent = 0.60;
        self.glass_window = 0.0;
        self.wood_deck = 0.0;
    }

    pub fn set_preset_window(&mut self) {
        self.tin = 0.1;
        self.leaves_broad = 0.05;
        self.pine_needles = 0.0;
        self.pavement = 0.1;
        self.water_deep = 0.0;
        self.puddle_shallow = 0.05;
        self.canvas_tent = 0.0;
        self.glass_window = 0.70;
        self.wood_deck = 0.0;
    }
}

/// Wind physics parameters
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WindParameters {
    pub speed: f32,
    pub gustiness: f32,
    pub turbulence: f32,
    pub howl: f32,
}

impl Default for WindParameters {
    fn default() -> Self {
        Self {
            speed: 0.35,
            gustiness: 0.2,
            turbulence: 0.15,
            howl: 0.1,
        }
    }
}

/// Spatialized side sounds positioned in the First-Order Ambisonic field
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SideSounds {
    // Insects
    pub insect_density: f32,
    pub insect_proximity: f32,
    pub insect_azimuth: f32, // -1.0 to 1.0 mapped to -pi to pi

    // Birds
    pub bird_activity: f32,
    pub bird_proximity: f32,
    pub bird_elevation: f32, // 0.0 to 1.0 mapped to 0 to pi/2

    // Fireplace
    pub fireplace_intensity: f32,
    pub fireplace_crackle_rate: f32,
    pub fireplace_azimuth: f32,
    pub fireplace_elevation: f32,

    // Thunder
    pub thunder_proximity: f32,
    pub thunder_rumble_length: f32,
    pub thunder_azimuth: f32,
    pub thunder_elevation: f32,

    // Traffic
    pub traffic_distance: f32,
    pub traffic_wetness: f32,
    pub traffic_azimuth_start: f32,
    pub traffic_azimuth_end: f32,
}

impl Default for SideSounds {
    fn default() -> Self {
        Self {
            insect_density: 0.1,
            insect_proximity: 0.8,
            insect_azimuth: 0.25,

            bird_activity: 0.05,
            bird_proximity: 0.9,
            bird_elevation: 0.4,

            fireplace_intensity: 0.0,
            fireplace_crackle_rate: 0.4,
            fireplace_azimuth: -0.3,
            fireplace_elevation: 0.0,

            thunder_proximity: 0.0,
            thunder_rumble_length: 0.6,
            thunder_azimuth: 0.7,
            thunder_elevation: 0.6,

            traffic_distance: 0.0,
            traffic_wetness: 0.8,
            traffic_azimuth_start: -0.8,
            traffic_azimuth_end: 0.8,
        }
    }
}

/// Base weather and acoustic space parameters
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BaseWeather {
    pub intensity: f32,
    pub runoff: f32,
    pub temperature: f32,
    pub humidity: f32,
    pub pitch_angle: f32,
    pub distance: f32,
    pub enclosure: f32, // 0.0: Outside in the open, 1.0: Deep indoors behind glass
}

impl Default for BaseWeather {
    fn default() -> Self {
        Self {
            intensity: 0.5,
            runoff: 0.4,
            temperature: 0.6,
            humidity: 0.85,
            pitch_angle: 0.1,
            distance: 0.3,
            enclosure: 0.2,
        }
    }
}

/// Real-time engine telemetry reported back to egui
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EngineTelemetry {
    pub buffer_health_ms: f32,
    pub cpu_headroom: f32,
    pub gpu_headroom: f32,
    pub delta_t_ms: f32,
    pub active_experts: usize,
    pub panic_factor: f32,
    pub jitter_factor: f32,
    pub quality_critic_score: f32,
    pub synthesis_blend: f32,
    pub effective_quant_floor: f32,
    pub effective_quant_ceiling: f32,
    pub governor_status: String,
    pub active_path_label: String,
    pub active_quantization_format: String,
    pub is_prebuffered: bool,
    pub active_optimization_profile: String,
    pub active_stress_profile: String,
    pub diffusion_bypassed: bool,
    pub ambisonic_order_reduced: bool,
}

impl Default for EngineTelemetry {
    fn default() -> Self {
        Self {
            buffer_health_ms: 45.0,
            cpu_headroom: 0.88,
            gpu_headroom: 0.92,
            delta_t_ms: 10.0,
            active_experts: 8,
            panic_factor: 0.0,
            jitter_factor: 0.02,
            quality_critic_score: 0.96,
            synthesis_blend: 0.0,
            effective_quant_floor: 1.58,
            effective_quant_ceiling: 8.0,
            governor_status: "Optimal Headroom".into(),
            active_path_label: "CPU Neural SIMD".into(),
            active_quantization_format: "INT8 (8-Bit, 256 Levels)".into(),
            is_prebuffered: false,
            active_optimization_profile: "Balanced Adaptive (Default)".into(),
            active_stress_profile: "0: Nominal Desktop (Pristine)".into(),
            diffusion_bypassed: false,
            ambisonic_order_reduced: false,
        }
    }
}

/// Operational optimization profiles for the Meta-Governor
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GovernorOptimizationProfile {
    EcoBatterySaver,
    LowLatencyInteractive,
    #[default]
    BalancedAdaptive,
    StudioMaster,
    BluetoothA2DPSink,
}

impl GovernorOptimizationProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => "Eco Battery Saver (<0.5W, ≤4b, 2 Exp)",
            Self::LowLatencyInteractive => "Low-Latency Interactive (15ms Buffer, Snappy)",
            Self::BalancedAdaptive => "Balanced Adaptive (45ms, 1.2s Hysteresis)",
            Self::StudioMaster => "Studio Master (≥16b Floor, 8 Exp, 120ms)",
            Self::BluetoothA2DPSink => "Bluetooth A2DP Sink (150ms Safety Reserve)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => "Eco Battery",
            Self::LowLatencyInteractive => "Low Latency",
            Self::BalancedAdaptive => "Balanced",
            Self::StudioMaster => "Studio Master",
            Self::BluetoothA2DPSink => "Bluetooth Sink",
        }
    }

    pub fn target_buffer_ms(self) -> f32 {
        match self {
            Self::EcoBatterySaver => 30.0,
            Self::LowLatencyInteractive => 15.0,
            Self::BalancedAdaptive => 45.0,
            Self::StudioMaster => 120.0,
            Self::BluetoothA2DPSink => 150.0,
        }
    }

    pub fn max_experts(self) -> usize {
        match self {
            Self::EcoBatterySaver => 2,
            Self::LowLatencyInteractive => 4,
            Self::BalancedAdaptive => 6,
            Self::StudioMaster => 8,
            Self::BluetoothA2DPSink => 6,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::EcoBatterySaver => "Extreme power saving mode capping bit-width at ≤4b, using 2 MoE experts, and keeping 50% procedural blend for thermal budget under 0.5W.",
            Self::LowLatencyInteractive => "Ultra-fast response with a tight 15ms buffer and quick fallback recovery for live real-time slider scrubbing.",
            Self::BalancedAdaptive => "Default operational profile with 45ms target buffer, 6 experts, and 1.2s anti-hunting hysteresis.",
            Self::StudioMaster => "Pristine audio priority locking a ≥16b precision floor, all 8 MoE experts, 0% procedural blend, and 120ms buffer reserve.",
            Self::BluetoothA2DPSink => "Extended 150ms safety reserve with jitter damping to prevent underruns on high-latency wireless audio sinks.",
        }
    }
}

/// Simulated hardware stress profiles matching RainAI deployment specification
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HardwareStressProfile {
    #[default]
    NominalDesktop,
    ThermalThrottlingCascade,
    GcWebAudioMicroStalls,
    UnifiedMemoryBusContention,
    DynamicGameDawInterference,
    BluetoothA2dpAudioSink,
    EcoSleepSoundscapeMode,
    HeterogeneousEcoreAsymmetry,
}

impl HardwareStressProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::NominalDesktop => "0: Nominal Desktop (Pristine)",
            Self::ThermalThrottlingCascade => "1: Thermal Throttling Cascade (Clock Collapse)",
            Self::GcWebAudioMicroStalls => "2: GC / WebAudio Micro-Stalls (Periodic Pauses)",
            Self::UnifiedMemoryBusContention => "3: Unified Memory Bus Contention (Transfer Choke)",
            Self::DynamicGameDawInterference => "4: Dynamic Game/DAW Interference (Host Bursts)",
            Self::BluetoothA2dpAudioSink => "5: Bluetooth A2DP Audio Sink (Jitter & Latency)",
            Self::EcoSleepSoundscapeMode => "6: Eco Sleep Soundscape Mode (Low-Power Throttle)",
            Self::HeterogeneousEcoreAsymmetry => "7: Heterogeneous E-Core Asymmetry (Core Bouncing)",
        }
    }

    pub fn profile_id(self) -> u8 {
        match self {
            Self::NominalDesktop => 0,
            Self::ThermalThrottlingCascade => 1,
            Self::GcWebAudioMicroStalls => 2,
            Self::UnifiedMemoryBusContention => 3,
            Self::DynamicGameDawInterference => 4,
            Self::BluetoothA2dpAudioSink => 5,
            Self::EcoSleepSoundscapeMode => 6,
            Self::HeterogeneousEcoreAsymmetry => 7,
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::NominalDesktop => "Unconstrained execution with maximum CPU/GPU headroom and minimal jitter.",
            Self::ThermalThrottlingCascade => "Simulates thermal clock collapse with progressive compute latency and rising panic factor.",
            Self::GcWebAudioMicroStalls => "Injects periodic 5-25ms garbage collector pauses into the buffer pipeline.",
            Self::UnifiedMemoryBusContention => "Simulates memory bandwidth saturation, inducing jitter in weight/latent tensor transfers.",
            Self::DynamicGameDawInterference => "Simulates competing heavy background workloads with sudden high-priority thread spikes.",
            Self::BluetoothA2dpAudioSink => "Simulates wireless audio output with high transmission latency and variable packet dispatch.",
            Self::EcoSleepSoundscapeMode => "Aggressively restricts compute to ultra-low frequency and throttles background tasks.",
            Self::HeterogeneousEcoreAsymmetry => "Simulates thread migration bouncing between high-frequency P-cores and low-power E-cores.",
        }
    }
}

/// Colors of noise for subtractive synthesis shaping
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NoiseColor {
    White,
    #[default]
    Pink,
    Brown,
    Blue,
    Violet,
}

impl NoiseColor {
    pub fn label(self) -> &'static str {
        match self {
            Self::White => "White (0 dB/oct - Flat & Crisp Spray)",
            Self::Pink => "Pink (-3 dB/oct - Natural Rain)",
            Self::Brown => "Brown (-6 dB/oct - Deep, Warm & Heavy Rain)",
            Self::Blue => "Blue (+3 dB/oct - High Mist Spray)",
            Self::Violet => "Violet (+6 dB/oct - Sharp Needles on Tin)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::White => "White Noise",
            Self::Pink => "Pink Noise",
            Self::Brown => "Brown Noise",
            Self::Blue => "Blue Noise",
            Self::Violet => "Violet Noise",
        }
    }

    pub fn spectral_decay_power(self) -> f32 {
        match self {
            Self::White => 0.0,
            Self::Pink => 0.5,
            Self::Brown => 1.0,
            Self::Blue => -0.5,
            Self::Violet => -1.0,
        }
    }
}

/// Complete RainAI engine state
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RainState {
    pub is_playing: bool,
    pub master_volume: f32,
    pub quality_tier: QualityTier,
    pub noise_color: NoiseColor,
    pub evolve_enabled: bool,
    pub evolve_speed: f32,
    pub auto_quantize: bool,
    pub drift_time: f32,
    pub weather: BaseWeather,
    pub surfaces: SurfaceMixture,
    pub wind: WindParameters,
    pub side_sounds: SideSounds,
    pub preferred_format: Option<String>,
    pub optimization_profile: GovernorOptimizationProfile,
    pub stress_profile: HardwareStressProfile,
    #[serde(skip)]
    pub telemetry: EngineTelemetry,
}

impl Default for RainState {
    fn default() -> Self {
        Self {
            is_playing: false,
            master_volume: 0.8,
            quality_tier: QualityTier::AdaptiveMinimum,
            noise_color: NoiseColor::Pink,
            evolve_enabled: true,
            evolve_speed: 0.2,
            auto_quantize: true,
            drift_time: 0.0,
            weather: BaseWeather::default(),
            surfaces: SurfaceMixture::default(),
            wind: WindParameters::default(),
            side_sounds: SideSounds::default(),
            preferred_format: None,
            optimization_profile: GovernorOptimizationProfile::default(),
            stress_profile: HardwareStressProfile::default(),
            telemetry: EngineTelemetry::default(),
        }
    }
}

impl RainState {
    /// Steps the procedural Brownian weather drift to keep the soundscape alive
    pub fn step_procedural_drift(&mut self, dt: f32) {
        if !self.evolve_enabled {
            return;
        }

        self.drift_time += dt * self.evolve_speed;
        let t = self.drift_time;

        // Subtle harmonic wind gusting
        let wind_drift = (t * 0.4).sin() * 0.15 + (t * 1.1).cos() * 0.05;
        self.wind.gustiness = (self.wind.gustiness + wind_drift * dt).clamp(0.0, 1.0);

        // Slow atmospheric humidity & runoff drift
        let runoff_drift = (t * 0.15).sin() * 0.08;
        self.weather.runoff = (self.weather.runoff + runoff_drift * dt).clamp(0.1, 1.0);

        // Gentle insect/bird diurnal fluctuation
        let bio_drift = (t * 0.25).cos() * 0.05;
        if self.side_sounds.bird_activity > 0.02 {
            self.side_sounds.bird_activity = (self.side_sounds.bird_activity + bio_drift * dt).clamp(0.0, 0.8);
        }
    }

    /// Converts current UI parameters into the 554-dim conditioning vector as a fixed array without heap allocation
    pub fn to_conditioning_array(&self) -> [f32; CONDITION_DIM] {
        let mut u = [0.0f32; CONDITION_DIM];

        // 1. CLAP acoustic semantic projection (512-dim)
        let clap_val = 1.0 / (512.0f32).sqrt();
        for i in 0..512 {
            u[i] = clap_val;
        }

        // 2. Base Sliders (10-dim: 512..522)
        let legacy_surface = 0.33;
        let wind_azimuth = 0.5;
        u[512] = self.weather.intensity;
        u[513] = self.wind.speed;
        u[514] = wind_azimuth;
        u[515] = legacy_surface;
        u[516] = self.weather.runoff;
        u[517] = self.weather.temperature;
        u[518] = self.weather.humidity;
        u[519] = self.weather.pitch_angle;
        u[520] = self.weather.distance;
        u[521] = self.weather.enclosure;

        // 3. Normalized Surfaces (9-dim: 522..531)
        let surf = self.surfaces.normalized();
        u[522..531].copy_from_slice(&surf);

        // 4. Wind Dynamics (4-dim: 531..535)
        u[531] = self.wind.speed;
        u[532] = self.wind.gustiness;
        u[533] = self.wind.turbulence;
        u[534] = self.wind.howl;

        // 5. Spatialized Side Sounds (18-dim: 535..553)
        // Insects (3)
        u[535] = self.side_sounds.insect_density;
        u[536] = self.side_sounds.insect_proximity;
        u[537] = self.side_sounds.insect_azimuth;

        // Birds (3)
        u[538] = self.side_sounds.bird_activity;
        u[539] = self.side_sounds.bird_proximity;
        u[540] = self.side_sounds.bird_elevation;

        // Fireplace (4)
        u[541] = self.side_sounds.fireplace_intensity;
        u[542] = self.side_sounds.fireplace_crackle_rate;
        u[543] = self.side_sounds.fireplace_azimuth;
        u[544] = self.side_sounds.fireplace_elevation;

        // Thunder (4)
        u[545] = self.side_sounds.thunder_proximity;
        u[546] = self.side_sounds.thunder_rumble_length;
        u[547] = self.side_sounds.thunder_azimuth;
        u[548] = self.side_sounds.thunder_elevation;

        // Traffic (4)
        u[549] = self.side_sounds.traffic_distance;
        u[550] = self.side_sounds.traffic_wetness;
        u[551] = self.side_sounds.traffic_azimuth_start;
        u[552] = self.side_sounds.traffic_azimuth_end;

        // 6. Physics parameter drift tolerance (1-dim: 553)
        u[553] = 0.2;

        u
    }

    /// Converts the current UI parameters into the exact 554-dimensional conditioning vector u
    pub fn to_conditioning_vector(&self) -> Vec<f32> {
        self.to_conditioning_array().to_vec()
    }
}
