//! Meta-Controller Dynamic Quantization Governor & Live Preference Mediator.
//!
//! Continuously evaluates real-time audio buffer health and compute headroom relative to dynamic
//! buffer sizing in ms and Bytes. Mediates live user preferences with physical momentum and handles
//! non-realtime offline export decoupling.

use inference::{ExecutionPath, MoeExecutionMode, PrecisionFormat};
use shared::rain::{
    EngineTelemetry, GovernorOptimizationProfile, HardwareStressProfile, MetaControllerInterceptionMode,
    QualityTier, RainState,
};

/// Autonomous actions decided by the Meta-Governor
#[derive(Clone, Debug, PartialEq)]
pub struct GovernorAction {
    pub recommended_tier: QualityTier,
    pub recommended_path: ExecutionPath,
    pub min_bits: f32,
    pub max_bits: f32,
    pub synthesis_blend: f32,
    pub status_label: &'static str,
    pub recommended_format: PrecisionFormat,
    pub recommended_moe_mode: MoeExecutionMode,
    pub active_experts: usize,
    pub diffusion_bypass: bool,
    pub ambisonic_order_reduced: bool,
    pub target_buffer_ms: f32,
    pub simulated_buffer_health_ms: f32,
    pub simulated_cpu_headroom: f32,
    pub simulated_panic_factor: f32,
    pub simulated_jitter_factor: f32,
    pub thinking_steps: usize,
    pub use_consistency_jump: bool,
    pub dynamic_buffer_bytes: usize,
    pub buffer_health_ratio: f32,
    pub target_capacity_frames: usize,
    pub env_max_buffer_bytes: usize,
    pub resize_commanded: bool,
    pub buffer_resize_cooldown: f32,
    pub quant_macro_cooldown: f32,
    pub buffer_resizes_count: usize,
    pub quant_swaps_count: usize,
}

/// Governor state machine with anti-hunting hysteresis and dynamic relative-health monitoring
#[derive(Clone, Debug)]
pub struct MetaGovernor {
    pub stability_timer: f32,
    pub stress_timer: f32,
    pub current_tier: QualityTier,
    pub current_path: ExecutionPath,
    pub current_format: PrecisionFormat,
    pub current_moe_mode: MoeExecutionMode,
    pub min_bits: f32,
    pub max_bits: f32,
    pub synthesis_blend: f32,
    pub active_experts: usize,
    pub diffusion_bypass: bool,
    pub ambisonic_order_reduced: bool,
    pub thinking_steps: usize,
    pub use_consistency_jump: bool,
    pub meta_controller_steps: Option<usize>,
    pub meta_controller_diffusion_bypass: Option<bool>,
    pub current_capacity_frames: usize,
    pub buffer_cooldown_timer: f32,
    pub quant_cooldown_timer: f32,
    pub total_buffer_resizes: usize,
    pub total_quant_swaps: usize,
}

impl Default for MetaGovernor {
    fn default() -> Self {
        Self {
            stability_timer: 0.0,
            stress_timer: 0.0,
            current_tier: QualityTier::AdaptiveMinimum,
            current_path: ExecutionPath::CpuNeural,
            current_format: PrecisionFormat::Int8,
            current_moe_mode: MoeExecutionMode::SparseDynamic,
            min_bits: 1.58,
            max_bits: 8.0,
            synthesis_blend: 0.0,
            active_experts: 8,
            diffusion_bypass: false,
            ambisonic_order_reduced: false,
            thinking_steps: 3,
            use_consistency_jump: false,
            meta_controller_steps: None,
            meta_controller_diffusion_bypass: None,
            current_capacity_frames: 4320,
            buffer_cooldown_timer: 2.0,
            quant_cooldown_timer: 15.0,
            total_buffer_resizes: 0,
            total_quant_swaps: 0,
        }
    }
}

impl MetaGovernor {
    pub fn compute_env_max_bytes(profile: GovernorOptimizationProfile, is_under_stress: bool) -> usize {
        match profile {
            GovernorOptimizationProfile::EcoBatterySaver => 24 * 1024,
            GovernorOptimizationProfile::LowLatencyInteractive => 32 * 1024,
            GovernorOptimizationProfile::BalancedAdaptive => {
                if is_under_stress { 32 * 1024 } else { 64 * 1024 }
            }
            GovernorOptimizationProfile::BluetoothA2DPSink => 48 * 1024,
            GovernorOptimizationProfile::StudioMaster => {
                if is_under_stress { 64 * 1024 } else { 128 * 1024 }
            }
        }
    }
    pub fn new() -> Self {
        Self::default()
    }

    /// Dynamically alters the pre-generated steps via the neural Meta-Controller
    pub fn set_meta_controller_steps(&mut self, steps: Option<usize>) {
        self.meta_controller_steps = steps.map(|s| s.clamp(1, 5));
    }

    /// Update governor directly with neural Meta-Controller output
    pub fn update_from_meta_controller(&mut self, recommended_steps: usize, stress: f32) {
        let clamped = recommended_steps.clamp(1, 5);
        self.meta_controller_steps = Some(clamped);
        if stress > 0.65 {
            self.meta_controller_diffusion_bypass = Some(true);
        } else {
            self.meta_controller_diffusion_bypass = None;
        }
    }

    /// Evaluates current telemetry against optimization profile, dynamic buffer size, and stress
    pub fn evaluate(
        &mut self,
        telemetry: &EngineTelemetry,
        user_target_tier: QualityTier,
        auto_quantize: bool,
        profile: GovernorOptimizationProfile,
        stress: HardwareStressProfile,
        mode: MetaControllerInterceptionMode,
        user_thinking_override: Option<usize>,
        dt: f32,
    ) -> GovernorAction {
        self.stress_timer += dt;
        self.buffer_cooldown_timer += dt;
        self.quant_cooldown_timer += dt;

        let target_buffer_ms = profile.target_buffer_ms();
        let max_experts = profile.max_experts();
        let sample_rate = 48000.0f32;
        let env_max_buffer_bytes = Self::compute_env_max_bytes(profile, false);
        let target_capacity_frames = ((target_buffer_ms * 2.0 / 1000.0) * sample_rate).ceil() as usize;
        let dynamic_buffer_bytes = target_capacity_frames * 2 * std::mem::size_of::<f32>();

        // Offline Max Quality Mode Decoupling: latency budget = inf, buffer = 100%, max fidelity
        if mode == MetaControllerInterceptionMode::OfflineMaxQuality {
            let steps = user_thinking_override
                .or(self.meta_controller_steps)
                .unwrap_or(5)
                .clamp(1, 5);
            return GovernorAction {
                recommended_tier: QualityTier::StudioFp32,
                recommended_path: self.current_path,
                min_bits: 16.0,
                max_bits: 32.0,
                synthesis_blend: 0.0,
                status_label: "Offline Master Quality (Unbounded Headroom, 100% Neural)",
                recommended_format: PrecisionFormat::Fp32,
                recommended_moe_mode: MoeExecutionMode::SparseDynamic,
                active_experts: 8,
                diffusion_bypass: false,
                ambisonic_order_reduced: false,
                target_buffer_ms: 1000.0,
                simulated_buffer_health_ms: 1000.0,
                simulated_cpu_headroom: 1.0,
                simulated_panic_factor: 0.0,
                simulated_jitter_factor: 0.0,
                thinking_steps: steps,
                use_consistency_jump: false,
                dynamic_buffer_bytes,
                buffer_health_ratio: 1.0,
                target_capacity_frames,
                env_max_buffer_bytes,
                resize_commanded: false,
                buffer_resize_cooldown: self.buffer_cooldown_timer,
                quant_macro_cooldown: self.quant_cooldown_timer,
                buffer_resizes_count: self.total_buffer_resizes,
                quant_swaps_count: self.total_quant_swaps,
            };
        }

        // 1. Simulate hardware stress profiles (0 to 7) matching RainAI deployment specification
        let mut sim_buffer = telemetry.buffer_health_ms;
        let mut sim_cpu = telemetry.cpu_headroom;
        let mut sim_panic = telemetry.panic_factor;
        let mut sim_jitter = telemetry.jitter_factor;

        match stress {
            HardwareStressProfile::NominalDesktop => {
                // Baseline pristine environment
            }
            HardwareStressProfile::ThermalThrottlingCascade => {
                let throttle = (self.stress_timer * 0.10).min(1.0);
                sim_cpu = (sim_cpu * (1.0 - 0.82 * throttle)).max(0.06);
                sim_panic = (sim_panic + 0.78 * throttle).min(1.0);
                sim_buffer = (sim_buffer * (1.0 - 0.65 * throttle)).max(8.0);
                sim_jitter = (sim_jitter + 0.12 * throttle).min(0.5);
            }
            HardwareStressProfile::GcWebAudioMicroStalls => {
                let cycle = self.stress_timer % 3.0;
                if cycle < 0.45 {
                    sim_buffer = (sim_buffer - 22.0).max(5.0);
                    sim_panic = (sim_panic + 0.65).min(1.0);
                    sim_jitter = 0.48;
                }
            }
            HardwareStressProfile::UnifiedMemoryBusContention => {
                sim_jitter = 0.42;
                sim_cpu = (sim_cpu * 0.45).max(0.12);
                sim_panic = (sim_panic + 0.40).min(1.0);
            }
            HardwareStressProfile::DynamicGameDawInterference => {
                let burst = (self.stress_timer % 2.5) < 0.55;
                if burst {
                    sim_cpu = 0.08;
                    sim_panic = (sim_panic + 0.68).min(1.0);
                    sim_buffer = (sim_buffer - 18.0).max(6.0);
                }
            }
            HardwareStressProfile::BluetoothA2dpAudioSink => {
                sim_jitter = 0.28;
                sim_buffer = (sim_buffer * 0.75).max(12.0);
            }
            HardwareStressProfile::EcoSleepSoundscapeMode => {
                sim_cpu = 0.22;
                sim_panic = (sim_panic + 0.15).min(1.0);
            }
            HardwareStressProfile::HeterogeneousEcoreAsymmetry => {
                let is_ecore = ((self.stress_timer / 1.5) as usize) % 2 == 1;
                if is_ecore {
                    sim_cpu = 0.16;
                    sim_jitter = 0.35;
                    sim_panic = (sim_panic + 0.45).min(1.0);
                } else {
                    sim_cpu = 0.85;
                    sim_jitter = 0.03;
                }
            }
        }

        // Relative Buffer Health: Health ratio normalized against dynamic target size in ms
        let buffer_health_ratio = (sim_buffer / target_buffer_ms.max(1.0)).clamp(0.0, 2.0);

        if !auto_quantize {
            let bits = match user_target_tier {
                QualityTier::Ternary158 => (1.58, 2.0),
                QualityTier::AdaptiveMinimum => (1.58, 8.0),
                QualityTier::HighInt16 => (16.0, 16.0),
                QualityTier::StudioFp32 => (32.0, 32.0),
            };
            let recommended_format = PrecisionFormat::from_continuous_bit_width(bits.1);
            let steps = user_thinking_override
                .or(self.meta_controller_steps)
                .unwrap_or(telemetry.thinking_steps)
                .clamp(1, 5);
            return GovernorAction {
                recommended_tier: user_target_tier,
                recommended_path: self.current_path,
                min_bits: bits.0,
                max_bits: bits.1,
                synthesis_blend: telemetry.synthesis_blend,
                status_label: if telemetry.is_prebuffered {
                    "Pre-Buffered & Ready (Happy)"
                } else {
                    "Manual Override (Governor Inactive)"
                },
                recommended_format,
                recommended_moe_mode: self.current_moe_mode,
                active_experts: 8,
                diffusion_bypass: false,
                ambisonic_order_reduced: false,
                target_buffer_ms,
                simulated_buffer_health_ms: sim_buffer,
                simulated_cpu_headroom: sim_cpu,
                simulated_panic_factor: sim_panic,
                simulated_jitter_factor: sim_jitter,
                thinking_steps: steps,
                use_consistency_jump: steps == 1,
                dynamic_buffer_bytes,
                buffer_health_ratio,
                target_capacity_frames,
                env_max_buffer_bytes,
                resize_commanded: false,
                buffer_resize_cooldown: self.buffer_cooldown_timer,
                quant_macro_cooldown: self.quant_cooldown_timer,
                buffer_resizes_count: self.total_buffer_resizes,
                quant_swaps_count: self.total_quant_swaps,
            };
        }

        // Stress evaluated relative to target size
        let is_under_stress = buffer_health_ratio < 0.60
            || sim_cpu < 0.22
            || sim_panic > 0.32;

        let is_ample_headroom = buffer_health_ratio >= 0.85
            && sim_cpu > 0.65
            && sim_panic < 0.08;

        // Dynamic Multi-Objective Buffer Sizing & Environmental Limits
        let env_max_buffer_bytes = Self::compute_env_max_bytes(profile, is_under_stress);
        let bytes_per_frame = 2 * std::mem::size_of::<f32>();
        let max_frames_from_env = env_max_buffer_bytes / bytes_per_frame;
        let min_frames_safety = ((15.0 / 1000.0) * sample_rate) as usize;
        let ideal_headroom = ((target_buffer_ms / 1000.0) * sample_rate) as usize;
        let ideal_capacity = ideal_headroom * 2;
        let safety_mult = if sim_jitter > 0.20 { 1.25 } else { 1.0 };
        let target_capacity_frames = ((ideal_capacity as f32 * safety_mult) as usize)
            .clamp(min_frames_safety * 2, max_frames_from_env);
        let dynamic_buffer_bytes = target_capacity_frames * bytes_per_frame;

        let frame_diff = (target_capacity_frames as isize - self.current_capacity_frames as isize).unsigned_abs();
        let emergency_resize = buffer_health_ratio < 0.25;
        let mut resize_commanded = false;
        if (emergency_resize || self.buffer_cooldown_timer >= 2.0) && frame_diff >= 128 {
            self.current_capacity_frames = target_capacity_frames;
            self.buffer_cooldown_timer = 0.0;
            self.total_buffer_resizes += 1;
            resize_commanded = true;
        }

        // Meta-Controller Action 1: MoE Expert Shedding (scale 2..=max_experts based on panic factor)
        if sim_panic > 0.25 {
            let shed_amount = ((sim_panic - 0.25) * 8.0).round() as usize;
            self.active_experts = max_experts.saturating_sub(shed_amount).max(2);
        } else {
            self.active_experts = max_experts;
        }

        // Meta-Controller Action 2: Latent Diffusion Bypass
        self.diffusion_bypass = self.meta_controller_diffusion_bypass.unwrap_or(false)
            || sim_panic > 0.70
            || buffer_health_ratio < 0.35;

        // Meta-Controller Action 3: Ambisonic Order Scaling
        self.ambisonic_order_reduced = sim_panic > 0.85;

        // Handle profile-driven defaults
        match profile {
            GovernorOptimizationProfile::EcoBatterySaver => {
                self.active_experts = self.active_experts.min(2);
                self.synthesis_blend = self.synthesis_blend.max(0.50);
                self.max_bits = self.max_bits.min(4.0);
                self.min_bits = 1.58;
                self.thinking_steps = 1;
                self.use_consistency_jump = true;
                self.current_moe_mode = MoeExecutionMode::DenseSoupStatic;
            }
            GovernorOptimizationProfile::StudioMaster => {
                self.active_experts = 8;
                self.synthesis_blend = 0.0;
                self.min_bits = 16.0;
                self.max_bits = 32.0;
                self.diffusion_bypass = false;
                self.ambisonic_order_reduced = false;
                self.thinking_steps = 5;
                self.use_consistency_jump = false;
                self.current_moe_mode = MoeExecutionMode::SparseDynamic;
            }
            GovernorOptimizationProfile::LowLatencyInteractive => {
                self.thinking_steps = 2;
                self.use_consistency_jump = true;
            }
            GovernorOptimizationProfile::BalancedAdaptive
            | GovernorOptimizationProfile::BluetoothA2DPSink => {
                self.thinking_steps = 3;
                self.use_consistency_jump = false;
            }
        }

        // Deliberation / Thinking steps resolution:
        // 1. User manual override takes highest priority
        // 2. Meta-Controller alterable recommendation takes second priority
        // 3. Profile defaults and stress response apply otherwise
        if let Some(steps) = user_thinking_override {
            self.thinking_steps = steps.clamp(1, 5);
            self.use_consistency_jump = self.thinking_steps == 1;
        } else if let Some(mc_steps) = self.meta_controller_steps {
            self.thinking_steps = mc_steps.clamp(1, 5);
            self.use_consistency_jump = self.thinking_steps == 1;
        }

        if is_under_stress {
            self.stability_timer = 0.0;
            if user_thinking_override.is_none() && self.meta_controller_steps.is_none() {
                self.use_consistency_jump = true;
                self.thinking_steps = 1;
            }

            if buffer_health_ratio < 0.35 || sim_panic > 0.60 {
                self.current_moe_mode = MoeExecutionMode::DenseSoupStatic;
            } else {
                self.current_moe_mode = MoeExecutionMode::DenseSoupDynamic;
            }

            let next_tier = if buffer_health_ratio < 0.35 {
                QualityTier::Ternary158
            } else {
                match self.current_tier {
                    QualityTier::StudioFp32 => QualityTier::HighInt16,
                    QualityTier::HighInt16 => QualityTier::AdaptiveMinimum,
                    other => other,
                }
            };

            if buffer_health_ratio < 0.35 {
                self.synthesis_blend = (self.synthesis_blend + dt * 4.0).min(1.0);
                self.current_path = ExecutionPath::CpuProcedural;
                self.min_bits = 1.58;
                self.max_bits = 2.0;
            } else {
                self.current_path = ExecutionPath::CpuNeural;
                self.min_bits = 1.58;
                self.max_bits = 8.0;
                self.synthesis_blend = (self.synthesis_blend + dt * 1.5).min(0.70);
            }

            // Macro quantization cooldown check
            if self.current_tier != next_tier && (self.quant_cooldown_timer >= 10.0 || buffer_health_ratio < 0.25) {
                self.current_tier = next_tier;
                self.quant_cooldown_timer = 0.0;
                self.total_quant_swaps += 1;
            }
        } else if is_ample_headroom {
            self.stability_timer += dt;
            if self.stability_timer > 1.2 {
                self.current_moe_mode = MoeExecutionMode::SparseDynamic;
                self.synthesis_blend = (self.synthesis_blend - dt * 0.8).max(0.0);
                if self.synthesis_blend <= 0.05 {
                    if self.current_tier != user_target_tier && (self.quant_cooldown_timer >= 10.0 || self.stability_timer > 3.0) {
                        self.current_tier = user_target_tier;
                        self.quant_cooldown_timer = 0.0;
                        self.total_quant_swaps += 1;
                    }
                    self.current_path = ExecutionPath::CpuNeural;
                    let (min_b, max_b) = match self.current_tier {
                        QualityTier::Ternary158 => (1.58, 2.0),
                        QualityTier::AdaptiveMinimum => (1.58, 8.0),
                        QualityTier::HighInt16 => (16.0, 16.0),
                        QualityTier::StudioFp32 => (32.0, 32.0),
                    };
                    self.min_bits = min_b;
                    self.max_bits = max_b;
                    if user_thinking_override.is_none() {
                        self.use_consistency_jump = self.current_tier == QualityTier::AdaptiveMinimum
                            && profile == GovernorOptimizationProfile::LowLatencyInteractive;
                    }
                }
            }
        } else {
            self.stability_timer = 0.0;
        }

        self.current_format = PrecisionFormat::from_continuous_bit_width(self.max_bits);

        let status_label = if self.synthesis_blend >= 0.95 {
            "Emergency Procedural Fallback (Zero Audio Drops)"
        } else if is_under_stress {
            "Dynamic Efficiency Squeeze (Under Stress)"
        } else if is_ample_headroom && self.stability_timer > 1.0 {
            "Pristine Headroom (Max Quality)"
        } else {
            "Balanced Real-Time Tracking (Optimal)"
        };

        GovernorAction {
            recommended_tier: self.current_tier,
            recommended_path: self.current_path,
            min_bits: self.min_bits,
            max_bits: self.max_bits,
            synthesis_blend: self.synthesis_blend,
            status_label,
            recommended_format: self.current_format,
            recommended_moe_mode: self.current_moe_mode,
            active_experts: self.active_experts,
            diffusion_bypass: self.diffusion_bypass,
            ambisonic_order_reduced: self.ambisonic_order_reduced,
            target_buffer_ms,
            simulated_buffer_health_ms: sim_buffer,
            simulated_cpu_headroom: sim_cpu,
            simulated_panic_factor: sim_panic,
            simulated_jitter_factor: sim_jitter,
            thinking_steps: self.thinking_steps,
            use_consistency_jump: self.use_consistency_jump,
            dynamic_buffer_bytes,
            buffer_health_ratio,
            target_capacity_frames: self.current_capacity_frames,
            env_max_buffer_bytes,
            resize_commanded,
            buffer_resize_cooldown: self.buffer_cooldown_timer,
            quant_macro_cooldown: self.quant_cooldown_timer,
            buffer_resizes_count: self.total_buffer_resizes,
            quant_swaps_count: self.total_quant_swaps,
        }
    }
}

/// Physical momentum & aerodynamic parameter mediator.
/// Eliminates slider scrubbing zipper artifacts and enforces real-world physical continuity.
#[derive(Clone, Debug)]
pub struct LivePreferenceMediator {
    pub current_intensity: f32,
    pub current_runoff: f32,
    pub current_wind_speed: f32,
    pub current_wind_gustiness: f32,
    pub current_surfaces: [f32; 9],
    pub current_yaw: f32,
}

impl Default for LivePreferenceMediator {
    fn default() -> Self {
        Self {
            current_intensity: 0.5,
            current_runoff: 0.4,
            current_wind_speed: 3.5,
            current_wind_gustiness: 0.2,
            current_surfaces: [0.1, 0.2, 0.1, 0.15, 0.05, 0.15, 0.05, 0.1, 0.1],
            current_yaw: 0.0,
        }
    }
}

impl LivePreferenceMediator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Slews target state towards physically continuous effective state.
    pub fn mediate_step(&mut self, target: &RainState, dt: f32) -> RainState {
        let mut state = target.clone();

        // 1. Rain Intensity: Cloud droplet condensation & precipitation inertia (tau ~ 0.35s)
        let alpha_rain = (1.0 - (-dt / 0.35).exp()).clamp(0.0, 1.0);
        self.current_intensity += (target.weather.intensity - self.current_intensity) * alpha_rain;
        state.weather.intensity = self.current_intensity;

        // 2. Runoff & Water Accumulation Dynamics: Ulbrich DSD / sheet flow inertia (tau ~ 0.40s)
        let alpha_runoff = (1.0 - (-dt / 0.40).exp()).clamp(0.0, 1.0);
        self.current_runoff += (target.weather.runoff - self.current_runoff) * alpha_runoff;
        state.weather.runoff = self.current_runoff;

        // 3. Wind Momentum: Aerodynamic fluid mass inertia (tau ~ 0.50s)
        let alpha_wind = (1.0 - (-dt / 0.50).exp()).clamp(0.0, 1.0);
        self.current_wind_speed += (target.wind.speed - self.current_wind_speed) * alpha_wind;
        self.current_wind_gustiness += (target.wind.gustiness - self.current_wind_gustiness) * alpha_wind;
        state.wind.speed = self.current_wind_speed;
        state.wind.gustiness = self.current_wind_gustiness;

        // 4. Asymmetric Surface Wetting (tau_wet ~ 0.2s) vs Drainage (tau_dry ~ 1.5s)
        let target_surfaces = target.surfaces.normalized();
        for i in 0..9 {
            let diff = target_surfaces[i] - self.current_surfaces[i];
            let tau = if diff > 0.0 { 0.20 } else { 1.50 };
            let alpha_s = (1.0 - (-dt / tau).exp()).clamp(0.0, 1.0);
            self.current_surfaces[i] += diff * alpha_s;
        }
        state.surfaces.tin = self.current_surfaces[0];
        state.surfaces.leaves_broad = self.current_surfaces[1];
        state.surfaces.pine_needles = self.current_surfaces[2];
        state.surfaces.pavement = self.current_surfaces[3];
        state.surfaces.water_deep = self.current_surfaces[4];
        state.surfaces.puddle_shallow = self.current_surfaces[5];
        state.surfaces.canvas_tent = self.current_surfaces[6];
        state.surfaces.glass_window = self.current_surfaces[7];
        state.surfaces.wood_deck = self.current_surfaces[8];

        state
    }
}