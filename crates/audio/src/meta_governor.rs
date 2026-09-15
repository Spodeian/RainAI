//! Meta-Controller Dynamic Quantization Governor.
//!
//! Continuously evaluates real-time audio buffer health and compute headroom,
//! dynamically adapting quantization floors and ceilings (and recommending tier step-downs)
//! to guarantee zero-glitch audio playback while maximizing sound fidelity.

use inference::{ExecutionPath, PrecisionFormat};
use shared::rain::{
    EngineTelemetry, GovernorOptimizationProfile, HardwareStressProfile, QualityTier,
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
    pub active_experts: usize,
    pub diffusion_bypass: bool,
    pub ambisonic_order_reduced: bool,
    pub target_buffer_ms: f32,
    pub simulated_buffer_health_ms: f32,
    pub simulated_cpu_headroom: f32,
    pub simulated_panic_factor: f32,
    pub simulated_jitter_factor: f32,
}

/// Governor state machine with anti-hunting hysteresis and hardware stress simulation
#[derive(Clone, Debug)]
pub struct MetaGovernor {
    stability_timer: f32,
    stress_timer: f32,
    current_tier: QualityTier,
    current_path: ExecutionPath,
    current_format: PrecisionFormat,
    min_bits: f32,
    max_bits: f32,
    synthesis_blend: f32,
    active_experts: usize,
    diffusion_bypass: bool,
    ambisonic_order_reduced: bool,
}

impl Default for MetaGovernor {
    fn default() -> Self {
        Self {
            stability_timer: 0.0,
            stress_timer: 0.0,
            current_tier: QualityTier::AdaptiveMinimum,
            current_path: ExecutionPath::CpuNeural,
            current_format: PrecisionFormat::Int8,
            min_bits: 1.58,
            max_bits: 8.0,
            synthesis_blend: 0.0,
            active_experts: 8,
            diffusion_bypass: false,
            ambisonic_order_reduced: false,
        }
    }
}

impl MetaGovernor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates current telemetry against optimization profile & simulated hardware stress
    pub fn evaluate(
        &mut self,
        telemetry: &EngineTelemetry,
        user_target_tier: QualityTier,
        auto_quantize: bool,
        profile: GovernorOptimizationProfile,
        stress: HardwareStressProfile,
        dt: f32,
    ) -> GovernorAction {
        self.stress_timer += dt;

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
                // Progressive thermal clock collapse
                let throttle = (self.stress_timer * 0.10).min(1.0);
                sim_cpu = (sim_cpu * (1.0 - 0.82 * throttle)).max(0.06);
                sim_panic = (sim_panic + 0.78 * throttle).min(1.0);
                sim_buffer = (sim_buffer * (1.0 - 0.65 * throttle)).max(8.0);
                sim_jitter = (sim_jitter + 0.12 * throttle).min(0.5);
            }
            HardwareStressProfile::GcWebAudioMicroStalls => {
                // Injects periodic 5-25ms garbage collector pauses every 3 seconds
                let cycle = self.stress_timer % 3.0;
                if cycle < 0.45 {
                    sim_buffer = (sim_buffer - 22.0).max(5.0);
                    sim_panic = (sim_panic + 0.65).min(1.0);
                    sim_jitter = 0.48;
                }
            }
            HardwareStressProfile::UnifiedMemoryBusContention => {
                // Bandwidth bottleneck causing tensor transfer jitter
                sim_jitter = 0.42;
                sim_cpu = (sim_cpu * 0.45).max(0.12);
                sim_panic = (sim_panic + 0.40).min(1.0);
            }
            HardwareStressProfile::DynamicGameDawInterference => {
                // Competing bursty DAW / game host load spikes
                let burst = (self.stress_timer % 2.5) < 0.55;
                if burst {
                    sim_cpu = 0.08;
                    sim_panic = (sim_panic + 0.68).min(1.0);
                    sim_buffer = (sim_buffer - 18.0).max(6.0);
                }
            }
            HardwareStressProfile::BluetoothA2dpAudioSink => {
                // Variable packet dispatch and transmission latency
                sim_jitter = 0.28;
                sim_buffer = (sim_buffer * 0.75).max(12.0);
            }
            HardwareStressProfile::EcoSleepSoundscapeMode => {
                // Ultra-low frequency power throttling
                sim_cpu = 0.22;
                sim_panic = (sim_panic + 0.15).min(1.0);
            }
            HardwareStressProfile::HeterogeneousEcoreAsymmetry => {
                // Thread migration bouncing between P-core and E-core every 1.5s
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

        let target_buffer_ms = profile.target_buffer_ms();
        let max_experts = profile.max_experts();

        if !auto_quantize {
            // Manual override mode
            let bits = match user_target_tier {
                QualityTier::Ternary158 => (1.58, 2.0),
                QualityTier::AdaptiveMinimum => (1.58, 8.0),
                QualityTier::HighInt16 => (16.0, 16.0),
                QualityTier::StudioFp32 => (32.0, 32.0),
            };
            let recommended_format = PrecisionFormat::from_continuous_bit_width(bits.1);
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
                active_experts: 8,
                diffusion_bypass: false,
                ambisonic_order_reduced: false,
                target_buffer_ms,
                simulated_buffer_health_ms: sim_buffer,
                simulated_cpu_headroom: sim_cpu,
                simulated_panic_factor: sim_panic,
                simulated_jitter_factor: sim_jitter,
            };
        }

        // Stress trigger thresholds scaled by profile target buffer
        let stress_buffer_thresh = (target_buffer_ms * 0.55).max(15.0);
        let critical_buffer_thresh = (target_buffer_ms * 0.28).max(9.0);

        let is_under_stress = sim_buffer < stress_buffer_thresh
            || sim_cpu < 0.22
            || sim_panic > 0.32;

        let is_ample_headroom = sim_buffer > (target_buffer_ms * 0.85)
            && sim_cpu > 0.65
            && sim_panic < 0.08;

        // Meta-Controller Action 1: MoE Expert Shedding (scale 2..=max_experts based on panic factor)
        if sim_panic > 0.25 {
            let shed_amount = ((sim_panic - 0.25) * 8.0).round() as usize;
            self.active_experts = max_experts.saturating_sub(shed_amount).max(2);
        } else {
            self.active_experts = max_experts;
        }

        // Meta-Controller Action 2: Latent Diffusion Bypass (panic > 0.70 or buffer critically low)
        self.diffusion_bypass = sim_panic > 0.70 || sim_buffer < critical_buffer_thresh;

        // Meta-Controller Action 3: Ambisonic Order Scaling (extreme panic > 0.85)
        self.ambisonic_order_reduced = sim_panic > 0.85;

        // Handle specific optimization profile constraints
        match profile {
            GovernorOptimizationProfile::EcoBatterySaver => {
                // Strict ≤4b cap, at least 50% procedural blend, max 2 experts
                self.active_experts = self.active_experts.min(2);
                self.synthesis_blend = self.synthesis_blend.max(0.50);
                self.max_bits = self.max_bits.min(4.0);
                self.min_bits = 1.58;
            }
            GovernorOptimizationProfile::StudioMaster => {
                // Pristine ≥16b floor, 8 experts, 0% blend
                self.active_experts = 8;
                self.synthesis_blend = 0.0;
                self.min_bits = 16.0;
                self.max_bits = 32.0;
                self.diffusion_bypass = false;
                self.ambisonic_order_reduced = false;
            }
            GovernorOptimizationProfile::LowLatencyInteractive
            | GovernorOptimizationProfile::BalancedAdaptive
            | GovernorOptimizationProfile::BluetoothA2DPSink => {}
        }

        if is_under_stress {
            // Reset stability timer
            self.stability_timer = 0.0;

            // Critical underrun defense: emergency procedural blend
            if sim_buffer < critical_buffer_thresh {
                self.synthesis_blend = (self.synthesis_blend + dt * 4.0).min(1.0);
                self.current_tier = QualityTier::Ternary158;
                self.min_bits = 1.58;
                self.max_bits = 2.0;
                self.current_path = ExecutionPath::CpuProcedural;
                self.current_format = PrecisionFormat::Ternary158;

                return GovernorAction {
                    recommended_tier: QualityTier::Ternary158,
                    recommended_path: ExecutionPath::CpuProcedural,
                    min_bits: 1.58,
                    max_bits: 2.0,
                    synthesis_blend: self.synthesis_blend,
                    status_label: "Emergency Procedural Blend (Buffer Critical)",
                    recommended_format: PrecisionFormat::Ternary158,
                    active_experts: 2,
                    diffusion_bypass: true,
                    ambisonic_order_reduced: true,
                    target_buffer_ms,
                    simulated_buffer_health_ms: sim_buffer,
                    simulated_cpu_headroom: sim_cpu,
                    simulated_panic_factor: sim_panic,
                    simulated_jitter_factor: sim_jitter,
                };
            }

            // Step-down tier cascade to shed compute
            if profile != GovernorOptimizationProfile::StudioMaster {
                self.current_tier = match self.current_tier {
                    QualityTier::StudioFp32 => QualityTier::HighInt16,
                    QualityTier::HighInt16 => QualityTier::AdaptiveMinimum,
                    QualityTier::AdaptiveMinimum => QualityTier::Ternary158,
                    QualityTier::Ternary158 => QualityTier::Ternary158,
                };

                self.max_bits = (self.max_bits - dt * 6.0).max(1.58);
                self.min_bits = 1.58;
            }

            self.current_format = PrecisionFormat::from_continuous_bit_width(self.max_bits);

            GovernorAction {
                recommended_tier: self.current_tier,
                recommended_path: ExecutionPath::CpuNeural,
                min_bits: self.min_bits,
                max_bits: self.max_bits,
                synthesis_blend: self.synthesis_blend,
                status_label: "Efficiency Squeeze (Throttled for Stability)",
                recommended_format: self.current_format,
                active_experts: self.active_experts,
                diffusion_bypass: self.diffusion_bypass,
                ambisonic_order_reduced: self.ambisonic_order_reduced,
                target_buffer_ms,
                simulated_buffer_health_ms: sim_buffer,
                simulated_cpu_headroom: sim_cpu,
                simulated_panic_factor: sim_panic,
                simulated_jitter_factor: sim_jitter,
            }
        } else if is_ample_headroom {
            // Anti-hunting hysteresis
            let hysteresis_target = match profile {
                GovernorOptimizationProfile::LowLatencyInteractive => 0.4,
                GovernorOptimizationProfile::BalancedAdaptive => 1.2,
                GovernorOptimizationProfile::EcoBatterySaver => 0.8,
                GovernorOptimizationProfile::BluetoothA2DPSink => 1.5,
                GovernorOptimizationProfile::StudioMaster => 1.8,
            };

            self.stability_timer += dt;

            // Slowly bleed down procedural blend towards profile minimum
            let blend_floor = if profile == GovernorOptimizationProfile::EcoBatterySaver {
                0.50
            } else {
                0.0
            };
            self.synthesis_blend = (self.synthesis_blend - dt * 2.0).max(blend_floor);

            if self.stability_timer > hysteresis_target {
                match user_target_tier {
                    QualityTier::StudioFp32 => {
                        self.current_tier = QualityTier::StudioFp32;
                        self.max_bits = 32.0;
                        self.min_bits = 16.0;
                    }
                    QualityTier::HighInt16 => {
                        self.current_tier = QualityTier::HighInt16;
                        self.max_bits = 16.0;
                        self.min_bits = 8.0;
                    }
                    QualityTier::AdaptiveMinimum => {
                        self.current_tier = QualityTier::AdaptiveMinimum;
                        self.max_bits = 8.0;
                        self.min_bits = 1.58;
                    }
                    QualityTier::Ternary158 => {
                        self.current_tier = QualityTier::Ternary158;
                        self.max_bits = 2.0;
                        self.min_bits = 1.58;
                    }
                }

                if profile == GovernorOptimizationProfile::EcoBatterySaver {
                    self.max_bits = self.max_bits.min(4.0);
                }
            }

            self.current_format = PrecisionFormat::from_continuous_bit_width(self.max_bits);

            GovernorAction {
                recommended_tier: self.current_tier,
                recommended_path: self.current_path,
                min_bits: self.min_bits,
                max_bits: self.max_bits,
                synthesis_blend: self.synthesis_blend,
                status_label: if telemetry.is_prebuffered {
                    "Pre-Buffered & Ready (Happy)"
                } else if self.stability_timer > hysteresis_target {
                    "Effectiveness Expansion (Quality Boost Active)"
                } else {
                    "Stabilizing Headroom..."
                },
                recommended_format: self.current_format,
                active_experts: self.active_experts,
                diffusion_bypass: false,
                ambisonic_order_reduced: false,
                target_buffer_ms,
                simulated_buffer_health_ms: sim_buffer,
                simulated_cpu_headroom: sim_cpu,
                simulated_panic_factor: sim_panic,
                simulated_jitter_factor: sim_jitter,
            }
        } else {
            // Nominal stable holding state
            self.stability_timer = 0.0;
            let blend_floor = if profile == GovernorOptimizationProfile::EcoBatterySaver {
                0.50
            } else {
                0.0
            };
            self.synthesis_blend = (self.synthesis_blend - dt * 0.5).max(blend_floor);
            self.current_format = PrecisionFormat::from_continuous_bit_width(self.max_bits);

            GovernorAction {
                recommended_tier: self.current_tier,
                recommended_path: self.current_path,
                min_bits: self.min_bits,
                max_bits: self.max_bits,
                synthesis_blend: self.synthesis_blend,
                status_label: if telemetry.is_prebuffered {
                    "Pre-Buffered & Ready (Happy)"
                } else {
                    "Nominal Execution (Balanced)"
                },
                recommended_format: self.current_format,
                active_experts: self.active_experts,
                diffusion_bypass: self.diffusion_bypass,
                ambisonic_order_reduced: self.ambisonic_order_reduced,
                target_buffer_ms,
                simulated_buffer_health_ms: sim_buffer,
                simulated_cpu_headroom: sim_cpu,
                simulated_panic_factor: sim_panic,
                simulated_jitter_factor: sim_jitter,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governor_efficiency_squeeze_under_stress() {
        let mut gov = MetaGovernor::new();
        gov.current_tier = QualityTier::StudioFp32;

        let mut stressed_telemetry = EngineTelemetry::default();
        stressed_telemetry.buffer_health_ms = 18.0; // low buffer
        stressed_telemetry.cpu_headroom = 0.12;      // high CPU load
        stressed_telemetry.panic_factor = 0.6;

        let action = gov.evaluate(
            &stressed_telemetry,
            QualityTier::StudioFp32,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::NominalDesktop,
            0.1,
        );
        // Should shed tier down from StudioFp32 to HighInt16
        assert_eq!(action.recommended_tier, QualityTier::HighInt16);
        assert!(action.status_label.contains("Efficiency Squeeze"));
    }

    #[test]
    fn test_governor_critical_procedural_fallback() {
        let mut gov = MetaGovernor::new();
        let mut critical_telemetry = EngineTelemetry::default();
        critical_telemetry.buffer_health_ms = 8.0; // severe underrun threat

        let action = gov.evaluate(
            &critical_telemetry,
            QualityTier::HighInt16,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::NominalDesktop,
            0.1,
        );
        assert_eq!(action.recommended_tier, QualityTier::Ternary158);
        assert_eq!(action.recommended_path, ExecutionPath::CpuProcedural);
        assert!(action.synthesis_blend > 0.0);
        assert!(action.diffusion_bypass);
    }

    #[test]
    fn test_governor_anti_hunting_upscale() {
        let mut gov = MetaGovernor::new();
        gov.current_tier = QualityTier::AdaptiveMinimum;

        let healthy_telemetry = EngineTelemetry::default(); // 45ms buffer, 88% CPU headroom

        // Frame 1: Should not instantly upscale (hysteresis delay)
        let action1 = gov.evaluate(
            &healthy_telemetry,
            QualityTier::StudioFp32,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::NominalDesktop,
            0.5,
        );
        assert_eq!(action1.recommended_tier, QualityTier::AdaptiveMinimum);

        // Frame 2: Still < 1.2s total (0.5 + 0.5 = 1.0s)
        let action2 = gov.evaluate(
            &healthy_telemetry,
            QualityTier::StudioFp32,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::NominalDesktop,
            0.5,
        );
        assert_eq!(action2.recommended_tier, QualityTier::AdaptiveMinimum);

        // Frame 3: Passed 1.2s (1.0 + 0.5 = 1.5s) -> upscales to StudioFp32!
        let action3 = gov.evaluate(
            &healthy_telemetry,
            QualityTier::StudioFp32,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::NominalDesktop,
            0.5,
        );
        assert_eq!(action3.recommended_tier, QualityTier::StudioFp32);
        assert!(action3.status_label.contains("Effectiveness Expansion"));
    }

    #[test]
    fn test_eco_battery_saver_profile_constraints() {
        let mut gov = MetaGovernor::new();
        let telemetry = EngineTelemetry::default();

        let action = gov.evaluate(
            &telemetry,
            QualityTier::StudioFp32,
            true,
            GovernorOptimizationProfile::EcoBatterySaver,
            HardwareStressProfile::NominalDesktop,
            0.1,
        );

        assert!(action.max_bits <= 4.0);
        assert!(action.active_experts <= 2);
        assert_eq!(action.target_buffer_ms, 30.0);
    }

    #[test]
    fn test_hardware_stress_thermal_throttling_simulation() {
        let mut gov = MetaGovernor::new();
        let telemetry = EngineTelemetry::default();

        // Evaluate under thermal throttle for 10 simulated seconds
        let action = gov.evaluate(
            &telemetry,
            QualityTier::AdaptiveMinimum,
            true,
            GovernorOptimizationProfile::BalancedAdaptive,
            HardwareStressProfile::ThermalThrottlingCascade,
            10.0,
        );

        assert!(action.simulated_panic_factor > 0.5);
        assert!(action.simulated_cpu_headroom < 0.3);
        assert!(action.active_experts < 8);
    }
}

