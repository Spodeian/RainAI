use audio::MetaGovernor;
use inference::ExecutionPath;
use shared::{EngineTelemetry, GovernorOptimizationProfile, HardwareStressProfile, QualityTier};

#[test]
fn test_governor_efficiency_squeeze_under_stress() {
    let mut gov = MetaGovernor::new();
    gov.current_tier = QualityTier::StudioFp32;

    let mut stressed_telemetry = EngineTelemetry::default();
    stressed_telemetry.buffer_health_ms = 18.0;
    stressed_telemetry.cpu_headroom = 0.12;
    stressed_telemetry.panic_factor = 0.6;

    let action = gov.evaluate(
        &stressed_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        0.1,
    );
    assert_eq!(action.recommended_tier, QualityTier::HighInt16);
    assert!(action.status_label.contains("Efficiency Squeeze"));
}

#[test]
fn test_governor_critical_procedural_fallback() {
    let mut gov = MetaGovernor::new();
    let mut critical_telemetry = EngineTelemetry::default();
    critical_telemetry.buffer_health_ms = 8.0;

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

    let healthy_telemetry = EngineTelemetry::default();

    let action1 = gov.evaluate(
        &healthy_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        0.5,
    );
    assert_eq!(action1.recommended_tier, QualityTier::AdaptiveMinimum);

    let action2 = gov.evaluate(
        &healthy_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        0.5,
    );
    assert_eq!(action2.recommended_tier, QualityTier::AdaptiveMinimum);

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
