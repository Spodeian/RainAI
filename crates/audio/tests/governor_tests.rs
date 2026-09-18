use audio::history::AcousticHistoryBuffer;
use audio::meta_governor::{LivePreferenceMediator, MetaGovernor};
use inference::{ExecutionPath, MoeExecutionMode};
use shared::{
    EngineTelemetry, GovernorOptimizationProfile, HardwareStressProfile,
    MetaControllerInterceptionMode, QualityTier, RainState,
};

#[test]
fn test_governor_efficiency_squeeze_under_stress() {
    let mut gov = MetaGovernor::new();
    gov.current_tier = QualityTier::StudioFp32;

    let stressed_telemetry = EngineTelemetry {
        buffer_health_ms: 18.0,
        cpu_headroom: 0.12,
        panic_factor: 0.6,
        ..Default::default()
    };

    let action = gov.evaluate(
        &stressed_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(action.recommended_tier, QualityTier::HighInt16);
    assert!(action.status_label.contains("Efficiency Squeeze"));
}

#[test]
fn test_governor_critical_procedural_fallback() {
    let mut gov = MetaGovernor::new();
    let critical_telemetry = EngineTelemetry {
        buffer_health_ms: 8.0,
        ..Default::default()
    };

    let action = gov.evaluate(
        &critical_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
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
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.5,
    );
    assert_eq!(action1.recommended_tier, QualityTier::AdaptiveMinimum);

    let action2 = gov.evaluate(
        &healthy_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.5,
    );
    assert_eq!(action2.recommended_tier, QualityTier::AdaptiveMinimum);

    let action3 = gov.evaluate(
        &healthy_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.5,
    );
    assert_eq!(action3.recommended_tier, QualityTier::StudioFp32);
    assert!(action3.status_label.contains("Pristine Headroom") || action3.status_label.contains("Max Quality"));
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
        MetaControllerInterceptionMode::MediatedLive,
        None,
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
        MetaControllerInterceptionMode::MediatedLive,
        None,
        10.0,
    );

    assert!(action.simulated_panic_factor > 0.5);
    assert!(action.simulated_cpu_headroom < 0.3);
    assert!(action.active_experts < 8);
}

#[test]
fn test_governor_offline_max_quality_decoupling() {
    let mut gov = MetaGovernor::new();
    let stressed_telemetry = EngineTelemetry {
        buffer_health_ms: 2.0,
        cpu_headroom: 0.01,
        panic_factor: 1.0,
        ..Default::default()
    };

    // Even under catastrophic live stress, OfflineMaxQuality must unlock unbounded headroom and K=5 deliberation
    let action = gov.evaluate(
        &stressed_telemetry,
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::ThermalThrottlingCascade,
        MetaControllerInterceptionMode::OfflineMaxQuality,
        None,
        0.1,
    );

    assert_eq!(action.recommended_tier, QualityTier::StudioFp32);
    assert_eq!(action.synthesis_blend, 0.0); // 100% Neural AI
    assert_eq!(action.active_experts, 8);
    assert_eq!(action.thinking_steps, 5);
    assert_eq!(action.simulated_panic_factor, 0.0);
    assert_eq!(action.buffer_health_ratio, 1.0);
    assert!(!action.diffusion_bypass);
}

#[test]
fn test_governor_user_thinking_steps_override() {
    let mut gov = MetaGovernor::new();
    let telemetry = EngineTelemetry::default();

    // User explicitly asks for 4 thinking steps in offline mode
    let action = gov.evaluate(
        &telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::StudioMaster,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::OfflineMaxQuality,
        Some(4),
        0.1,
    );
    assert_eq!(action.thinking_steps, 4);

    // In live mode with override 2
    let action_live = gov.evaluate(
        &telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        Some(2),
        0.1,
    );
    assert_eq!(action_live.thinking_steps, 2);
}

#[test]
fn test_live_preference_mediator_physical_momentum() {
    let mut mediator = LivePreferenceMediator::new();
    let mut target = RainState::default();
    target.weather.intensity = 1.0; // Sudden jump from 0.5 to 1.0
    target.wind.speed = 15.0; // Sudden gale from 3.5 to 15.0

    // Step 50ms: value should smoothly ramp, not jump discontinuously
    let stepped = mediator.mediate_step(&target, 0.05);
    assert!(stepped.weather.intensity > 0.5);
    assert!(stepped.weather.intensity < 0.7); // Limited by cloud condensation lag
    assert!(stepped.wind.speed > 3.5);
    assert!(stepped.wind.speed < 6.0); // Limited by fluid mass inertia
}

#[test]
fn test_acoustic_history_buffer_and_retro_refine() {
    let sample_rate = 1000.0; // Fast test sample rate
    let mut history = AcousticHistoryBuffer::new(5.0, sample_rate);
    assert_eq!(history.available_seconds(), 0.0);

    let dummy_cond = [0.1f32; shared::rain::CONDITION_DIM];
    let dummy_foa = audio::decoder::FoaFrame::new(0.5, 0.1, -0.1, 0.0);

    // Record 500 frames (0.5s)
    for _ in 0..500 {
        history.record_frame(&dummy_cond, dummy_foa);
    }

    assert_eq!(history.available_frames(), 500);
    assert!((history.available_seconds() - 0.5).abs() < 1e-3);

    let (cond_win, foa_win) = history.extract_window(0.2);
    assert_eq!(cond_win.len(), 200);
    assert_eq!(foa_win.len(), 200);
}

#[test]
fn test_buffer_resize_cooldown_and_emergency_bypass() {
    let mut gov = MetaGovernor::new();
    gov.current_capacity_frames = 1024; // Starting from smaller capacity to trigger initial sizing
    let initial_telemetry = EngineTelemetry {
        buffer_health_ms: 45.0,
        cpu_headroom: 0.8,
        ..Default::default()
    };

    // First evaluation initializes buffer capacity
    let action1 = gov.evaluate(
        &initial_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert!(action1.resize_commanded);
    assert_eq!(action1.buffer_resizes_count, 1);

    // Minor telemetry variation after 0.5s (< 2.0s cooldown): should NOT resize
    let action2 = gov.evaluate(
        &initial_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.5,
    );
    assert!(!action2.resize_commanded);
    assert_eq!(action2.buffer_resizes_count, 1);

    // Severe buffer health drop (ratio < 0.25): emergency bypass triggers immediate resize even during cooldown!
    let emergency_telemetry = EngineTelemetry {
        buffer_health_ms: 5.0, // target is 45ms, so ratio is 5/45 = 0.11 < 0.25
        cpu_headroom: 0.1,
        panic_factor: 0.9,
        ..Default::default()
    };
    let action3 = gov.evaluate(
        &emergency_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert!(action3.resize_commanded, "Emergency health drop must bypass cooldown");
    assert_eq!(action3.buffer_resizes_count, 2);
}

#[test]
fn test_quantization_macro_cooldown() {
    let mut gov = MetaGovernor::new();
    gov.current_tier = QualityTier::StudioFp32;

    let stressed_telemetry = EngineTelemetry {
        buffer_health_ms: 20.0, // ratio 20/45 = 0.44 (under stress, but > 0.35)
        cpu_headroom: 0.15,
        panic_factor: 0.5,
        ..Default::default()
    };

    // First stress evaluation shifts tier StudioFp32 -> HighInt16 and starts macro cooldown
    let action1 = gov.evaluate(
        &stressed_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(action1.recommended_tier, QualityTier::HighInt16);
    assert_eq!(action1.quant_swaps_count, 1);

    // Next step after 1.0s: cooldown timer is 1.0s < 10.0s, so tier change is blocked
    let action2 = gov.evaluate(
        &stressed_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        1.0,
    );
    assert_eq!(action2.recommended_tier, QualityTier::HighInt16);
    assert_eq!(action2.quant_swaps_count, 1);

    // Advance time past 10s cooldown: now tier downshift to AdaptiveMinimum is allowed
    let action3 = gov.evaluate(
        &stressed_telemetry,
        QualityTier::StudioFp32,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        9.5,
    );
    assert_eq!(action3.recommended_tier, QualityTier::AdaptiveMinimum);
    assert_eq!(action3.quant_swaps_count, 2);
}

#[test]
fn test_environmental_byte_ceilings() {
    let eco_nominal = MetaGovernor::compute_env_max_bytes(GovernorOptimizationProfile::EcoBatterySaver, false);
    let low_lat = MetaGovernor::compute_env_max_bytes(GovernorOptimizationProfile::LowLatencyInteractive, false);
    assert_eq!(eco_nominal, 24 * 1024);
    assert_eq!(low_lat, 32 * 1024);

    let studio_nominal = MetaGovernor::compute_env_max_bytes(GovernorOptimizationProfile::StudioMaster, false);
    let studio_stressed = MetaGovernor::compute_env_max_bytes(GovernorOptimizationProfile::StudioMaster, true);
    assert_eq!(studio_nominal, 128 * 1024);
    assert_eq!(studio_stressed, 64 * 1024);
}

#[test]
fn test_governor_recommends_dense_soup_under_battery_and_stress() {
    let mut gov = MetaGovernor::new();

    // 1. Battery Saver profile recommends DenseSoupStatic
    let eco_action = gov.evaluate(
        &EngineTelemetry::default(),
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::EcoBatterySaver,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(eco_action.recommended_moe_mode, MoeExecutionMode::DenseSoupStatic);

    // 2. Moderate stress recommends DenseSoupDynamic
    let moderate_stress_telemetry = EngineTelemetry {
        buffer_health_ms: 22.0, // target is 45ms, ratio = 22/45 = ~0.48 (< 0.60, > 0.35)
        cpu_headroom: 0.18,     // < 0.22
        panic_factor: 0.40,     // > 0.32, < 0.60
        ..Default::default()
    };
    let stress_action = gov.evaluate(
        &moderate_stress_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(stress_action.recommended_moe_mode, MoeExecutionMode::DenseSoupDynamic);

    // 3. Severe stress / emergency recommends DenseSoupStatic
    let severe_stress_telemetry = EngineTelemetry {
        buffer_health_ms: 10.0, // ratio = 10/45 = 0.22 (< 0.35)
        cpu_headroom: 0.08,
        panic_factor: 0.85,
        ..Default::default()
    };
    let severe_action = gov.evaluate(
        &severe_stress_telemetry,
        QualityTier::HighInt16,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(severe_action.recommended_moe_mode, MoeExecutionMode::DenseSoupStatic);
}

#[test]
fn test_meta_controller_alterable_pre_generated_steps() {
    let mut gov = MetaGovernor::new();
    let nominal_telemetry = EngineTelemetry {
        buffer_health_ms: 45.0,
        cpu_headroom: 0.80,
        panic_factor: 0.05,
        ..Default::default()
    };

    // 1. Baseline: BalancedAdaptive default thinking steps is 3
    let base_action = gov.evaluate(
        &nominal_telemetry,
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(base_action.thinking_steps, 3);
    assert!(!base_action.use_consistency_jump);

    // 2. Meta-Controller alterable command: recommend 4 pre-generated steps
    gov.set_meta_controller_steps(Some(4));
    let mc_action = gov.evaluate(
        &nominal_telemetry,
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(mc_action.thinking_steps, 4);
    assert!(!mc_action.use_consistency_jump);

    // 3. Meta-Controller dynamic update under high stress: recommend 1 step
    gov.update_from_meta_controller(1, 0.85);
    let stress_action = gov.evaluate(
        &nominal_telemetry,
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        None,
        0.1,
    );
    assert_eq!(stress_action.thinking_steps, 1);
    assert!(stress_action.use_consistency_jump);
    assert!(stress_action.diffusion_bypass);

    // 4. User override takes ultimate precedence even over Meta-Controller
    let override_action = gov.evaluate(
        &nominal_telemetry,
        QualityTier::AdaptiveMinimum,
        true,
        GovernorOptimizationProfile::BalancedAdaptive,
        HardwareStressProfile::NominalDesktop,
        MetaControllerInterceptionMode::MediatedLive,
        Some(5),
        0.1,
    );
    assert_eq!(override_action.thinking_steps, 5);
}


