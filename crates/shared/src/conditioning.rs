//! 554-Dimensional Conditioning Vector Schema and Layout Definitions.
//!
//! Provides canonical slice ranges and zero-heap encoding for the continuous conditioning
//! manifold shared between neural training (Mamba2 MoE, Spatial VAE) and runtime inference.

use std::ops::Range;

/// Total dimensionality of the continuous conditioning vector:
/// 512 (CLAP semantic embedding) + 10 (Base sliders) + 9 (Normalized surfaces)
/// + 4 (Wind dynamics) + 18 (Spatialized side sounds) + 1 (Physics parameter drift) = 554.
pub const CONDITION_DIM: usize = 554;

/// Precise slice layout of the 554-dimensional conditioning vector.
pub struct ConditioningLayout;

impl ConditioningLayout {
    /// 512-dimensional CLAP acoustic semantic embedding projection.
    pub const CLAP: Range<usize> = 0..512;

    /// 10-dimensional macro sliders:
    /// [intensity, wind_speed, wind_azimuth, legacy_surface, runoff, temperature, humidity, pitch_angle, distance, enclosure].
    pub const BASE_CONTROLS: Range<usize> = 512..522;

    /// 9-dimensional canonical physical surface mixture (partition of unity, sum to 1.0).
    /// Ordered by `CanonicalSurface::ALL`:
    /// [asphalt, pavement, tin_roof, canvas_tent, foliage, wood_deck, glass, puddle_shallow, water_deep].
    pub const SURFACES: Range<usize> = 522..531;

    /// 4-dimensional aeroacoustic wind dynamics:
    /// [speed, gustiness, turbulence, howl].
    pub const WIND_DYNAMICS: Range<usize> = 531..535;

    /// 18-dimensional spatialized side acoustic elements:
    /// Insects (3), Birds (3), Fireplace (4), Thunder (4), Traffic (4).
    pub const SIDE_SOUNDS: Range<usize> = 535..553;

    /// 1-dimensional physical parameter drift tolerance.
    pub const DRIFT_INDEX: usize = 553;

    /// Total dimension.
    pub const TOTAL_DIM: usize = CONDITION_DIM;
}

/// Assembles a 554-dimensional conditioning vector without heap allocation.
#[inline]
pub fn encode_conditioning_vector(
    clap: &[f32; 512],
    base_controls: &[f32; 10],
    surfaces: &[f32; 9],
    wind_dynamics: &[f32; 4],
    side_sounds: &[f32; 18],
    drift: f32,
) -> [f32; CONDITION_DIM] {
    let mut u = [0.0f32; CONDITION_DIM];
    u[ConditioningLayout::CLAP].copy_from_slice(clap);
    u[ConditioningLayout::BASE_CONTROLS].copy_from_slice(base_controls);
    u[ConditioningLayout::SURFACES].copy_from_slice(surfaces);
    u[ConditioningLayout::WIND_DYNAMICS].copy_from_slice(wind_dynamics);
    u[ConditioningLayout::SIDE_SOUNDS].copy_from_slice(side_sounds);
    u[ConditioningLayout::DRIFT_INDEX] = drift;
    u
}
