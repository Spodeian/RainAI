//! Personalized Head-Related Transfer Function (HRTF) and SOFA spatialization.
//!
//! Provides coordinate mapping (azimuth, elevation, distance), spherical interpolation,
//! and binaural time-domain convolution with measured Head-Related Impulse Responses (HRIRs).

use serde::{Deserialize, Serialize};

/// 3D spherical sound source position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphericalPosition {
    /// Azimuth angle in degrees: [-180.0, 180.0] (0 = straight ahead, 90 = right, -90 = left)
    pub azimuth_deg: f32,
    /// Elevation angle in degrees: [-90.0, 90.0] (0 = horizontal plane, 90 = directly above)
    pub elevation_deg: f32,
    /// Distance from head center in meters
    pub distance_m: f32,
}

impl SphericalPosition {
    pub fn new(azimuth_deg: f32, elevation_deg: f32, distance_m: f32) -> Self {
        Self {
            azimuth_deg: azimuth_deg.clamp(-180.0, 180.0),
            elevation_deg: elevation_deg.clamp(-90.0, 90.0),
            distance_m: distance_m.max(0.1),
        }
    }
}

/// Pair of Head-Related Impulse Responses (HRIR) for left and right ears.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrirPair {
    pub position: SphericalPosition,
    pub left_ir: Vec<f32>,
    pub right_ir: Vec<f32>,
}

/// SOFA (Spatially Oriented Format for Acoustics) Binaural Spatializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SofaSpatializer {
    pub impulse_responses: Vec<HrirPair>,
    pub sample_rate: u32,
}

impl SofaSpatializer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            impulse_responses: Vec::new(),
            sample_rate,
        }
    }

    /// Adds a measured HRIR pair for a spherical coordinate.
    pub fn add_hrir(&mut self, position: SphericalPosition, left_ir: Vec<f32>, right_ir: Vec<f32>) {
        self.impulse_responses.push(HrirPair {
            position,
            left_ir,
            right_ir,
        });
    }

    /// Finds the nearest measured HRIR pair by Euclidean angular distance.
    pub fn find_nearest_hrir(&self, pos: SphericalPosition) -> Option<&HrirPair> {
        if self.impulse_responses.is_empty() {
            return None;
        }

        self.impulse_responses.iter().min_by(|a, b| {
            let dist_a = (a.position.azimuth_deg - pos.azimuth_deg).powi(2)
                + (a.position.elevation_deg - pos.elevation_deg).powi(2);
            let dist_b = (b.position.azimuth_deg - pos.azimuth_deg).powi(2)
                + (b.position.elevation_deg - pos.elevation_deg).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Performs binaural convolution of a mono audio buffer into a stereo (left, right) buffer.
    pub fn spatialize_mono(&self, input: &[f32], pos: SphericalPosition) -> (Vec<f32>, Vec<f32>) {
        let n = input.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        if let Some(hrir) = self.find_nearest_hrir(pos) {
            let ir_len = hrir.left_ir.len().min(hrir.right_ir.len());
            let out_len = n + ir_len.saturating_sub(1);
            let mut left_out = vec![0.0f32; out_len];
            let mut right_out = vec![0.0f32; out_len];

            // Direct time-domain convolution
            for i in 0..n {
                let s = input[i];
                for j in 0..ir_len {
                    left_out[i + j] += s * hrir.left_ir[j];
                    right_out[i + j] += s * hrir.right_ir[j];
                }
            }

            // Attenuate by 1 / distance (inverse square law for sound pressure)
            let atten = 1.0 / pos.distance_m;
            for sample in &mut left_out {
                *sample *= atten;
            }
            for sample in &mut right_out {
                *sample *= atten;
            }

            (left_out, right_out)
        } else {
            // Fallback: simple stereo panning based on azimuth
            let pan = (pos.azimuth_deg / 180.0).clamp(-1.0, 1.0);
            let left_gain = ((1.0 - pan) * 0.5).sqrt();
            let right_gain = ((1.0 + pan) * 0.5).sqrt();

            let left = input.iter().map(|&s| s * left_gain).collect();
            let right = input.iter().map(|&s| s * right_gain).collect();
            (left, right)
        }
    }
}
