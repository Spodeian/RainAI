//! Adaptive environmental ambient noise masking engine.
//!
//! Analyzes incoming background room noise level and spectral distribution,
//! dynamically adapting rainfall acoustic density, droplet velocity, and gain
//! to actively mask distracting room noise.

use serde::{Deserialize, Serialize};

/// Ambient noise spectral measurement across standard acoustic bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSpectrum {
    /// Overall RMS energy level in dBFS [-96.0, 0.0]
    pub rms_db: f32,
    /// Low frequency energy (sub-bass & rumble, 20 - 250 Hz)
    pub low_energy: f32,
    /// Mid frequency energy (speech & HVAC, 250 - 4000 Hz)
    pub mid_energy: f32,
    /// High frequency energy (clicks & hiss, 4000 - 20000 Hz)
    pub high_energy: f32,
}

impl Default for NoiseSpectrum {
    fn default() -> Self {
        Self {
            rms_db: -60.0,
            low_energy: 0.01,
            mid_energy: 0.01,
            high_energy: 0.005,
        }
    }
}

/// Adaptive acoustic mask recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaskingRecommendation {
    /// Recommended rain density multiplier (1.0 = normal, up to 2.5 in loud rooms)
    pub rain_density_scale: f32,
    /// Recommended droplet impact velocity multiplier
    pub droplet_velocity_scale: f32,
    /// Recommended master audio output gain boost in dB
    pub gain_boost_db: f32,
    /// Low-cut filter cutoff adjustment in Hz (to avoid muddy build-up)
    pub low_cut_hz: f32,
}

/// Dynamic noise masking controller.
#[derive(Debug, Clone)]
pub struct AmbientNoiseMasker {
    pub target_snr_db: f32,
    pub adaptation_rate: f32,
    pub current_recommendation: MaskingRecommendation,
}

impl AmbientNoiseMasker {
    pub fn new(target_snr_db: f32) -> Self {
        Self {
            target_snr_db,
            adaptation_rate: 0.1,
            current_recommendation: MaskingRecommendation {
                rain_density_scale: 1.0,
                droplet_velocity_scale: 1.0,
                gain_boost_db: 0.0,
                low_cut_hz: 80.0,
            },
        }
    }

    /// Analyzes an incoming mono buffer of microphone / ambient room samples.
    pub fn analyze_buffer(samples: &[f32]) -> NoiseSpectrum {
        if samples.is_empty() {
            return NoiseSpectrum::default();
        }

        let mut sum_sq = 0.0;
        let mut diff_sq = 0.0;

        for (i, &s) in samples.iter().enumerate() {
            sum_sq += s * s;
            if i > 0 {
                let diff = s - samples[i - 1];
                diff_sq += diff * diff;
            }
        }

        let rms = (sum_sq / samples.len() as f32).sqrt().max(1e-6);
        let rms_db = (20.0 * rms.log10()).clamp(-96.0, 0.0);

        // High frequency proxy from first-order differences
        let hf_ratio = (diff_sq / sum_sq.max(1e-6)).min(1.0);

        NoiseSpectrum {
            rms_db,
            low_energy: rms * (1.0 - hf_ratio),
            mid_energy: rms * 0.5,
            high_energy: rms * hf_ratio,
        }
    }

    /// Computes updated masking parameters based on ambient room spectrum.
    pub fn update(&mut self, spectrum: &NoiseSpectrum) -> MaskingRecommendation {
        // If room is louder than -45 dBFS, scale up rainfall density
        let excess_noise = (spectrum.rms_db - (-50.0)).max(0.0);
        let target_density = 1.0 + (excess_noise / 20.0).min(1.5);
        let target_velocity = 1.0 + (spectrum.mid_energy * 2.0).min(0.8);
        let target_boost = (excess_noise * 0.5).min(9.0);
        let target_low_cut = if spectrum.low_energy > 0.05 {
            120.0
        } else {
            80.0
        };

        // Smooth adaptation
        let alpha = self.adaptation_rate;
        self.current_recommendation.rain_density_scale +=
            alpha * (target_density - self.current_recommendation.rain_density_scale);
        self.current_recommendation.droplet_velocity_scale +=
            alpha * (target_velocity - self.current_recommendation.droplet_velocity_scale);
        self.current_recommendation.gain_boost_db +=
            alpha * (target_boost - self.current_recommendation.gain_boost_db);
        self.current_recommendation.low_cut_hz = target_low_cut;

        self.current_recommendation
    }
}
