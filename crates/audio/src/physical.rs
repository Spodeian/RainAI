//! Physical fluid dynamics & terminal velocity droplet acoustic synthesizer.
//!
//! Implements:
//! - Gunn-Kinzer terminal velocity: v_t(D) = 9.65 * (1 - exp(-0.6 * D))
//! - Ulbrich Gamma Drop Size Distribution (DSD): N(D) = N0 * D^mu * exp(-Lambda * D)
//! - Minnaert bubble acoustics for puddle/water impacts: f0 = 3.26 / r
//! - Surface-dependent acoustic modal resonance for tin, canvas, glass, wood, pavement, leaves
//! - True 4-channel First Order Ambisonics (FOA) spatial particle emission with wind drift

use crate::decoder::FoaFrame;
use crate::procedural::{BrownNoiseFilter, FastRng};
use shared::rain::RainState;
use std::f32::consts::PI;

/// Gunn-Kinzer empirical terminal fall velocity for a raindrop of diameter `d_mm` (m/s).
#[inline]
pub fn gunn_kinzer_terminal_velocity(diameter_mm: f32) -> f32 {
    let d = diameter_mm.clamp(0.1, 7.0);
    9.65 * (1.0 - (-0.6 * d).exp())
}

/// Ulbrich Gamma Drop Size Distribution density function N(D).
#[inline]
pub fn ulbrich_dsd(diameter_mm: f32, rain_rate_mmh: f32) -> f32 {
    let r = rain_rate_mmh.clamp(0.1, 150.0);
    let n0 = 8000.0 * r.powf(0.37);
    let mu = 2.0;
    let lambda = 4.1 * r.powf(-0.21);
    n0 * diameter_mm.powf(mu) * (-lambda * diameter_mm).exp()
}

/// Real-time streaming physical raindrop acoustic synthesizer state.
#[derive(Clone, Debug)]
pub struct PhysicalRainSynthesizer {
    pub sample_rate: f32,
    rng: FastRng,
    brown_filter: BrownNoiseFilter,
    // Overlapping acoustic impulse ring buffers for 4 FOA channels
    ring_w: Vec<f32>,
    ring_x: Vec<f32>,
    ring_y: Vec<f32>,
    ring_z: Vec<f32>,
    ring_pos: usize,
    capacity: usize,
}

impl Default for PhysicalRainSynthesizer {
    fn default() -> Self {
        Self::new(48000.0)
    }
}

impl PhysicalRainSynthesizer {
    pub fn new(sample_rate: f32) -> Self {
        let capacity = (sample_rate * 0.12) as usize; // 120ms impulse tail capacity
        Self {
            sample_rate,
            rng: FastRng::new(99),
            brown_filter: BrownNoiseFilter::default(),
            ring_w: vec![0.0; capacity],
            ring_x: vec![0.0; capacity],
            ring_y: vec![0.0; capacity],
            ring_z: vec![0.0; capacity],
            ring_pos: 0,
            capacity,
        }
    }

    /// Spawns a physical droplet impact into the internal circular FOA tail buffer.
    pub fn trigger_droplet(
        &mut self,
        diameter_mm: f32,
        surface_type: usize, // 0: Tin, 1: Leaves, 2: Pine, 3: Pavement, 4: Water, 5: Canvas, 6: Glass, 7: Wood
        wind_speed_ms: f32,
        wind_angle_rad: f32,
        pitch_angle_rad: f32,
    ) {
        let vt = gunn_kinzer_terminal_velocity(diameter_mm);
        let vres = (vt * vt + wind_speed_ms * wind_speed_ms).sqrt();
        let kinetic_energy = 0.5 * diameter_mm.powi(3) * vres * vres;

        let radius_m = (diameter_mm * 0.5) * 1e-3;
        let minnaert_f0 = (3.26 / radius_m.max(1e-4)).clamp(250.0, 14000.0);
        let duration_sec = (0.006 + 0.005 * diameter_mm).clamp(0.006, 0.040);
        let n_samples = ((self.sample_rate * duration_sec) as usize).min(self.capacity / 2);

        let norm = (kinetic_energy / 45.0).clamp(0.04, 1.6);

        // Compute 3D arrival angle for droplet (influenced by wind and pitch angle)
        let azim = self.rng.next_f32() * PI + (wind_angle_rad * 0.3);
        let elev = pitch_angle_rad + (self.rng.next_f32() * 0.2);
        let (cos_e, sin_e) = (elev.cos(), elev.sin());
        let (sin_a, cos_a) = azim.sin_cos();

        let w_gain = norm * 0.7071;
        let x_gain = norm * cos_e * cos_a;
        let y_gain = norm * cos_e * sin_a;
        let z_gain = norm * sin_e;

        let inv_sr = 1.0 / self.sample_rate;

        match surface_type {
            0 => {
                // Tin roof: Dual metallic resonance (1250 Hz & 2550 Hz)
                let damp1_mult = (-180.0 * inv_sr).exp();
                let damp2_mult = (-270.0 * inv_sr).exp();
                let mut env1 = 0.5 * (diameter_mm / 2.0);
                let mut env2 = 0.25 * (diameter_mm / 2.0);
                let phase_add1 = 2.0 * PI * 1250.0 * inv_sr;
                let phase_add2 = 2.0 * PI * 2550.0 * inv_sr;
                let mut phase1 = 0.0f32;
                let mut phase2 = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase1.sin() * env1 + phase2.sin() * env2;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase1 += phase_add1;
                    phase2 += phase_add2;
                    env1 *= damp1_mult;
                    env2 *= damp2_mult;
                }
            }
            1 => {
                // Broad leaves: Soft damp slap (1400 Hz)
                let damp_mult = (-420.0 * inv_sr).exp();
                let mut env = 0.45f32;
                let phase_add = 2.0 * PI * 1400.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            2 => {
                // Pine needles: Fast micro-clicks (3200 Hz)
                let damp_mult = (-700.0 * inv_sr).exp();
                let mut env = 0.35f32;
                let phase_add = 2.0 * PI * 3200.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            3 => {
                // Pavement: Crisp splatter splash
                let damp_mult = (-600.0 * inv_sr).exp();
                let mut env = 0.40f32;
                let phase_add = 2.0 * PI * 1100.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            5 => {
                // Canvas tent: Low damped thud (400 Hz)
                let damp_mult = (-450.0 * inv_sr).exp();
                let mut env = 0.5f32;
                let phase_add = 2.0 * PI * 400.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            6 => {
                // Glass window: Bright sharp transient (4500 Hz)
                let damp_mult = (-800.0 * inv_sr).exp();
                let mut env = 0.6f32;
                let phase_add = 2.0 * PI * 4500.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            7 => {
                // Wood deck: Warm knock (800 Hz)
                let damp_mult = (-300.0 * inv_sr).exp();
                let mut env = 0.45f32;
                let phase_add = 2.0 * PI * 800.0 * inv_sr;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    phase += phase_add;
                    env *= damp_mult;
                }
            }
            _ => {
                // Water / Puddle: Minnaert bubble chirping with upward glide
                let damping = (0.13 * minnaert_f0 + 0.0072 * minnaert_f0.powf(1.333)).clamp(100.0, 2500.0);
                let damp_mult = (-damping * inv_sr).exp();
                let mut env = 0.7f32;
                let inv_dur = 1.0 / duration_sec;
                let mut t = 0.0f32;
                let mut phase = 0.0f32;

                for i in 0..n_samples {
                    let idx = (self.ring_pos + i) % self.capacity;
                    let chirp = minnaert_f0 * (1.0 + 0.12 * t * inv_dur);
                    phase += 2.0 * PI * chirp * inv_sr;
                    let s = phase.sin() * env;
                    self.ring_w[idx] += s * w_gain;
                    self.ring_x[idx] += s * x_gain;
                    self.ring_y[idx] += s * y_gain;
                    self.ring_z[idx] += s * z_gain;
                    t += inv_sr;
                    env *= damp_mult;
                }
            }
        }
    }

    /// Process one frame of physical rain soundscape in 4-channel FOA format.
    pub fn process_frame(&mut self, state: &RainState) -> FoaFrame {
        if !state.is_playing {
            return FoaFrame::default();
        }

        // 1. Poisson droplet arrival rate derived from rain intensity and Ulbrich DSD
        let intensity = state.weather.intensity.clamp(0.01, 1.0);
        // Probability of a new droplet event spawning on this sample
        let spawn_prob = (intensity * 0.06).clamp(0.002, 0.35);

        if self.rng.next_unit_f32() < spawn_prob {
            // Sample droplet diameter (Marshall-Palmer / Ulbrich Gamma distribution proxy)
            let u = self.rng.next_unit_f32().max(1e-4);
            let diameter_mm = (-u.ln() * (0.8 + intensity * 1.5)).clamp(0.4, 6.0);

            // Select dominant active surface
            let surfaces = [
                (0, state.surfaces.tin),
                (1, state.surfaces.leaves_broad),
                (2, state.surfaces.pine_needles),
                (3, state.surfaces.pavement),
                (4, state.surfaces.water_deep + state.surfaces.puddle_shallow),
                (5, state.surfaces.canvas_tent),
                (6, state.surfaces.glass_window),
                (7, state.surfaces.wood_deck),
            ];

            let mut total_surf = 0.0f32;
            for (_, weight) in &surfaces {
                total_surf += *weight;
            }

            let mut chosen_surface = 4; // default to water/puddle
            if total_surf > 0.001 {
                let mut pick = self.rng.next_unit_f32() * total_surf;
                for (id, weight) in &surfaces {
                    if pick <= *weight {
                        chosen_surface = *id;
                        break;
                    }
                    pick -= *weight;
                }
            }

            let wind_angle_rad = state.wind.turbulence * PI;
            let pitch_angle_rad = state.weather.pitch_angle;

            self.trigger_droplet(
                diameter_mm,
                chosen_surface,
                state.wind.speed * 15.0,
                wind_angle_rad,
                pitch_angle_rad,
            );
        }

        // 2. Continuous wind bed and turbulence
        let white = self.rng.next_f32();
        let wind_drive = self.brown_filter.process(white) * state.wind.speed;
        let wind_master = wind_drive * (1.0 + state.wind.gustiness * 0.5) * state.master_volume * 0.4;

        // 3. Pop droplet tail frame
        let pw = self.ring_w[self.ring_pos];
        let px = self.ring_x[self.ring_pos];
        let py = self.ring_y[self.ring_pos];
        let pz = self.ring_z[self.ring_pos];

        self.ring_w[self.ring_pos] = 0.0;
        self.ring_x[self.ring_pos] = 0.0;
        self.ring_y[self.ring_pos] = 0.0;
        self.ring_z[self.ring_pos] = 0.0;
        self.ring_pos = (self.ring_pos + 1) % self.capacity;

        // 4. Combine droplet acoustic impacts with spatial wind vector
        let w = (pw * 0.85 + wind_master * 0.7071) * state.master_volume;
        let x = (px * 0.85 + wind_master * 0.4) * state.master_volume;
        let y = py * 0.85 * state.master_volume;
        let z = pz * 0.85 * state.master_volume;

        FoaFrame::new(w, x, y, z)
    }
}
