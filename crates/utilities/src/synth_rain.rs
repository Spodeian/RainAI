//! Physical Rain Sound Synthesizer & Fluid Dynamics Acoustics.
//!
//! Models raindrop acoustics grounded in empirical fluid mechanics:
//! - Gunn-Kinzer terminal droplet velocity
//! - Ulbrich Gamma Drop Size Distribution (DSD)
//! - Van den Doel / Minnaert cavitation bubble entrapment
//! - Multi-material resonant body response (tin roof, canvas tent, glass, wood, water, pavement)

use rand::Rng;
use rand_distr::{Distribution, Gamma};
use shared::surface::CanonicalSurface;
use std::f32::consts::PI;

pub use audio::physical::gunn_kinzer_terminal_velocity;

pub const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Samples droplet diameters (in mm) from the Ulbrich Gamma Drop Size Distribution:
/// $\Lambda = 4.1 \cdot R^{-0.21}$, shape $\mu + 1 = 3.0$, scale $= 1/\Lambda$.
pub fn sample_gamma_dsd(rain_rate_mmh: f32, count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    let mut rng = rand::thread_rng();
    let r = rain_rate_mmh.max(0.05);
    let lambda = 4.1 * r.powf(-0.21);
    let shape = 3.0;
    let scale = (1.0 / lambda).max(0.1);
    let gamma = Gamma::new(shape, scale).unwrap();
    (0..count)
        .map(|_| gamma.sample(&mut rng).clamp(0.2, 5.5))
        .collect()
}

/// Renders a single physical droplet impact directly into output audio buffers.
pub fn render_single_droplet(
    left: &mut [f32],
    right: &mut [f32],
    start_idx: usize,
    diameter_mm: f32,
    surface: &str,
    wind_speed_ms: f32,
    pan: f32,
    sample_rate: u32,
) {
    let vt = gunn_kinzer_terminal_velocity(diameter_mm);
    let vres = (vt * vt + wind_speed_ms * wind_speed_ms).sqrt();
    let kinetic_energy = 0.5 * diameter_mm.powi(3) * vres * vres;

    let radius_m = (diameter_mm * 0.5) * 1e-3;
    let f0 = (3.26 / radius_m.max(1e-4)).clamp(250.0, 14000.0);
    let duration_sec = (0.006 + 0.005 * diameter_mm).clamp(0.006, 0.035);
    let n_samples = (sample_rate as f32 * duration_sec) as usize;
    let actual_samples = n_samples.min(left.len().saturating_sub(start_idx));

    if actual_samples == 0 {
        return;
    }

    let norm = (kinetic_energy / 50.0).clamp(0.05, 1.5);
    let l_gain = norm * pan;
    let r_gain = norm * (1.0 - pan);

    let shock_len = (sample_rate as f32 * 0.00008).max(2.0) as usize;
    let shock_amp = diameter_mm.powf(2.5) * (vres / 9.0);

    let shock_actual = shock_len.min(actual_samples);
    let shock_phase_add = PI / shock_len as f32;
    let mut shock_phase = 0.0f32;

    for i in 0..shock_actual {
        let s = (1.0 - shock_phase.cos()) * 0.5 * shock_amp;
        left[start_idx + i] += s * l_gain;
        right[start_idx + i] += s * r_gain;
        shock_phase += shock_phase_add;
    }

    let inv_sr = 1.0 / sample_rate as f32;

    let canonical = CanonicalSurface::from_tag(surface);
    match canonical {
        CanonicalSurface::WaterDeep | CanonicalSurface::PuddleShallow => {
            // Minnaert cavitation bubble resonance for fluid surfaces
            let damping = (0.13 * f0 + 0.0072 * f0.powf(1.333)).clamp(100.0, 2500.0);
            let damp_mult = (-damping * inv_sr).exp();
            let mut env = 0.7f32;
            let inv_dur = 1.0 / duration_sec;
            let mut t = 0.0f32;
            let mut phase = 0.0f32;

            for i in 0..actual_samples {
                let chirp = f0 * (1.0 + 0.12 * t * inv_dur);
                phase += 2.0 * PI * chirp * inv_sr;

                let s = phase.sin() * env;
                left[start_idx + i] += s * l_gain;
                right[start_idx + i] += s * r_gain;
                t += inv_sr;
                env *= damp_mult;
            }
        }
        _ => {
            // Universal dual-mode plate/membrane acoustic resonator
            let profile = canonical.modal_profile();
            let damp1_mult = (-profile.damp1 * inv_sr).exp();
            let damp2_mult = (-profile.damp2 * inv_sr).exp();
            let mut env1 = profile.amp1 * (diameter_mm / 2.0);
            let mut env2 = profile.amp2 * (diameter_mm / 2.0);
            let phase_add1 = 2.0 * PI * profile.freq1 * inv_sr;
            let phase_add2 = 2.0 * PI * profile.freq2 * inv_sr;
            let mut phase1 = 0.0f32;
            let mut phase2 = 0.0f32;

            for i in 0..actual_samples {
                let s = phase1.sin() * env1 + phase2.sin() * env2;
                left[start_idx + i] += s * l_gain;
                right[start_idx + i] += s * r_gain;
                phase1 += phase_add1;
                phase2 += phase_add2;
                env1 *= damp1_mult;
                env2 *= damp2_mult;
            }
        }
    }
}

/// Generates a complete physical stereo rain audio block.
pub fn generate_rain_texture(
    duration_sec: f32,
    rain_rate_mmh: f32,
    surface: &str,
    sample_rate: u32,
) -> [Vec<f32>; 2] {
    let n_samples = (sample_rate as f32 * duration_sec) as usize;
    let mut left = vec![0.0f32; n_samples];
    let mut right = vec![0.0f32; n_samples];

    let intensity = (rain_rate_mmh / 60.0).clamp(0.05, 1.0);
    let total_droplets = (intensity * 400.0 * duration_sec) as usize;
    let diameters = sample_gamma_dsd(rain_rate_mmh, total_droplets);

    let mut rng = rand::thread_rng();

    for d in diameters {
        let t_start = rng.gen_range(0..n_samples.saturating_sub(1000).max(1));
        let pan: f32 = rng.gen_range(0.2..0.8);
        render_single_droplet(&mut left, &mut right, t_start, d, surface, 4.0, pan, sample_rate);
    }

    let mut max_val = 0.0f32;
    for i in 0..n_samples {
        max_val = max_val.max(left[i].abs()).max(right[i].abs());
    }

    if max_val > 0.0 {
        let factor = 0.90 / max_val;
        for i in 0..n_samples {
            left[i] *= factor;
            right[i] *= factor;
        }
    }
    [left, right]
}

/// Real-time streaming physical raindrop acoustic synthesizer state.
#[derive(Clone, Debug)]
pub struct PhysicalDropletSynthesizer {
    pub sample_rate: f32,
    ring_l: Vec<f32>,
    ring_r: Vec<f32>,
    ring_pos: usize,
    capacity: usize,
}

impl PhysicalDropletSynthesizer {
    pub fn new(sample_rate: f32) -> Self {
        let capacity = (sample_rate * 0.1) as usize; // 100ms internal acoustic tail buffer
        Self {
            sample_rate,
            ring_l: vec![0.0; capacity],
            ring_r: vec![0.0; capacity],
            ring_pos: 0,
            capacity,
        }
    }

    /// Spawns a physical droplet directly into the streaming buffer.
    pub fn trigger_droplet(
        &mut self,
        diameter_mm: f32,
        surface: &str,
        wind_speed_ms: f32,
        pan: f32,
    ) {
        let vt = gunn_kinzer_terminal_velocity(diameter_mm);
        let vres = (vt * vt + wind_speed_ms * wind_speed_ms).sqrt();
        let kinetic_energy = 0.5 * diameter_mm.powi(3) * vres * vres;

        let radius_m = (diameter_mm * 0.5) * 1e-3;
        let f0 = (3.26 / radius_m.max(1e-4)).clamp(250.0, 14000.0);
        let duration_sec = (0.006 + 0.005 * diameter_mm).clamp(0.006, 0.035);
        let n_samples = ((self.sample_rate * duration_sec) as usize).min(self.capacity / 2);

        let norm = (kinetic_energy / 50.0).clamp(0.05, 1.5);
        let l_gain = norm * pan;
        let r_gain = norm * (1.0 - pan);

        let inv_sr = 1.0 / self.sample_rate;
        let (f_res, damp) = match surface {
            "tin" | "roof" => (1250.0f32, 180.0f32),
            "canvas" | "tent" => (400.0f32, 450.0f32),
            "glass" | "window" => (4500.0f32, 800.0f32),
            "wood" | "deck" => (800.0f32, 300.0f32),
            _ => (f0, 250.0f32),
        };

        let damp_mult = (-damp * inv_sr).exp();
        let mut env = 0.5f32;
        let phase_add = 2.0 * PI * f_res * inv_sr;
        let mut phase = 0.0f32;

        for i in 0..n_samples {
            let buf_idx = (self.ring_pos + i) % self.capacity;
            let s = phase.sin() * env;
            self.ring_l[buf_idx] += s * l_gain;
            self.ring_r[buf_idx] += s * r_gain;
            phase += phase_add;
            env *= damp_mult;
        }
    }

    /// Pops next stereo frame `(left, right)`.
    #[inline]
    pub fn next_frame(&mut self) -> (f32, f32) {
        let l = self.ring_l[self.ring_pos];
        let r = self.ring_r[self.ring_pos];

        // Clear after pop for cyclic reuse
        self.ring_l[self.ring_pos] = 0.0;
        self.ring_r[self.ring_pos] = 0.0;
        self.ring_pos = (self.ring_pos + 1) % self.capacity;

        (l, r)
    }
}

use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

/// Pure-Rust Physical Rain Synthesizer pipeline callable in-process.
pub fn run_synth_pipeline(
    out_dir: &Path,
    target_surfaces: Option<&[String]>,
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let emit_log = |msg: String| {
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    emit_log(format!("[*] Starting in-process Physical Rain Synthesizer into {:?}...", out_dir));
    fs::create_dir_all(out_dir)?;

    let default_configs: Vec<(&str, f32, &str)> = vec![
        ("gentle_drizzle", 1.5, "pavement"),
        ("steady_rain", 12.0, "pavement"),
        ("heavy_downpour", 45.0, "pavement"),
        ("urban_pavement", 15.0, "pavement"),
        ("window_rain", 8.0, "glass"),
        ("roof_rain", 25.0, "tin"),
        ("canvas_tent", 12.0, "canvas"),
        ("wood_deck", 18.0, "wood"),
        ("pine_needles", 8.0, "pine"),
        ("forest_foliage", 10.0, "foliage"),
        ("water_deep", 25.0, "water"),
        ("puddle_shallow", 15.0, "puddle"),
        ("compound_urban_balcony", 20.0, "tin"),
        ("compound_forest_camp", 15.0, "canvas"),
        ("compound_porch_storm", 35.0, "wood"),
        ("thunderstorm", 50.0, "pavement"),
    ];

    let configs: Vec<(&str, f32, &str)> = if let Some(targets) = target_surfaces {
        default_configs
            .into_iter()
            .filter(|(_, _, surf)| targets.iter().any(|t| t.contains(surf) || surf.contains(t.as_str())))
            .collect()
    } else {
        default_configs
    };

    let mut generated_count = 0usize;
    for (name, rate, surf) in configs {
        if stop_signal.load(Ordering::Relaxed) {
            emit_log("[!] Physical rain synthesis aborted by user token.".to_string());
            break;
        }

        let file_path = out_dir.join(format!("synth_{}.wav", name));
        emit_log(format!("  -> Synthesizing '{}' (Rate: {:.1} mm/h, Surf: {})", name, rate, surf));
        let stereo = generate_rain_texture(15.0, rate, surf, DEFAULT_SAMPLE_RATE);

        let spec = WavSpec {
            channels: 2,
            sample_rate: DEFAULT_SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        if let Ok(mut writer) = WavWriter::create(&file_path, spec) {
            for i in 0..stereo[0].len() {
                let _ = writer.write_sample(stereo[0][i]);
                let _ = writer.write_sample(stereo[1][i]);
            }
            if writer.finalize().is_ok() {
                generated_count += 1;
            }
        }
    }

    emit_log(format!("[+] Synthetic generation complete! {} chunks created.", generated_count));
    Ok(generated_count)
}
