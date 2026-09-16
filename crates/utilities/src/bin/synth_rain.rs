use anyhow::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use tracing::info;

const SAMPLE_RATE: u32 = 48000;

fn gunn_kinzer_terminal_velocity(d_mm: f32) -> f32 {
    let d = d_mm.clamp(0.1, 5.5);
    (9.65 - 10.3 * (-0.6 * d).exp()).max(0.8)
}

fn sample_gamma_dsd(rain_rate_mmh: f32, count: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let r = rain_rate_mmh.max(0.05);
    let lambda = 4.1 * r.powf(-0.21);
    let shape = 3.0; // mu + 1.0 (mu=2.0)
    let scale = (1.0 / lambda).max(0.1);
    let gamma = Gamma::new(shape, scale).unwrap();
    (0..count)
        .map(|_| gamma.sample(&mut rng).clamp(0.2, 5.5))
        .collect()
}

fn generate_single_droplet(diameter_mm: f32, surface: &str, wind_speed_ms: f32) -> Vec<f32> {
    let vt = gunn_kinzer_terminal_velocity(diameter_mm);
    let vres = (vt * vt + wind_speed_ms * wind_speed_ms).sqrt();
    let kinetic_energy = 0.5 * diameter_mm.powi(3) * vres * vres;

    let radius_m = (diameter_mm * 0.5) * 1e-3;
    let f0 = (3.26 / radius_m.max(1e-4)).clamp(250.0, 14000.0);
    let duration_sec = (0.006 + 0.005 * diameter_mm).clamp(0.006, 0.035);
    let n_samples = (SAMPLE_RATE as f32 * duration_sec) as usize;

    let mut drop = vec![0.0f32; n_samples];
    let shock_len = (SAMPLE_RATE as f32 * 0.00008).max(2.0) as usize;
    let shock_amp = diameter_mm.powf(2.5) * (vres / 9.0);

    for i in 0..shock_len.min(n_samples) {
        drop[i] += (1.0 - (PI * i as f32 / shock_len as f32).cos()) * 0.5 * shock_amp;
    }

    if surface == "tin" || surface == "roof" {
        let damp = 180.0;
        for i in 0..n_samples {
            let t = i as f32 / SAMPLE_RATE as f32;
            let m1 = (2.0 * PI * 1250.0 * t).sin() * (-damp * t).exp();
            let m2 = 0.5 * (2.0 * PI * 2550.0 * t).sin() * (-damp * 1.5 * t).exp();
            drop[i] += (m1 + m2) * 0.5 * (diameter_mm / 2.0);
        }
    } else {
        // Water/van den Doel bubble
        let damping = (0.13 * f0 + 0.0072 * f0.powf(1.333)).clamp(100.0, 2500.0);
        for i in 0..n_samples {
            let t = i as f32 / SAMPLE_RATE as f32;
            let chirp = f0 * (1.0 + 0.12 * (t / duration_sec));
            drop[i] += (2.0 * PI * chirp * t).sin() * (-damping * t).exp() * 0.7;
        }
    }

    let norm = (kinetic_energy / 50.0).clamp(0.05, 1.5);
    for s in &mut drop {
        *s *= norm;
    }
    drop
}

fn generate_rain_texture(duration_sec: f32, rain_rate: f32, surface: &str) -> [Vec<f32>; 2] {
    let n_samples = (SAMPLE_RATE as f32 * duration_sec) as usize;
    let mut left = vec![0.0f32; n_samples];
    let mut right = vec![0.0f32; n_samples];

    let intensity = (rain_rate / 60.0).clamp(0.05, 1.0);
    let total_droplets = (intensity * 400.0 * duration_sec) as usize;
    let diameters = sample_gamma_dsd(rain_rate, total_droplets);

    let mut rng = rand::thread_rng();
    for d in diameters {
        let t_start = rng.gen_range(0..n_samples.saturating_sub(1000));
        let drop = generate_single_droplet(d, surface, 4.0);
        let pan: f32 = rng.gen_range(0.2..0.8);

        for (j, &s) in drop.iter().enumerate() {
            if t_start + j < n_samples {
                left[t_start + j] += s * pan;
                right[t_start + j] += s * (1.0 - pan);
            }
        }
    }

    // Normalization
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Running Native Physical Rain Synthesizer...");

    let out_dir = Path::new("Data/rain/Synthetic");
    fs::create_dir_all(out_dir)?;

    let configs = [
        ("gentle_drizzle", 1.5, "pavement"),
        ("steady_rain", 12.0, "pavement"),
        ("heavy_downpour", 45.0, "pavement"),
        ("tin_roof", 25.0, "tin"),
    ];

    for (name, rate, surf) in configs {
        let file_path = out_dir.join(format!("synth_{}.wav", name));
        info!("Synthesizing {} -> {:?}", name, file_path);
        let stereo = generate_rain_texture(10.0, rate, surf);

        let spec = WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&file_path, spec)?;
        for i in 0..stereo[0].len() {
            writer.write_sample(stereo[0][i])?;
            writer.write_sample(stereo[1][i])?;
        }
        writer.finalize()?;
    }

    info!("Synthetic generation complete!");
    Ok(())
}
