use anyhow::Result;
use hound::{SampleFormat, WavReader};
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tracing::info;

const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;
const SAMPLE_RATE: f32 = 48000.0;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AcousticFeatures {
    pub rms_energy: f32,
    pub rain_rate: f32,
    pub droplet_density: f32,
    pub drops_per_second: f32,
    pub high_freq_ratio: f32,
    pub spectral_centroid: f32,
    pub spectral_rolloff: f32,
    pub spectral_flatness: f32,
    pub surface_tag: String,
}

struct HighPassClickFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl HighPassClickFilter {
    fn new(cutoff: f32, fs: f32) -> Self {
        let w0 = 2.0 * PI * cutoff / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * (2.0f32).sqrt());
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) / 2.0) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: ((1.0 + cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    fn filter(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples.len());
        for &x in samples {
            let y = self.b0 * x + self.s1;
            self.s1 = self.b1 * x - self.a1 * y + self.s2;
            self.s2 = self.b2 * x - self.a2 * y;
            out.push(y);
        }
        out
    }
}

fn extract_features(samples: &[f32], filename: &str) -> AcousticFeatures {
    let n = samples.len();
    let duration_sec = (n as f32 / SAMPLE_RATE).max(0.01);

    // 1. RMS Energy
    let mut sum_sq = 0.0f32;
    for &s in samples {
        sum_sq += s * s;
    }
    let rms = (sum_sq / n as f32).sqrt();
    let rms_db = 20.0 * (rms + 1e-12).log10();
    let measured_rate = ((rms_db - (-52.0)) / (-12.0 - (-52.0))).clamp(0.05, 1.0);

    // 2. STFT Spectral Analysis
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut window = vec![0.0f32; FFT_SIZE];
    for i in 0..FFT_SIZE {
        window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos());
    }

    let mut total_power = 0.0f32;
    let mut hf_power = 0.0f32;
    let mut centroid_num = 0.0f32;
    let mut power_bins = vec![0.0f32; FFT_SIZE / 2 + 1];
    let mut frame_count = 0;

    let mut offset = 0;
    while offset + FFT_SIZE <= n {
        let mut buffer: Vec<Complex<f32>> = (0..FFT_SIZE)
            .map(|i| Complex::new(samples[offset + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buffer);

        for bin in 0..=FFT_SIZE / 2 {
            let p = buffer[bin].norm_sqr();
            power_bins[bin] += p;
            total_power += p;
            let freq = bin as f32 * SAMPLE_RATE / FFT_SIZE as f32;
            if freq >= 2000.0 {
                hf_power += p;
            }
            centroid_num += freq * p;
        }
        frame_count += 1;
        offset += HOP_SIZE;
    }

    let hf_ratio = (hf_power / total_power.max(1e-12)).clamp(0.0, 1.0);
    let spectral_centroid = (centroid_num / total_power.max(1e-12)).clamp(0.0, SAMPLE_RATE / 2.0);

    // Rolloff 85%
    let cutoff = 0.85 * total_power;
    let mut acc = 0.0f32;
    let mut spectral_rolloff = 0.0f32;
    for (bin, &p) in power_bins.iter().enumerate() {
        acc += p * frame_count as f32;
        if acc >= cutoff {
            spectral_rolloff = bin as f32 * SAMPLE_RATE / FFT_SIZE as f32;
            break;
        }
    }

    // Flatness (Wiener entropy)
    let mut log_sum = 0.0f32;
    let mut arith_sum = 0.0f32;
    let num_bins = power_bins.len();
    for &p in &power_bins {
        let val = (p / frame_count.max(1) as f32) + 1e-12;
        log_sum += val.ln();
        arith_sum += val;
    }
    let geom_mean = (log_sum / num_bins as f32).exp();
    let arith_mean = arith_sum / num_bins as f32;
    let spectral_flatness = (geom_mean / arith_mean.max(1e-12)).clamp(0.0, 1.0);

    // 3. Transient clicks
    let mut hp = HighPassClickFilter::new(3000.0, SAMPLE_RATE);
    let filtered = hp.filter(samples);
    let mut mean_click = 0.0f32;
    for &c in &filtered {
        mean_click += c.abs();
    }
    let thresh = 3.5 * (mean_click / n as f32) + 1e-6;

    let mut peaks = 0;
    let min_dist = (SAMPLE_RATE * 0.005) as usize;
    let mut last_peak = 0;
    for (i, &c) in filtered.iter().enumerate() {
        if c.abs() > thresh && (i.saturating_sub(last_peak) > min_dist) {
            peaks += 1;
            last_peak = i;
        }
    }

    let drops_per_sec = peaks as f32 / duration_sec;
    let droplet_density = (drops_per_sec / 80.0).clamp(0.02, 1.0);

    let fn_lower = filename.to_lowercase();
    let surface_tag = if fn_lower.contains("glass") || fn_lower.contains("window") {
        "glass".to_string()
    } else if fn_lower.contains("roof") || fn_lower.contains("tin") {
        "tin".to_string()
    } else if fn_lower.contains("canvas") || fn_lower.contains("tent") {
        "canvas".to_string()
    } else if fn_lower.contains("foliage") || fn_lower.contains("leaf") {
        "foliage".to_string()
    } else {
        "pavement".to_string()
    };

    AcousticFeatures {
        rms_energy: rms,
        rain_rate: measured_rate,
        droplet_density,
        drops_per_second: drops_per_sec,
        high_freq_ratio: hf_ratio,
        spectral_centroid,
        spectral_rolloff,
        spectral_flatness,
        surface_tag,
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Running Native Rust Acoustic Feature Extractor...");

    let processed_dir = Path::new("Data/processed");
    let wav_files: Vec<PathBuf> = fs::read_dir(processed_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    info!("Extracting acoustic features for {} audio chunks...", wav_files.len());

    let results: Vec<(String, AcousticFeatures)> = wav_files
        .par_iter()
        .filter_map(|path| {
            let stem = path.file_stem()?.to_string_lossy().into_owned();
            let mut reader = WavReader::open(path).ok()?;
            let spec = reader.spec();
            let samples: Vec<f32> = match spec.sample_format {
                SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
                SampleFormat::Int => {
                    let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
                    reader.samples::<i32>().filter_map(|s| s.ok().map(|v| v as f32 * scale)).collect()
                }
            };
            // Downmix to mono
            let n_frames = samples.len() / spec.channels as usize;
            let mut mono = Vec::with_capacity(n_frames);
            for frame in samples.chunks(spec.channels as usize) {
                mono.push(frame.iter().sum::<f32>() / spec.channels as f32);
            }

            let feats = extract_features(&mono, &stem);
            Some((stem, feats))
        })
        .collect();

    let mut manifest = HashMap::new();
    for (stem, feats) in results {
        manifest.insert(stem, feats);
    }

    let out_file = processed_dir.join("acoustic_manifest.json");
    let f = File::create(&out_file)?;
    serde_json::to_writer_pretty(f, &manifest)?;
    info!("Wrote verified features to {:?}", out_file);

    Ok(())
}
