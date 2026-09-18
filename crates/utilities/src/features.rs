//! Acoustic Feature Extraction for Rain Audio Analysis.
//!
//! Computes spectral centroid, spectral flatness, high-frequency energy ratio,
//! droplet arrival density, and surface classification tags.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader};
use rustfft::num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;
pub const TARGET_SAMPLE_RATE: f32 = 48000.0;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AudioMetadata {
    pub path: String,
    pub filename: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
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

#[derive(Clone, Debug)]
pub struct HighPassClickFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl HighPassClickFilter {
    pub fn new(cutoff: f32, fs: f32) -> Self {
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

    #[inline(always)]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Extracts acoustic and physical metadata from a WAV audio file.
pub fn extract_features(
    path: &Path,
    fft: &std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: &[f32],
    buffer: &mut [Complex<f32>],
) -> Result<AudioMetadata> {
    let mut reader = WavReader::open(path).context("Failed to open WAV")?;
    let spec = reader.spec();

    let raw_samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
        SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok().map(|v| v as f32 * scale))
                .collect()
        }
    };

    let channels = spec.channels as usize;
    let n_frames = raw_samples.len() / channels;
    let duration_secs = n_frames as f32 / spec.sample_rate as f32;

    let mut mono = Vec::with_capacity(n_frames);
    let inv_channels = 1.0 / channels as f32;
    for frame in raw_samples.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() * inv_channels);
    }

    let mut sum_sq = 0.0f32;
    for &s in &mono {
        sum_sq += s * s;
    }
    let rms = (sum_sq / n_frames.max(1) as f32).sqrt();

    // Spectral analysis via STFT
    let num_hops = (n_frames.saturating_sub(FFT_SIZE)) / HOP_SIZE;
    let mut total_centroid = 0.0f32;
    let mut total_flatness = 0.0f32;
    let mut total_hf_ratio = 0.0f32;
    let mut valid_hops = 0;

    let bin_freq = TARGET_SAMPLE_RATE / FFT_SIZE as f32;
    let hf_bin_start = (4000.0 / bin_freq) as usize;

    for h in 0..num_hops {
        let offset = h * HOP_SIZE;
        for (i, c) in buffer.iter_mut().enumerate().take(FFT_SIZE) {
            *c = Complex {
                re: mono[offset + i] * window[i],
                im: 0.0,
            };
        }

        fft.process(buffer);

        let half = FFT_SIZE / 2;
        let mut sum_mag = 0.0f32;
        let mut weighted_freq_sum = 0.0f32;
        let mut log_mag_sum = 0.0f32;
        let mut hf_energy = 0.0f32;

        for (k, c) in buffer.iter().enumerate().take(half) {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            let freq = k as f32 * bin_freq;

            sum_mag += mag;
            weighted_freq_sum += freq * mag;
            log_mag_sum += (mag + 1e-12).ln();

            if k >= hf_bin_start {
                hf_energy += mag * mag;
            }
        }

        if sum_mag > 1e-6 {
            let centroid = weighted_freq_sum / sum_mag;
            let geometric_mean = (log_mag_sum / half as f32).exp();
            let arithmetic_mean = sum_mag / half as f32;
            let flatness = (geometric_mean / (arithmetic_mean + 1e-12)).clamp(0.0, 1.0);
            let total_energy = sum_mag * sum_mag;
            let hf_ratio = (hf_energy / (total_energy + 1e-12)).clamp(0.0, 1.0);

            total_centroid += centroid;
            total_flatness += flatness;
            total_hf_ratio += hf_ratio;
            valid_hops += 1;
        }
    }

    let avg_centroid = if valid_hops > 0 {
        total_centroid / valid_hops as f32
    } else {
        1200.0
    };
    let avg_flatness = if valid_hops > 0 {
        total_flatness / valid_hops as f32
    } else {
        0.3
    };
    let avg_hf_ratio = if valid_hops > 0 {
        total_hf_ratio / valid_hops as f32
    } else {
        0.2
    };

    // Droplet click detector via highpass filter
    let mut click_filter = HighPassClickFilter::new(3500.0, TARGET_SAMPLE_RATE);
    let mut click_count = 0usize;
    let click_thresh = rms * 2.5;

    for &s in &mono {
        let hp = click_filter.process_sample(s);
        if hp.abs() > click_thresh {
            click_count += 1;
        }
    }

    let drops_per_second = (click_count as f32 / duration_secs.max(0.1)).min(5000.0);
    let rain_rate = (rms * 85.0).clamp(0.1, 100.0);
    let droplet_density = (drops_per_second / 2500.0).clamp(0.01, 1.0);

    let surface_tag = if avg_centroid > 3500.0 && avg_hf_ratio > 0.45 {
        "tin_roof"
    } else if avg_centroid > 2400.0 {
        "glass_window"
    } else if avg_flatness > 0.4 {
        "canvas_tent"
    } else if drops_per_second > 1500.0 {
        "pavement"
    } else if avg_centroid < 900.0 {
        "deep_water"
    } else {
        "foliage"
    };

    Ok(AudioMetadata {
        path: path.to_string_lossy().to_string(),
        filename: path.file_name().unwrap().to_string_lossy().to_string(),
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        duration_secs,
        rms_energy: rms,
        rain_rate,
        droplet_density,
        drops_per_second,
        high_freq_ratio: avg_hf_ratio,
        spectral_centroid: avg_centroid,
        spectral_rolloff: avg_centroid * 1.35,
        spectral_flatness: avg_flatness,
        surface_tag: surface_tag.to_string(),
    })
}

use rayon::prelude::*;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

/// In-process feature extraction and manifest generator.
pub fn run_features_pipeline(
    processed_dir: &Path,
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let emit_log = |msg: String| {
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    emit_log(format!("[*] Extracting acoustic features from {:?} in-process...", processed_dir));
    if !processed_dir.exists() {
        fs::create_dir_all(processed_dir)?;
    }

    let wav_files: Vec<PathBuf> = fs::read_dir(processed_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    emit_log(format!("[*] Discovered {} audio chunks to index.", wav_files.len()));

    if stop_signal.load(Ordering::Relaxed) {
        emit_log("[!] Feature extraction aborted by stop token.".to_string());
        return Ok(0);
    }

    let results: Vec<(String, AudioMetadata)> = wav_files
        .par_iter()
        .map_init(
            || {
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(FFT_SIZE);
                let mut window = vec![0.0f32; FFT_SIZE];
                for i in 0..FFT_SIZE {
                    window[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos());
                }
                let buffer = vec![Complex::new(0.0, 0.0); FFT_SIZE];
                (fft, window, buffer)
            },
            |(fft, window, buffer), path| {
                if stop_signal.load(Ordering::Relaxed) {
                    return None;
                }
                let stem = path.file_stem()?.to_string_lossy().into_owned();
                match extract_features(path, fft, window, buffer) {
                    Ok(meta) => Some((stem, meta)),
                    Err(_) => None,
                }
            },
        )
        .filter_map(|x| x)
        .collect();

    let count = results.len();
    let mut manifest = HashMap::new();
    for (stem, meta) in results {
        manifest.insert(stem, meta);
    }

    let out_file = processed_dir.join("manifest.json");
    let f = File::create(&out_file)?;
    serde_json::to_writer_pretty(f, &manifest)?;
    emit_log(format!("[+] Indexed {} chunks into {:?}", count, out_file));

    Ok(count)
}
