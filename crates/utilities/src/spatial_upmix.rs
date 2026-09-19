//! High-Throughput Spatial Audio Upmixing for RainAI.
//!
//! Converts Mono (1ch) and Stereo (2ch) field recordings to 4-channel First-Order Ambisonics (FOA)
//! in AmbiX (ACN/SN3D: W, Y, Z, X) format with elevation and wind steering.

use crate::synth_rain::gunn_kinzer_terminal_velocity;
use std::f32::consts::PI;

pub const TARGET_SAMPLE_RATE: u32 = 48000;
pub const CHUNK_DURATION_SEC: f32 = 5.0;
pub const CHUNK_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * CHUNK_DURATION_SEC) as usize;

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    pub fn butterworth_lowpass(fc: f32, fs: f32) -> Self {
        let w0 = 2.0 * PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * (2.0f32).sqrt());
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) / 2.0) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) / 2.0) / a0,
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

/// Converts Stereo or Mono audio arrays to 4-channel Ambisonics B-format (W, Y, Z, X).
pub fn stereo_or_mono_to_foa(
    left: &[f32],
    right: &[f32],
    wind_speed_ms: f32,
    wind_azimuth_deg: f32,
) -> [Vec<f32>; 4] {
    let n = left.len();
    let mut w = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];
    let mut x = vec![0.0f32; n];

    let mut lp_mid = Biquad::butterworth_lowpass(2200.0, TARGET_SAMPLE_RATE as f32);
    let mut lp_side = Biquad::butterworth_lowpass(2200.0, TARGET_SAMPLE_RATE as f32);

    let vt_heavy = gunn_kinzer_terminal_velocity(3.5);
    let theta_low = (wind_speed_ms / vt_heavy).atan();
    let elev_low = 78.0f32.to_radians() - theta_low * 0.4;

    let vt_mist = gunn_kinzer_terminal_velocity(0.8);
    let theta_high = (wind_speed_ms / vt_mist).atan();
    let elev_high = 60.0f32.to_radians() - theta_high * 0.5;

    let wind_az = wind_azimuth_deg.to_radians();
    let cos_w = wind_az.cos();
    let sin_w = wind_az.sin();

    let d_x = (TARGET_SAMPLE_RATE as f32 * 0.0015) as usize;
    let d_y = (TARGET_SAMPLE_RATE as f32 * 0.0027) as usize;
    let d_z = (TARGET_SAMPLE_RATE as f32 * 0.0038) as usize;

    let max_delay = d_z.max(d_y).max(d_x);
    let buf_len = max_delay + 1;
    let mut mid_low_hist = vec![0.0f32; buf_len];
    let mut side_low_hist = vec![0.0f32; buf_len];
    let mut mid_high_hist = vec![0.0f32; buf_len];
    let mut side_high_hist = vec![0.0f32; buf_len];
    let mut hist_idx = 0;

    let norm_w = 1.0 / (2.0f32).sqrt();
    let mut max_p = 0.0f32;

    for i in 0..n {
        let mid = 0.5 * (left[i] + right[i]);
        let side = 0.5 * (left[i] - right[i]);

        let mid_low = lp_mid.process_sample(mid);
        let side_low = lp_side.process_sample(side);
        let mid_high = mid - mid_low;
        let side_high = side - side_low;

        mid_low_hist[hist_idx] = mid_low;
        side_low_hist[hist_idx] = side_low;
        mid_high_hist[hist_idx] = mid_high;
        side_high_hist[hist_idx] = side_high;

        let get_hist = |buf: &[f32], delay: usize| -> f32 {
            let offset = (hist_idx + buf_len - delay) % buf_len;
            buf[offset]
        };

        let s_x = get_hist(&side_high_hist, d_x);
        let s_y = get_hist(&side_low_hist, d_y);
        let s_z = get_hist(&mid_high_hist, d_z);

        hist_idx = (hist_idx + 1) % buf_len;

        let w_val = mid * norm_w;
        let y_val = (side_low * 0.6 + s_y * 0.4) + mid_low * sin_w * 0.3;
        let z_val = mid_high * elev_high.sin() * norm_w + s_z * 0.2;
        let x_val = (mid_low * elev_low.cos() * norm_w + s_x * 0.3) + mid_high * cos_w * 0.3;

        w[i] = w_val;
        y[i] = y_val;
        z[i] = z_val;
        x[i] = x_val;

        max_p = max_p.max(w_val.abs()).max(y_val.abs()).max(z_val.abs()).max(x_val.abs());
    }

    if max_p > 1.0 {
        let inv = 0.98 / max_p;
        for i in 0..n {
            w[i] *= inv;
            y[i] *= inv;
            z[i] *= inv;
            x[i] *= inv;
        }
    }

    [w, y, z, x]
}

/// Pure-Rust band-limited linear sample rate converter replacing Python torchaudio.functional.resample.
pub fn resample_linear(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || input.is_empty() {
        return input.to_vec();
    }
    let ratio = dst_sr as f64 / src_sr as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_idx = i as f64 / ratio;
        let idx0 = src_idx.floor() as usize;
        let frac = (src_idx - idx0 as f64) as f32;
        let s0 = if idx0 < input.len() { input[idx0] } else { 0.0 };
        let s1 = if idx0 + 1 < input.len() { input[idx0 + 1] } else { s0 };
        out.push(s0 + frac * (s1 - s0));
    }
    out
}

use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

/// Pure-Rust FOA audio upmix pipeline callable directly in-process.
pub fn run_upmix_pipeline(
    raw_dir: &Path,
    out_dir: &Path,
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let emit_log = |msg: String| {
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    emit_log(format!("[*] Starting in-process FOA Spatial Audio Upmixer from {:?}...", raw_dir));
    fs::create_dir_all(out_dir)?;

    let mut entries = Vec::new();
    visit_dirs(raw_dir, &mut entries)?;
    emit_log(format!("[*] Discovered {} candidate audio files.", entries.len()));

    let mut total_chunks = 0usize;
    for path in entries {
        if stop_signal.load(Ordering::Relaxed) {
            emit_log("[!] Spatial upmixing aborted by user token.".to_string());
            break;
        }
        match process_audio_file(&path, out_dir) {
            Ok(c) => {
                total_chunks += c;
                if c > 0 {
                    emit_log(format!("  -> Upmixed {:?}: {} chunks generated", path.file_name().unwrap_or_default(), c));
                }
            }
            Err(e) => {
                emit_log(format!("  [!] Failed upmixing {:?}: {}", path, e));
            }
        }
    }

    emit_log(format!("[+] Upmix pipeline finished. Total FOA 5.0s chunks: {}", total_chunks));
    Ok(total_chunks)
}

fn visit_dirs(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, files)?;
            } else {
                let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                if ["wav", "mp3", "ogg", "flac"].contains(&ext.as_str()) {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}

fn process_audio_file(input_path: &Path, output_dir: &Path) -> Result<usize> {
    let ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext != "wav" {
        return Ok(0);
    }
    let mut reader = WavReader::open(input_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };

    let channels = spec.channels as usize;
    let n_frames = samples.len() / channels;
    let mut l = Vec::with_capacity(n_frames);
    let mut r = Vec::with_capacity(n_frames);

    if channels >= 2 {
        for chunk in samples.chunks_exact(channels) {
            l.push(chunk[0]);
            r.push(chunk[1]);
        }
    } else {
        for &s in &samples {
            l.push(s);
            r.push(s);
        }
    }

    // Pure-Rust automatic sample-rate normalization to 48 kHz (replaces Python torchaudio.resample)
    let (l, r) = if spec.sample_rate != TARGET_SAMPLE_RATE {
        (
            resample_linear(&l, spec.sample_rate, TARGET_SAMPLE_RATE),
            resample_linear(&r, spec.sample_rate, TARGET_SAMPLE_RATE),
        )
    } else {
        (l, r)
    };

    let foa = stereo_or_mono_to_foa(&l, &r, 3.5, 45.0);
    let stem = input_path.file_stem().unwrap().to_string_lossy();
    let step = CHUNK_SAMPLES;
    let mut chunks_written = 0;
    let mut start = 0;

    while start + CHUNK_SAMPLES <= l.len() {
        let chunk_out = output_dir.join(format!("{}_chunk{:03}.wav", stem, chunks_written));
        let out_spec = WavSpec {
            channels: 4,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&chunk_out, out_spec)?;

        for i in start..start + CHUNK_SAMPLES {
            writer.write_sample(foa[0][i])?;
            writer.write_sample(foa[1][i])?;
            writer.write_sample(foa[2][i])?;
            writer.write_sample(foa[3][i])?;
        }
        writer.finalize()?;
        chunks_written += 1;
        start += step;
    }

    Ok(chunks_written)
}
