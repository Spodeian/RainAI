use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rayon::prelude::*;
use std::f32::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{error, info};

const TARGET_SAMPLE_RATE: u32 = 48000;
const CHUNK_DURATION_SEC: f32 = 5.0;
const CHUNK_SAMPLES: usize = (TARGET_SAMPLE_RATE as f32 * CHUNK_DURATION_SEC) as usize;

/// Gunn-Kinzer (1949) terminal fall velocity.
fn gunn_kinzer_terminal_velocity(d_mm: f32) -> f32 {
    let d = d_mm.clamp(0.1, 5.5);
    (9.65 - 10.3 * (-0.6 * d).exp()).max(0.8)
}

/// 2nd-order IIR Biquad filter (Direct Form II Transposed).
#[derive(Clone)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    fn butterworth_lowpass(fc: f32, fs: f32) -> Self {
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

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            let y = self.b0 * x + self.s1;
            self.s1 = self.b1 * x - self.a1 * y + self.s2;
            self.s2 = self.b2 * x - self.a2 * y;
            out.push(y);
        }
        out
    }
}

fn delay_samples(arr: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; arr.len()];
    if d < arr.len() {
        out[d..].copy_from_slice(&arr[..arr.len() - d]);
    }
    out
}

/// Converts stereo or mono input to 4-channel FOA (W, Y, Z, X)[cite: 23].
fn stereo_or_mono_to_foa(
    left: &[f32],
    right: &[f32],
    wind_speed_ms: f32,
    wind_azimuth_deg: f32,
) -> [Vec<f32>; 4] {
    let n = left.len();
    let mut mid = vec![0.0f32; n];
    let mut side = vec![0.0f32; n];
    for i in 0..n {
        mid[i] = 0.5 * (left[i] + right[i]);
        side[i] = 0.5 * (left[i] - right[i]);
    }

    let mut lp_mid = Biquad::butterworth_lowpass(2200.0, TARGET_SAMPLE_RATE as f32);
    let mut lp_side = Biquad::butterworth_lowpass(2200.0, TARGET_SAMPLE_RATE as f32);

    let mid_low = lp_mid.process(&mid);
    let side_low = lp_side.process(&side);

    let mut mid_high = vec![0.0f32; n];
    let mut side_high = vec![0.0f32; n];
    for i in 0..n {
        mid_high[i] = mid[i] - mid_low[i];
        side_high[i] = side[i] - side_low[i];
    }

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

    let mid_low_dz = delay_samples(&mid_low, d_z);
    let mid_high_dz = delay_samples(&mid_high, d_z);
    let side_low_dy = delay_samples(&side_low, d_y);
    let side_high_dy = delay_samples(&side_high, d_y);
    let mid_low_dx = delay_samples(&mid_low, d_x);
    let side_low_dx = delay_samples(&side_low, d_x);
    let mid_high_dx = delay_samples(&mid_high, d_x);
    let side_high_dx = delay_samples(&side_high, d_x);

    let mut w = vec![0.0f32; n];
    let mut y = vec![0.0f32; n];
    let mut z = vec![0.0f32; n];
    let mut x = vec![0.0f32; n];

    let norm_w = 1.0 / (2.0f32).sqrt();

    for i in 0..n {
        w[i] = norm_w * mid[i];

        let z_l = (0.85 * mid_low[i] + 0.15 * mid_low_dz[i]) * elev_low.sin();
        let z_h = (0.65 * mid_high[i] + 0.35 * mid_high_dz[i]) * elev_high.sin();
        z[i] = z_l + z_h;

        let y_l = (0.7 * side_low[i] + 0.3 * side_low_dy[i]) * elev_low.cos();
        let y_h = ((0.75 * side_high[i] + 0.25 * side_high_dy[i]) + 0.25 * mid_high[i] * sin_w)
            * elev_high.cos();
        y[i] = y_l + y_h;

        let x_l = 0.5 * (mid_low_dx[i] - side_low_dx[i]) * elev_low.cos();
        let x_h = (0.5 * (mid_high_dx[i] - side_high_dx[i]) + 0.3 * mid_high[i] * cos_w)
            * elev_high.cos();
        x[i] = x_l + x_h;
    }

    let mut max_p = 0.0f32;
    for i in 0..n {
        max_p = max_p.max(w[i].abs()).max(y[i].abs()).max(z[i].abs()).max(x[i].abs());
    }
    if max_p > 0.99 {
        let factor = 0.95 / max_p;
        for i in 0..n {
            w[i] *= factor;
            y[i] *= factor;
            z[i] *= factor;
            x[i] *= factor;
        }
    }

    [w, y, z, x]
}

fn process_audio_file(path: &Path, output_dir: &Path) -> Result<usize> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };

    let channels = spec.channels as usize;
    let n_frames = samples.len() / channels;
    let mut left = Vec::with_capacity(n_frames);
    let mut right = Vec::with_capacity(n_frames);

    if channels >= 2 {
        for frame in samples.chunks(channels) {
            left.push(frame[0]);
            right.push(frame[1]);
        }
    } else {
        for &s in &samples {
            left.push(s);
            right.push(s);
        }
    }

    let foa = stereo_or_mono_to_foa(&left, &right, 4.0, 45.0);
    let stem = path.file_stem().unwrap().to_string_lossy();
    let step = (CHUNK_SAMPLES as f32 * 0.75) as usize;
    let mut chunks_written = 0;

    let mut start = 0;
    while start + CHUNK_SAMPLES <= n_frames {
        let chunk_out = output_dir.join(format!("{}_chunk{:03}.wav", stem, chunks_written));
        let out_spec = WavSpec {
            channels: 4,
            sample_rate: TARGET_SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&chunk_out, out_spec)?;

        for i in start..start + CHUNK_SAMPLES {
            writer.write_sample(foa[0][i])?; // W
            writer.write_sample(foa[1][i])?; // Y
            writer.write_sample(foa[2][i])?; // Z
            writer.write_sample(foa[3][i])?; // X
        }
        writer.finalize()?;
        chunks_written += 1;
        start += step;
    }

    Ok(chunks_written)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Parallel FOA Spatial Audio Upmixer...");

    let raw_dir = Path::new("Data/rain");
    let out_dir = Path::new("Data/processed");
    fs::create_dir_all(out_dir)?;

    let entries: Vec<PathBuf> = fs::read_dir(raw_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    info!("Discovered {} audio candidates in {:?}", entries.len(), raw_dir);

    let total_chunks: usize = entries
        .par_iter()
        .map(|path| match process_audio_file(path, out_dir) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed processing {:?}: {}", path, e);
                0
            }
        })
        .sum();

    info!("Finished upmixing! Total 5.0s FOA chunks created: {}", total_chunks);
    Ok(())
}
