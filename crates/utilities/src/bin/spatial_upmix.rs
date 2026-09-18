use anyhow::{bail, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rayon::prelude::*;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tracing::{error, info};
use utilities::spatial_upmix::{stereo_or_mono_to_foa, CHUNK_SAMPLES, TARGET_SAMPLE_RATE};


/// Invokes `ffmpeg` to decode arbitrary audio files directly to a f32 memory buffer, completely bypassing disk I/O.
fn decode_to_memory(input: &Path) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args(["-i"])
        .arg(input)
        .args(["-ar", &TARGET_SAMPLE_RATE.to_string(), "-ac", "2", "-f", "f32le", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
        
    let mut raw_bytes = Vec::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_end(&mut raw_bytes)?;
    }
    
    let status = child.wait()?;
    if !status.success() {
        bail!("FFmpeg stream decoding failed.");
    }
    
    let mut samples = Vec::with_capacity(raw_bytes.len() / 4);
    for chunk in raw_bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    
    Ok(samples)
}

fn process_audio_file(input_path: &Path, output_dir: &Path) -> Result<usize> {
    let ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    
    let (left, right) = if ext == "wav" {
        // Use Hound for fast native WAV parsing
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
            for frame in samples.chunks(channels) {
                l.push(frame[0]);
                r.push(frame[1]);
            }
        } else {
            for &s in &samples {
                l.push(s);
                r.push(s);
            }
        }
        (l, r)
    } else {
        // Use FFmpeg purely in-memory for non-WAV formats
        let samples = decode_to_memory(input_path)?;
        let n_frames = samples.len() / 2;
        let mut l = Vec::with_capacity(n_frames);
        let mut r = Vec::with_capacity(n_frames);
        for chunk in samples.chunks_exact(2) {
            l.push(chunk[0]);
            r.push(chunk[1]);
        }
        (l, r)
    };

    let foa = stereo_or_mono_to_foa(&left, &right, 4.0, 45.0);
    let stem = input_path.file_stem().unwrap().to_string_lossy();
    let step = (CHUNK_SAMPLES as f32 * 0.75) as usize;
    let mut chunks_written = 0;

    let mut start = 0;
    while start + CHUNK_SAMPLES <= left.len() {
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Parallel FOA Spatial Audio Upmixer...");

    let raw_dir = Path::new("Data/rain");
    let out_dir = Path::new("Data/processed");
    fs::create_dir_all(out_dir)?;

    let mut entries = Vec::new();
    if let Err(e) = visit_dirs(raw_dir, &mut entries) {
        error!("Error reading directory tree: {}", e);
    }

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