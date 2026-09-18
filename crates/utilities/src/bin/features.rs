use anyhow::Result;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tracing::{error, info};
use utilities::features::{extract_features, AudioMetadata, FFT_SIZE};


fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting multi-threaded Rust dataset preprocessing & manifest indexing...");

    let processed_dir = Path::new("Data/processed");
    if !processed_dir.exists() {
        std::fs::create_dir_all(processed_dir)?;
    }

    let wav_files: Vec<PathBuf> = fs::read_dir(processed_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| {
            p.extension().and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        })
        .collect();

    info!("Extracting structural and acoustic features for {} audio chunks...", wav_files.len());

    let results: Vec<(String, AudioMetadata)> = wav_files
        .par_iter()
        .map_init(
            || {
                // Initialize expensive structs and allocations exactly once per thread
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
                let stem = path.file_stem()?.to_string_lossy().into_owned();
                match extract_features(path, fft, window, buffer) {
                    Ok(meta) => Some((stem, meta)),
                    Err(e) => {
                        error!("Error processing {:?}: {}", path, e);
                        None
                    }
                }
            }
        )
        .filter_map(|x| x)
        .collect();

    let mut manifest = HashMap::new();
    for (stem, meta) in results {
        manifest.insert(stem, meta);
    }

    let out_file = processed_dir.join("manifest.json");
    let f = File::create(&out_file)?;
    serde_json::to_writer_pretty(f, &manifest)?;
    info!("Wrote verified unified metadata to {:?}", out_file);

    Ok(())
}