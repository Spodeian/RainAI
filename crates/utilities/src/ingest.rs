//! Dataset Ingestion, Ethical License Verification, Acoustic Quality Screening, and Async Download Engine.
//!
//! Provides:
//! - Multi-source async downloads with exponential backoff retries and HF auth.
//! - Ethical licensing verification (`LicenseVerifier`) enforcing CC0, CC-BY, and Public Domain.
//! - Acoustic quality metrics calculation (`AcousticQualityMetrics`) measuring RMS, clipping ratio,
//!   Wiener spectral flatness, and Shannon spectral entropy.
//! - 9-class physical surface classification & quota tracking (`SurfaceBalanceQuota`).
//! - SHA-256 cryptographic provenance records (`ProvenanceRecord`, `ProvenanceManifest`).

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use hound::{SampleFormat, WavReader};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File as StdFile;
use std::io::Read;
use std::path::Path;
use std::time::Duration;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadItem {
    pub url: String,
    pub filename: String,
    pub category: String,
    pub license: String,
    pub source_platform: String,
    #[serde(default = "default_media_type")]
    pub media_type: String,
    #[serde(default = "default_ingest_method")]
    pub ingest_method: String,
}

fn default_media_type() -> String {
    "audio".to_string()
}
fn default_ingest_method() -> String {
    "direct_http".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseTier {
    PublicDomain,
    AttributionOnly,
    ShareAlike,
    Restricted,
}

pub struct LicenseVerifier;

impl LicenseVerifier {
    pub fn verify(license: &str) -> (bool, LicenseTier, &'static str) {
        let clean = license.trim().to_lowercase();

        if clean.contains("nc") || clean.contains("noncommercial") || clean.contains("nd") {
            return (
                false,
                LicenseTier::Restricted,
                "Rejected: NonCommercial or NoDerivatives clause detected",
            );
        }
        if clean.contains("cc-by-sa") {
            return (
                true,
                LicenseTier::ShareAlike,
                "Approved: CC-BY-SA (Commercial compatible with share-alike)",
            );
        }
        if clean.contains("cc-by")
            || clean.contains("attribution")
            || clean.contains("mixkit free license")
        {
            return (
                true,
                LicenseTier::AttributionOnly,
                "Approved: CC-BY / Commercial free with attribution",
            );
        }
        if clean.contains("cc0")
            || clean.contains("public domain")
            || clean.contains("open access")
            || clean.contains("nps natural sound")
        {
            return (
                true,
                LicenseTier::PublicDomain,
                "Approved: Public domain / CC0 / Government unconstrained",
            );
        }
        (
            false,
            LicenseTier::Restricted,
            "Rejected: Unverified or restrictive license format",
        )
    }
}

pub use shared::surface::CanonicalSurface;

/// Tracks stratified distribution and diversity quotas across the 9 canonical surfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceBalanceQuota {
    pub counts: HashMap<CanonicalSurface, usize>,
    pub target_per_surface: usize,
}

impl SurfaceBalanceQuota {
    pub fn new(target_per_surface: usize) -> Self {
        let mut counts = HashMap::new();
        for s in CanonicalSurface::ALL {
            counts.insert(s, 0);
        }
        Self {
            counts,
            target_per_surface,
        }
    }

    pub fn record(&mut self, surface: CanonicalSurface) {
        *self.counts.entry(surface).or_insert(0) += 1;
    }

    pub fn total_samples(&self) -> usize {
        self.counts.values().sum()
    }

    /// Computes Shannon entropy: H = -sum(p_i * ln(p_i)).
    /// Maximum entropy across 9 categories is ln(9) ≈ 2.1972.
    pub fn shannon_entropy(&self) -> f32 {
        let total = self.total_samples() as f32;
        if total == 0.0 {
            return 0.0;
        }
        let mut h = 0.0f32;
        for &count in self.counts.values() {
            if count > 0 {
                let p = count as f32 / total;
                h -= p * p.ln();
            }
        }
        h
    }

    /// Returns normalized diversity index in range [0.0, 1.0].
    pub fn normalized_diversity(&self) -> f32 {
        let max_h = (9.0f32).ln();
        (self.shannon_entropy() / max_h).clamp(0.0, 1.0)
    }

    /// Returns surfaces currently below the targeted quota.
    pub fn underrepresented_surfaces(&self) -> Vec<(CanonicalSurface, usize)> {
        self.counts
            .iter()
            .filter(|&(_, &c)| c < self.target_per_surface)
            .map(|(&s, &c)| (s, c))
            .collect()
    }

}

/// Acoustic quality metrics for validation and filtering of downloaded precipitation audio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcousticQualityMetrics {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
    pub rms_energy: f32,
    pub peak_amplitude: f32,
    pub clipping_ratio: f32,
    pub spectral_entropy: f32,
    pub spectral_flatness: f32,
    pub high_freq_ratio: f32,
    pub is_valid_rain_texture: bool,
}

/// Computes acoustic quality metrics from mono/downmixed PCM floating point audio samples.
pub fn analyze_pcm_samples(samples: &[f32], sample_rate: u32, channels: u16) -> AcousticQualityMetrics {
    let n = samples.len();
    if n == 0 {
        return AcousticQualityMetrics {
            sample_rate,
            channels,
            duration_secs: 0.0,
            rms_energy: 0.0,
            peak_amplitude: 0.0,
            clipping_ratio: 0.0,
            spectral_entropy: 0.0,
            spectral_flatness: 0.0,
            high_freq_ratio: 0.0,
            is_valid_rain_texture: false,
        };
    }

    let duration_secs = n as f32 / sample_rate as f32;
    let mut sum_sq = 0.0f32;
    let mut peak_amplitude = 0.0f32;
    let mut clipped_count = 0usize;

    for &s in samples {
        let abs_s = s.abs();
        sum_sq += s * s;
        if abs_s > peak_amplitude {
            peak_amplitude = abs_s;
        }
        if abs_s >= 0.999 {
            clipped_count += 1;
        }
    }

    let rms_energy = (sum_sq / n as f32).sqrt();
    let clipping_ratio = clipped_count as f32 / n as f32;

    // FFT spectral analysis
    const FFT_SIZE: usize = 1024;
    const HOP_SIZE: usize = 512;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let num_hops = n.saturating_sub(FFT_SIZE) / HOP_SIZE;
    let mut total_entropy = 0.0f32;
    let mut total_flatness = 0.0f32;
    let mut total_hf_ratio = 0.0f32;
    let mut valid_hops = 0usize;

    let mut buffer = vec![Complex { re: 0.0f32, im: 0.0f32 }; FFT_SIZE];
    let half = FFT_SIZE / 2;
    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;
    let hf_bin_start = (4000.0 / bin_hz).clamp(1.0, (half - 1) as f32) as usize;

    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos()))
        .collect();

    for h in 0..num_hops {
        let offset = h * HOP_SIZE;
        for (i, c) in buffer.iter_mut().enumerate().take(FFT_SIZE) {
            *c = Complex {
                re: samples[offset + i] * window[i],
                im: 0.0,
            };
        }

        fft.process(&mut buffer);

        let mut sum_power = 0.0f32;
        let mut log_power_sum = 0.0f32;
        let mut hf_power = 0.0f32;
        let mut powers = Vec::with_capacity(half);

        for (k, c) in buffer.iter().enumerate().take(half) {
            let p = c.re * c.re + c.im * c.im + 1e-12;
            powers.push(p);
            sum_power += p;
            log_power_sum += p.ln();
            if k >= hf_bin_start {
                hf_power += p;
            }
        }

        if sum_power > 1e-8 {
            // Shannon spectral entropy
            let mut hop_entropy = 0.0f32;
            let inv_sum = 1.0 / sum_power;
            for &p in &powers {
                let prob = p * inv_sum;
                if prob > 1e-9 {
                    hop_entropy -= prob * prob.log2();
                }
            }
            let norm_entropy = (hop_entropy / (half as f32).log2()).clamp(0.0, 1.0);

            // Spectral flatness (Wiener entropy)
            let geom_mean = (log_power_sum / half as f32).exp();
            let arith_mean = sum_power / half as f32;
            let flatness = (geom_mean / (arith_mean + 1e-12)).clamp(0.0, 1.0);

            // High frequency ratio
            let hf_ratio = (hf_power / sum_power).clamp(0.0, 1.0);

            total_entropy += norm_entropy;
            total_flatness += flatness;
            total_hf_ratio += hf_ratio;
            valid_hops += 1;
        }
    }

    let spectral_entropy = if valid_hops > 0 {
        total_entropy / valid_hops as f32
    } else {
        0.5
    };
    let spectral_flatness = if valid_hops > 0 {
        total_flatness / valid_hops as f32
    } else {
        0.2
    };
    let high_freq_ratio = if valid_hops > 0 {
        total_hf_ratio / valid_hops as f32
    } else {
        0.2
    };

    // Quality gate: require audible signal, limited clipping, broadband entropy
    let is_valid_rain_texture = rms_energy >= 1e-4
        && peak_amplitude >= 1e-3
        && clipping_ratio <= 0.05
        && spectral_entropy >= 0.35;

    AcousticQualityMetrics {
        sample_rate,
        channels,
        duration_secs,
        rms_energy,
        peak_amplitude,
        clipping_ratio,
        spectral_entropy,
        spectral_flatness,
        high_freq_ratio,
        is_valid_rain_texture,
    }
}

/// Reads a WAV audio file from disk and computes its `AcousticQualityMetrics`.
pub fn analyze_wav_file(path: &Path) -> Result<AcousticQualityMetrics> {
    let mut reader = WavReader::open(path).context("Failed to open WAV for acoustic inspection")?;
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
    let n_frames = raw_samples.len() / channels.max(1);
    let mut mono = Vec::with_capacity(n_frames);
    let inv_ch = 1.0 / channels as f32;

    for frame in raw_samples.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() * inv_ch);
    }

    Ok(analyze_pcm_samples(&mono, spec.sample_rate, spec.channels))
}

/// Computes the cryptographic SHA-256 hexadecimal hash of a file on disk.
pub fn compute_file_sha256(path: &Path) -> Result<String> {
    let mut file = StdFile::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Provenance metadata record for an ingested audio asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub filename: String,
    pub source_url: String,
    pub source_platform: String,
    pub category: String,
    pub canonical_surface: CanonicalSurface,
    pub license: String,
    pub license_tier: LicenseTier,
    pub sha256: String,
    pub file_size_bytes: u64,
    pub quality: Option<AcousticQualityMetrics>,
}

/// Complete dataset provenance audit manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceManifest {
    pub generated_at_utc: String,
    pub total_sources: usize,
    pub normalized_surface_diversity: f32,
    pub category_distribution: HashMap<String, usize>,
    pub surface_distribution: HashMap<String, usize>,
    pub records: Vec<ProvenanceRecord>,
}

pub async fn download_file_with_retry(
    client: &reqwest::Client,
    url: &str,
    destination: &Path,
) -> Result<()> {
    let mut retries = 3;
    let mut delay = Duration::from_secs(2);

    loop {
        let mut request = client.get(url);

        if url.contains("huggingface.co") {
            if let Ok(token) = std::env::var("HF_TOKEN") {
                request = request.bearer_auth(token);
            }
        }

        match request.send().await {
            Ok(resp) if resp.status().is_success() => {
                let mut file = File::create(destination).await?;
                let mut stream = resp.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let data = chunk?;
                    file.write_all(&data).await?;
                }
                return Ok(());
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_client_error() {
                    bail!("Fatal HTTP status: {}", status);
                }
                if retries == 0 {
                    bail!("Failed with HTTP status {} after retries", status);
                }
            }
            Err(e) => {
                if retries == 0 {
                    bail!("Network error: {}", e);
                }
            }
        }

        retries -= 1;
        tokio::time::sleep(delay).await;
        delay *= 2;
    }
}

fn chrono_lite_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}s_since_epoch", duration.as_secs(), duration.subsec_millis())
}

use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

/// In-process dataset ingestion pipeline callable on a background thread.
pub fn run_ingestion_pipeline(
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_ingestion_pipeline_async(stop_signal, log_tx))
}

pub async fn run_ingestion_pipeline_async(
    stop_signal: Arc<AtomicBool>,
    log_tx: Option<Sender<String>>,
) -> Result<usize> {
    let emit_log = |msg: String| {
        tracing::info!("{}", msg);
        if let Some(ref tx) = log_tx {
            let _ = tx.send(msg);
        }
    };

    emit_log("[*] Starting in-process Asynchronous Multi-Source Audio Ingest (Pure-Rust)...".to_string());

    let db_path = shared::paths::WorkspacePaths::resolve_sources()
        .unwrap_or_else(|| PathBuf::from("sources.json"));
    if !db_path.exists() {
        emit_log("[!] sources.json catalog not found.".to_string());
        return Ok(0);
    }

    let db_content = tokio::fs::read_to_string(&db_path).await?;
    let curated_sources: Vec<DownloadItem> = serde_json::from_str(&db_content)?;

    let attr_path = shared::paths::WorkspacePaths::resolve_attributions()
        .unwrap_or_else(|| PathBuf::from("Data/rain/ATTRIBUTIONS.txt"));
    let target_dir = attr_path.parent().unwrap_or(Path::new("Data/rain")).to_path_buf();
    tokio::fs::create_dir_all(&target_dir).await?;
    let provenance_file = target_dir.join("manifest_provenance.json");

    let client = reqwest::Client::builder()
        .user_agent("RainAI-Dataset-Collector/4.0 (Pure-Rust; Zero-Python)")
        .timeout(Duration::from_secs(45))
        .pool_idle_timeout(Duration::from_secs(15))
        .build()?;

    let mut quota = SurfaceBalanceQuota::new(5);
    for item in &curated_sources {
        quota.record(CanonicalSurface::from_category_tag(&item.category));
    }

    emit_log(format!(
        "[*] Catalog: {} candidate sources across 9 canonical surfaces (Normalized Diversity: {:.1}%)",
        curated_sources.len(),
        quota.normalized_diversity() * 100.0
    ));

    let mut downloaded_count = 0usize;
    let mut provenance_records = Vec::new();
    let mut category_distribution: HashMap<String, usize> = HashMap::new();
    let mut surface_distribution: HashMap<String, usize> = HashMap::new();

    for item in curated_sources {
        if stop_signal.load(Ordering::Relaxed) {
            emit_log("[!] Audio ingest aborted by user token.".to_string());
            break;
        }

        let canonical = CanonicalSurface::from_category_tag(&item.category);
        let (allowed, tier, reason) = LicenseVerifier::verify(&item.license);
        if !allowed {
            emit_log(format!("  [i] Skipping non-approved source '{}': {}", item.filename, reason));
            continue;
        }

        // Account for absence of easily accessed Python environment:
        // Validate direct HTTP/HTTPS endpoint. If an external CLI method is declared,
        // log a clear informative note and skip rather than attempting an invalid request.
        if item.ingest_method != "direct_http" || (!item.url.starts_with("http://") && !item.url.starts_with("https://")) {
            emit_log(format!(
                "  [i] Skipping source '{}' (Method: {}, requires external Python CLI; pure-Rust HTTP mode active)",
                item.filename, item.ingest_method
            ));
            continue;
        }

        let dest = target_dir.join(&item.filename);
        let already_existed = dest.exists();

        if !already_existed {
            emit_log(format!("  -> Downloading '{}' ({}, {:?})", item.filename, canonical.as_str(), tier));
            match download_file_with_retry(&client, &item.url, &dest).await {
                Ok(()) => {
                    downloaded_count += 1;
                }
                Err(e) => {
                    emit_log(format!("  [!] Failed downloading {}: {}", item.filename, e));
                    if dest.exists() {
                        let _ = tokio::fs::remove_file(&dest).await;
                    }
                    continue;
                }
            }
        }

        // Provenance metadata & acoustic screening
        let sha256 = compute_file_sha256(&dest).unwrap_or_else(|_| "hash_error".to_string());
        let file_size_bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);

        let quality = if item.filename.ends_with(".wav") {
            analyze_wav_file(&dest).ok()
        } else {
            None
        };

        let log_line = format!(
            "Platform: {} | File: {} | Category: {} (Surface: {}) | Tier: {:?} | License: {} | SHA256: {} | URL: {}\n",
            item.source_platform, item.filename, item.category, canonical.as_str(), tier, item.license, sha256, item.url
        );

        // Append to ATTRIBUTIONS.txt if newly acquired
        if !already_existed {
            if let Ok(mut f) = tokio::fs::OpenOptions::new().create(true).append(true).open(&attr_path).await {
                let _ = f.write_all(log_line.as_bytes()).await;
            }
        }

        *category_distribution.entry(item.category.clone()).or_insert(0) += 1;
        *surface_distribution.entry(canonical.as_str().to_string()).or_insert(0) += 1;

        provenance_records.push(ProvenanceRecord {
            filename: item.filename,
            source_url: item.url,
            source_platform: item.source_platform,
            category: item.category,
            canonical_surface: canonical,
            license: item.license,
            license_tier: tier,
            sha256,
            file_size_bytes,
            quality,
        });
    }

    // Persist complete provenance manifest
    let manifest = ProvenanceManifest {
        generated_at_utc: chrono_lite_timestamp(),
        total_sources: provenance_records.len(),
        normalized_surface_diversity: quota.normalized_diversity(),
        category_distribution,
        surface_distribution,
        records: provenance_records,
    };
    if let Ok(json_str) = serde_json::to_string_pretty(&manifest) {
        let _ = tokio::fs::write(&provenance_file, json_str).await;
    }

    emit_log(format!(
        "[+] Ingestion finished. {} new sources acquired (total verified: {}). Provenance saved to {:?}.",
        downloaded_count, manifest.total_sources, provenance_file
    ));
    Ok(downloaded_count)
}

