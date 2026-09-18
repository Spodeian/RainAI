use anyhow::Result;
use futures::StreamExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tracing::{error, info, warn};
use utilities::ingest::{
    analyze_wav_file, compute_file_sha256, download_file_with_retry, CanonicalSurface,
    DownloadItem, LicenseVerifier, ProvenanceManifest, ProvenanceRecord, SurfaceBalanceQuota,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Asynchronous Multi-Source Open Audio Ingest Engine v4.0...");

    // 1. Read and parse the JSON database
    let db_path = PathBuf::from("sources.json");
    let db_content = fs::read_to_string(&db_path).await?;
    let curated_sources: Vec<DownloadItem> = serde_json::from_str(&db_content)?;

    let target_dir = PathBuf::from("Data/rain");
    fs::create_dir_all(&target_dir).await?;
    let attr_file = target_dir.join("ATTRIBUTIONS.txt");
    let provenance_file = target_dir.join("manifest_provenance.json");

    // 2. Audit surface representation and diversity entropy
    let mut quota = SurfaceBalanceQuota::new(5);
    for item in &curated_sources {
        quota.record(CanonicalSurface::from_category_tag(&item.category));
    }

    info!(
        "Audited {} candidate sources across 9 canonical surfaces. Shannon diversity index: {:.3} (Normalized: {:.1}%)",
        curated_sources.len(),
        quota.shannon_entropy(),
        quota.normalized_diversity() * 100.0
    );

    let underrepresented = quota.underrepresented_surfaces();
    if !underrepresented.is_empty() {
        warn!(
            "Underrepresented surfaces (< {} sources): {:?}",
            quota.target_per_surface,
            underrepresented
                .into_iter()
                .map(|(s, c)| format!("{}: {}", s.as_str(), c))
                .collect::<Vec<_>>()
        );
    }

    let client = reqwest::Client::builder()
        .user_agent("RainAI-Dataset-Collector/4.0")
        .timeout(Duration::from_secs(45))
        .pool_idle_timeout(Duration::from_secs(15))
        .build()?;

    // 3. Create concurrent async pipeline capping out at 8 active HTTP sockets
    let stream = futures::stream::iter(curated_sources.into_iter().map(|item| {
        let client = client.clone();
        let target_dir = target_dir.clone();

        async move {
            let canonical = CanonicalSurface::from_category_tag(&item.category);
            let (approved, tier, reason) = LicenseVerifier::verify(&item.license);
            if !approved {
                warn!("Skipping [{}] {}: {}", item.source_platform, item.filename, reason);
                return None;
            }

            if item.ingest_method != "direct_http" {
                info!("Delegating {} to external pipeline (Method: {})", item.filename, item.ingest_method);
                return None;
            }

            let dest = target_dir.join(&item.filename);
            let already_existed = dest.exists();

            if !already_existed {
                info!("Downloading [{}] {:?} [{}]...", item.source_platform, item.filename, item.license);
                if let Err(e) = download_file_with_retry(&client, &item.url, &dest).await {
                    error!("Download failed for {} ({}), URL {}: {}", item.source_platform, item.filename, item.url, e);
                    if dest.exists() {
                        let _ = fs::remove_file(&dest).await;
                    }
                    return None;
                }
                info!("Successfully downloaded {:?}", dest.file_name().unwrap());
            } else {
                info!("File already exists: {:?}", dest.file_name().unwrap());
            }

            // Cryptographic checksum and file size
            let sha256 = compute_file_sha256(&dest).unwrap_or_else(|_| "hash_error".to_string());
            let file_size_bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);

            // Acoustic quality screening for WAV audio
            let quality = if item.filename.ends_with(".wav") {
                match analyze_wav_file(&dest) {
                    Ok(q) => {
                        if !q.is_valid_rain_texture {
                            warn!(
                                "Acoustic screening warning for {}: low energy/flatness (RMS={:.4}, Flatness={:.3}, Entropy={:.3})",
                                item.filename, q.rms_energy, q.spectral_flatness, q.spectral_entropy
                            );
                        }
                        Some(q)
                    }
                    Err(e) => {
                        warn!("Could not compute acoustic quality for {}: {}", item.filename, e);
                        None
                    }
                }
            } else {
                None
            };

            let log_line = format!(
                "Platform: {} | File: {} | Category: {} (Surface: {}) | Tier: {:?} | License: {} | SHA256: {} | URL: {}\n",
                item.source_platform, item.filename, item.category, canonical.as_str(), tier, item.license, sha256, item.url
            );

            let record = ProvenanceRecord {
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
            };

            Some((log_line, record))
        }
    }))
    .buffer_unordered(8);

    // Collect results
    let results: Vec<Option<(String, ProvenanceRecord)>> = stream.collect().await;
    let mut successful_logs = Vec::new();
    let mut provenance_records = Vec::new();
    let mut cat_distribution: HashMap<String, usize> = HashMap::new();
    let mut surf_distribution: HashMap<String, usize> = HashMap::new();

    for (log, rec) in results.into_iter().flatten() {
        *cat_distribution.entry(rec.category.clone()).or_insert(0) += 1;
        *surf_distribution.entry(rec.canonical_surface.as_str().to_string()).or_insert(0) += 1;
        successful_logs.push(log);
        provenance_records.push(rec);
    }

    if !successful_logs.is_empty() {
        let mut file = OpenOptions::new().create(true).append(true).open(&attr_file).await?;
        for log in successful_logs {
            file.write_all(log.as_bytes()).await?;
        }
    }

    let manifest = ProvenanceManifest {
        generated_at_utc: chrono_lite_timestamp(),
        total_sources: provenance_records.len(),
        normalized_surface_diversity: quota.normalized_diversity(),
        category_distribution: cat_distribution,
        surface_distribution: surf_distribution,
        records: provenance_records,
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&provenance_file, manifest_json).await?;

    info!(
        "Multi-source ingestion pass complete. Provenance manifest saved to {:?}, attributions logged to {:?}",
        provenance_file, attr_file
    );
    Ok(())
}

fn chrono_lite_timestamp() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{}s_since_epoch", duration.as_secs(), duration.subsec_millis())
}