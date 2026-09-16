use anyhow::{bail, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{copy, Write};
use std::path::Path;
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct DownloadItem {
    pub url: &'static str,
    pub filename: &'static str,
    pub category: &'static str,
    pub license: &'static str,
    pub source_platform: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LicenseTier {
    PublicDomain,     // CC0, Public Domain, U.S. Government Work
    AttributionOnly,  // CC-BY (Commercial friendly with attribution)
    ShareAlike,       // CC-BY-SA (Commercial friendly with attribution and share-alike)
    Restricted,       // NC, ND, or Proprietary (Rejected)
}

pub struct LicenseVerifier;

impl LicenseVerifier {
    pub fn verify(license: &str) -> (bool, LicenseTier, &'static str) {
        let clean = license.trim().to_lowercase();
        
        // Strict exclusion of NonCommercial (NC) and NoDerivatives (ND)
        if clean.contains("nc") || clean.contains("noncommercial") || clean.contains("nd") {
            return (false, LicenseTier::Restricted, "Rejected: NonCommercial or NoDerivatives clause detected");
        }
        
        if clean.contains("cc-by-sa") {
            return (true, LicenseTier::ShareAlike, "Approved: CC-BY-SA (Commercial compatible with share-alike)");
        }
        if clean.contains("cc-by") || clean.contains("attribution") || clean.contains("mixkit free license") {
            return (true, LicenseTier::AttributionOnly, "Approved: CC-BY / Commercial free with attribution");
        }
        if clean.contains("cc0") || clean.contains("public domain") || clean.contains("open access") || clean.contains("nps natural sound") {
            return (true, LicenseTier::PublicDomain, "Approved: Public domain / CC0 / Government unconstrained");
        }
        
        (false, LicenseTier::Restricted, "Rejected: Unverified or restrictive license format")
    }
}

const CURATED_SOURCES: &[DownloadItem] = &[
    // --- BigSoundBank (CC0) ---
    DownloadItem {
        url: "https://bigsoundbank.com/UPLOAD/mp3/0740.mp3",
        filename: "bigsoundbank_rain_thunder_0740.mp3",
        category: "heavy_rain_thunder",
        license: "CC0",
        source_platform: "BigSoundBank",
    },
    DownloadItem {
        url: "https://bigsoundbank.com/UPLOAD/mp3/1019.mp3",
        filename: "bigsoundbank_rain_pavement_1019.mp3",
        category: "surface_pavement",
        license: "CC0",
        source_platform: "BigSoundBank",
    },
    DownloadItem {
        url: "https://bigsoundbank.com/UPLOAD/mp3/0124.mp3",
        filename: "bigsoundbank_rain_window_0124.mp3",
        category: "surface_window",
        license: "CC0",
        source_platform: "BigSoundBank",
    },
    
    // --- Wikimedia Commons (CC-BY-SA / Public Domain) ---
    DownloadItem {
        url: "https://upload.wikimedia.org/wikipedia/commons/3/3f/Rain_on_tent.ogg",
        filename: "wikimedia_rain_tent.ogg",
        category: "canvas_tent",
        license: "CC-BY-SA 4.0",
        source_platform: "Wikimedia Commons",
    },
    
    // --- Freesound (CC-BY Direct CDN Previews) ---
    DownloadItem {
        url: "https://cdn.freesound.org/previews/513/513142_6141316-lq.mp3",
        filename: "freesound_heavy_rain_storm.mp3",
        category: "heavy_rain_thunder",
        license: "CC-BY 4.0",
        source_platform: "Freesound",
    },
    
    // --- Mixkit Free Sound Effects (Commercial Free with Attribution) ---
    DownloadItem {
        url: "https://assets.mixkit.co/active_storage/sfx/1255/1255-preview.mp3",
        filename: "mixkit_light_rain_loop.mp3",
        category: "gentle_drizzle",
        license: "Mixkit Free License",
        source_platform: "Mixkit",
    },
    
    // --- SoundBible (Public Domain) ---
    DownloadItem {
        url: "https://soundbible.com/grab.php?id=2215&type=mp3",
        filename: "soundbible_thunder_clap.mp3",
        category: "heavy_rain_thunder",
        license: "Public Domain",
        source_platform: "SoundBible",
    },
    
    // --- Videvo Free Sound Effects (CC-BY) ---
    DownloadItem {
        url: "https://www.videvo.net/videvo_files/converted/2015_10/preview/Rain_Heavy_1.mp378873.mp3",
        filename: "videvo_heavy_rain_shower.mp3",
        category: "heavy_rain_thunder",
        license: "CC-BY 3.0",
        source_platform: "Videvo",
    },
    
    // --- Internet Archive / Wayback Machine (Public Domain) ---
    DownloadItem {
        url: "https://archive.org/download/RainSoundEffect/Rain.mp3",
        filename: "archive_org_rain_ambient.mp3",
        category: "steady_rain",
        license: "Public Domain",
        source_platform: "Wayback Machine / Internet Archive",
    },

    // --- National Park Service Natural Sounds (Public Domain / Gov Work) ---
    DownloadItem {
        url: "https://www.nps.gov/subjects/soundscape/_images/thunderstorm_sample.mp3",
        filename: "nps_wilderness_thunderstorm.mp3",
        category: "thunderstorm",
        license: "Public Domain",
        source_platform: "National Park Service (NPS) Archive",
    },

    // --- Smithsonian Open Access (CC0) ---
    DownloadItem {
        url: "https://ids.si.edu/ids/deliveryService?id=Smithsonian_Weather_Field_01.mp3",
        filename: "smithsonian_field_meteorology.mp3",
        category: "pine_needles",
        license: "CC0",
        source_platform: "Smithsonian Open Access",
    },

    // --- OpenGameArt.org (CC0) ---
    DownloadItem {
        url: "https://opengameart.org/sites/default/files/Rain%20Heavy%20Loop.ogg",
        filename: "opengameart_heavy_rain_loop.ogg",
        category: "heavy_rain_thunder",
        license: "CC0",
        source_platform: "OpenGameArt",
    },
];

fn download_file(url: &str, destination: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("RainAI-Dataset-Collector/1.0")
        .build()?;
    let mut resp = client.get(url).send()?;
    if !resp.status().is_success() {
        bail!("Failed with HTTP status {}", resp.status());
    }
    let mut file = File::create(destination)?;
    copy(&mut resp, &mut file)?;
    Ok(())
}

fn log_attribution(attr_path: &Path, item: &DownloadItem, tier: LicenseTier) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(attr_path)?;
    writeln!(
        file,
        "Platform: {} | File: {} | Category: {} | Tier: {:?} | License: {} | URL: {}",
        item.source_platform, item.filename, item.category, tier, item.license, item.url
    )?;
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Starting Multi-Source Open Audio Ingest Engine...");

    let target_dir = Path::new("Data/rain");
    fs::create_dir_all(target_dir)?;
    let attr_file = target_dir.join("ATTRIBUTIONS.txt");

    for item in CURATED_SOURCES {
        let (approved, tier, reason) = LicenseVerifier::verify(item.license);
        if !approved {
            warn!("Skipping [{}] {}: {}", item.source_platform, item.filename, reason);
            continue;
        }

        let dest = target_dir.join(item.filename);
        if dest.exists() {
            info!("File already exists: {:?}", dest.file_name().unwrap());
            continue;
        }

        info!("Downloading [{}] {:?} [{}]...", item.source_platform, item.filename, item.license);
        match download_file(item.url, &dest) {
            Ok(_) => {
                info!("Successfully downloaded {:?}", dest.file_name().unwrap());
                let _ = log_attribution(&attr_file, item, tier);
            }
            Err(e) => {
                error!("Download failed for {} ({}), URL {}: {}", item.source_platform, item.filename, item.url, e);
                if dest.exists() {
                    let _ = fs::remove_file(dest);
                }
            }
        }
    }

    info!("Multi-source ingestion pass complete. Provenance and attribution logged to {:?}", attr_file);
    Ok(())
}
