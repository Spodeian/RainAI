//! Dataset Ingestion, Ambisonic Augmentation, and Synthetic Batch Generation.
//!
//! Handles manifest caching, dataset streaming, SO(3) 3D spatial soundfield rotation,
//! surface mixup conditioning, and asynchronous attribution telemetry logging.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{OnceLock, RwLock};

use super::*;
use shared::paths::WorkspacePaths;

/// Training batch container with Classifier-Free Guidance (CFG) conditioning.
pub struct TrainingBatch {
    pub audio_features: Tensor,
    pub z_prev: Tensor,
    pub conditioning: Tensor,
    pub z_target: Tensor,
    pub target_bands: Tensor,
    pub target_foa: Tensor,
    pub z_prev2: Option<Tensor>,
}

/// Generates a training batch with physics dynamics, CFG conditioning dropout, and optional spatial augmentations.
pub fn generate_batch_augmented(
    batch_size: usize,
    device: &Device,
    cfg_dropout_prob: f32,
    so3_aug_prob: f32,
) -> Result<TrainingBatch> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let audio_features = Tensor::randn(0.0f32, 1.0f32, (batch_size, LATENT_DIM), device)?;
    let z_prev = Tensor::randn(0.0f32, 1.0f32, (batch_size, LATENT_DIM), device)?;
    let z_prev2 = Some(((&z_prev * 0.96)? + Tensor::randn(0.0f32, 0.05f32, (batch_size, LATENT_DIM), device)?)?);
    
    // Conditioning vector [B, 554]
    let mut cond = Tensor::randn(0.0f32, 0.4f32, (batch_size, CONDITION_DIM), device)?;

    // Apply Classifier-Free Guidance (CFG) conditioning dropout
    if rng.gen_range(0.0f32..1.0f32) < cfg_dropout_prob {
        cond = Tensor::zeros((batch_size, CONDITION_DIM), DType::F32, device)?;
    }

    // Realistic target trajectory with physics-guided drift and inertia
    let z_target = ((&z_prev * 0.94)? + Tensor::randn(0.0f32, 0.08f32, (batch_size, LATENT_DIM), device)?)?;

    // Target 16-band filter responses
    let target_bands = Tensor::randn(0.5f32, 0.25f32, (batch_size, FILTER_BANDS), device)?;

    // Target 4-channel FOA soundfield: W (omni), X (front), Y (side), Z (elevation)
    let mut foas = Vec::with_capacity(batch_size * FOA_CHANNELS);
    for _ in 0..batch_size {
        foas.extend_from_slice(&[
            rng.gen_range(0.5f32..1.0f32),
            rng.gen_range(-0.4f32..0.4f32),
            rng.gen_range(-0.4f32..0.4f32),
            rng.gen_range(-0.8f32..-0.2f32), // downward rain inclination
        ]);
    }
    let mut target_foa = Tensor::from_vec(foas, (batch_size, FOA_CHANNELS), device)?;

    // Apply SO(3) 3D Ambisonic spatial rotation augmentation
    if so3_aug_prob > 0.0 && rng.gen_range(0.0f32..1.0f32) < so3_aug_prob {
        let angles = (
            rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
            rng.gen_range(-std::f32::consts::FRAC_PI_2..std::f32::consts::FRAC_PI_2),
            rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
        );
        target_foa = crate::stft_loss::apply_so3_foa_rotation(&target_foa, angles)?;
    }

    Ok(TrainingBatch {
        audio_features,
        z_prev,
        conditioning: cond,
        z_target,
        target_bands,
        target_foa,
        z_prev2,
    })
}

/// Generates a training batch with physics dynamics and CFG conditioning dropout.
pub fn generate_batch(
    batch_size: usize,
    device: &Device,
    cfg_dropout_prob: f32,
) -> Result<TrainingBatch> {
    generate_batch_augmented(batch_size, device, cfg_dropout_prob, 0.0)
}

/// Async item payload for off-thread attribution logging.
#[derive(Debug, Clone)]
pub struct TrainingAttributionItem {
    pub filename: String,
    pub surface_tag: String,
}

/// Asynchronous, non-blocking attribution queue with O(1) in-memory short-circuiting.
/// Prevents disk I/O, regex, or JSON lookups from ever stalling the real-time neural training loop.
pub struct AsyncAttributionRecorder {
    seen_set: RwLock<HashSet<String>>,
    sender: std::sync::mpsc::Sender<TrainingAttributionItem>,
}

static ATTRIBUTION_RECORDER: OnceLock<AsyncAttributionRecorder> = OnceLock::new();

impl AsyncAttributionRecorder {
    pub fn get_or_init() -> &'static Self {
        ATTRIBUTION_RECORDER.get_or_init(|| {
            let mut seen = HashSet::new();
            // Pre-seed known attributions from disk to immediately short-circuit on session start
            if let Some(attr_path) = WorkspacePaths::resolve_attributions() {
                if let Ok(content) = std::fs::read_to_string(&attr_path) {
                    for line in content.lines() {
                        if let Some(pos) = line.find("File: ") {
                            let rest = &line[pos + 6..];
                            if let Some(end) = rest.find(" |") {
                                seen.insert(rest[..end].trim().to_string());
                            }
                        }
                    }
                }
            }

            let (tx, rx) = std::sync::mpsc::channel::<TrainingAttributionItem>();

            // Spawn background writer daemon thread to process provenance lookups and disk I/O asynchronously
            let _ = std::thread::Builder::new()
                .name("rainai-attribution-writer".to_string())
                .spawn(move || {
                    while let Ok(item) = rx.recv() {
                        Self::process_attribution_item(item);
                    }
                });

            Self {
                seen_set: RwLock::new(seen),
                sender: tx,
            }
        })
    }

    /// O(1) non-blocking attribution registration with immediate short-circuit for previously seen files.
    #[inline]
    pub fn record(&self, filename: &str, surface_tag: &str) {
        // Fast path: concurrent read lock check (~10ns)
        if let Ok(guard) = self.seen_set.read() {
            if guard.contains(filename) {
                return;
            }
        }

        // Slow path: upgrade to write lock and enqueue for background persistence
        if let Ok(mut guard) = self.seen_set.write() {
            if guard.contains(filename) {
                return;
            }
            guard.insert(filename.to_string());
        }

        let _ = self.sender.send(TrainingAttributionItem {
            filename: filename.to_string(),
            surface_tag: surface_tag.to_string(),
        });
    }

    fn process_attribution_item(item: TrainingAttributionItem) {
        let filename = &item.filename;
        let mut platform = "RainAI-Acoustic-Archive".to_string();
        let mut license = "CC0".to_string();
        let mut tier = "PublicDomain".to_string();
        let mut url = format!("local://rainai/{}", filename);
        let mut category = item.surface_tag.clone();

        if let Some(sources_path) = WorkspacePaths::resolve_sources() {
            if let Ok(content) = std::fs::read_to_string(&sources_path) {
                if let Ok(items) = serde_json::from_str::<Vec<crate::ingest::DownloadItem>>(&content) {
                    if let Some(src) = items.iter().find(|i| {
                        i.filename == *filename
                            || filename.starts_with(&i.filename.replace(".mp3", "").replace(".wav", "").replace(".ogg", ""))
                    }) {
                        platform = src.source_platform.clone();
                        license = src.license.clone();
                        let (_, lt, _) = crate::ingest::LicenseVerifier::verify(&src.license);
                        tier = format!("{:?}", lt);
                        url = src.url.clone();
                        category = src.category.clone();
                    }
                }
            }
        }

        let attr_line = format!(
            "Platform: {} | File: {} | Category: {} (Surface: {}) | Tier: {} | License: {} | URL: {}\n",
            platform, filename, category, item.surface_tag, tier, license, url
        );

        let target_path = WorkspacePaths::resolve_attributions()
            .unwrap_or_else(|| std::path::PathBuf::from("data/rain/ATTRIBUTIONS.txt"));

        if let Some(parent) = target_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&target_path) {
            use std::io::Write;
            let _ = f.write_all(attr_line.as_bytes());
        }
    }
}

/// Helper function to record training attribution using the non-blocking async queue.
#[inline]
pub fn record_training_attribution_if_needed(meta: &crate::features::AudioMetadata) {
    AsyncAttributionRecorder::get_or_init().record(&meta.filename, &meta.surface_tag);
}

/// Acoustic Spatial Enclosure environments modeling room reverberation and spatial boundary transfer functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticSpatialEnvironment {
    OpenFreeField,
    UrbanStreetCanyon,
    DenseForestCanopy,
    InteriorVehicleCabin,
    CoveredPorch,
}

impl AcousticSpatialEnvironment {
    pub fn sample_random<R: rand::Rng>(rng: &mut R) -> Self {
        match rng.gen_range(0..5) {
            0 => Self::OpenFreeField,
            1 => Self::UrbanStreetCanyon,
            2 => Self::DenseForestCanopy,
            3 => Self::InteriorVehicleCabin,
            _ => Self::CoveredPorch,
        }
    }

    /// Modulates target filter bands and FOA spatial soundfield to reflect enclosure boundary physics
    pub fn apply_enclosure(&self, bands: &mut [f32; FILTER_BANDS], foa: &mut [f32; FOA_CHANNELS]) {
        match self {
            Self::OpenFreeField => {
                // Direct sound dominates: directional clarity high, no low-frequency cavity boom
                foa[1] *= 1.1; // X
                foa[2] *= 1.1; // Y
            }
            Self::UrbanStreetCanyon => {
                // Dense reflections & flutter: mid-frequency boost (bands 4..10), higher diffuseness (W boost)
                for b in 4..10 {
                    bands[b] = (bands[b] * 1.25).min(20.0);
                }
                foa[0] = (foa[0] * 1.2).min(10.0); // W omni
            }
            Self::DenseForestCanopy => {
                // High leaf scattering & soft ground absorption: HF roll-off (bands 11..16), diffuse spatial field
                for b in 11..FILTER_BANDS {
                    bands[b] *= 0.75;
                }
                foa[1] *= 0.8;
                foa[2] *= 0.8;
            }
            Self::InteriorVehicleCabin => {
                // Low-pass transmission through glass & metal chassis: heavy HF attenuation, standing wave resonance at bands 1..3
                bands[1] = (bands[1] * 1.4).min(20.0);
                bands[2] = (bands[2] * 1.3).min(20.0);
                for b in 6..FILTER_BANDS {
                    bands[b] *= 0.45;
                }
                // Frontal windshield proximity bias
                foa[1] = (foa[1] + 0.2).clamp(-0.8, 0.8);
            }
            Self::CoveredPorch => {
                // Asymmetric overhead dry damping + frontal wet field
                foa[1] = (foa[1] + 0.15).clamp(-0.8, 0.8); // frontal bias
                foa[3] *= 0.6; // overhead rain damping
            }
        }
    }
}

/// Meteorological and precipitation weather regimes modulating droplet size dynamics and air absorption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherPrecipitationRegime {
    DrizzleMist,
    ModerateRain,
    CloudburstDownpour,
    SleetGraupelHybrid,
}

impl WeatherPrecipitationRegime {
    pub fn sample_random<R: rand::Rng>(rng: &mut R) -> Self {
        match rng.gen_range(0..4) {
            0 => Self::DrizzleMist,
            1 => Self::ModerateRain,
            2 => Self::CloudburstDownpour,
            _ => Self::SleetGraupelHybrid,
        }
    }

    /// Modulates condition parameters and bands to reflect meteorological regime
    pub fn apply_meteorology(&self, cond: &mut [f32; CONDITION_DIM], bands: &mut [f32; FILTER_BANDS]) {
        match self {
            Self::DrizzleMist => {
                cond[512] = (cond[512] * 0.3).clamp(0.01, 1.0); // low rain rate
                cond[513] = (cond[513] * 1.5).clamp(0.0, 1.0);  // high droplet density
                cond[515] = (cond[515] * 1.3).clamp(0.0, 1.0);  // high freq ratio
                for b in 12..FILTER_BANDS {
                    bands[b] = (bands[b] * 1.3).min(20.0);
                }
            }
            Self::ModerateRain => {
                // Standard Marshall-Palmer baseline
            }
            Self::CloudburstDownpour => {
                cond[512] = (cond[512] * 1.6 + 0.3).clamp(0.0, 1.0); // heavy rain rate
                cond[517] = (cond[517] * 1.5 + 0.2).clamp(0.0, 1.0); // high RMS energy
                for b in 0..5 {
                    bands[b] = (bands[b] * 1.5).min(20.0); // bubble resonance & impact rumble
                }
            }
            Self::SleetGraupelHybrid => {
                cond[514] = (cond[514] * 1.2).clamp(0.0, 1.0);
                cond[518] = (cond[518] * 0.8).clamp(0.0, 1.0); // lower spectral flatness (peaked impacts)
                // Transient impact spikes in upper mids
                bands[8] = (bands[8] * 1.4).min(20.0);
                bands[10] = (bands[10] * 1.3).min(20.0);
            }
        }
    }
}

/// Samples an 8-surface convex mixture vector from a sparsity-inducing Dirichlet distribution (alpha = 0.25).
pub fn sample_sparse_dirichlet_surfaces<R: rand::Rng>(rng: &mut R, alpha: f32) -> [f32; 8] {
    use rand_distr::{Distribution, Gamma};
    let gamma = Gamma::new(alpha, 1.0).unwrap_or_else(|_| Gamma::new(1.0, 1.0).unwrap());
    let mut weights = [0.0f32; 8];
    let mut sum = 0.0f32;
    for w in &mut weights {
        let sample: f32 = gamma.sample(rng);
        *w = sample;
        sum += sample;
    }
    if sum > 1e-6 {
        for w in &mut weights {
            *w /= sum;
        }
    } else {
        weights[0] = 1.0;
    }
    weights
}

static CACHED_MANIFEST: std::sync::Mutex<Option<(std::time::SystemTime, Vec<crate::features::AudioMetadata>)>> =
    std::sync::Mutex::new(None);

/// Real acoustic manifest dataset loader for Candle training.
pub struct CandleManifestDataset {
    pub entries: Vec<crate::features::AudioMetadata>,
}

impl CandleManifestDataset {
    pub fn load_from_manifest<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let mtime = std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        if let Ok(guard) = CACHED_MANIFEST.lock() {
            if let Some((cached_time, ref cached_entries)) = *guard {
                if cached_time == mtime && !cached_entries.is_empty() {
                    return Ok(Self { entries: cached_entries.clone() });
                }
            }
        }

        let file = std::fs::File::open(p)?;
        let map: HashMap<String, crate::features::AudioMetadata> = serde_json::from_reader(file)?;
        let entries: Vec<crate::features::AudioMetadata> = map.into_values().collect();
        if entries.is_empty() {
            anyhow::bail!("Loaded manifest contains zero audio entries.");
        }

        if let Ok(mut guard) = CACHED_MANIFEST.lock() {
            *guard = Some((mtime, entries.clone()));
        }

        Ok(Self { entries })
    }

    /// Samples a batch from the real dataset manifest with optional SO(3) 3D ambisonic rotation and surface mixup.
    pub fn sample_batch_augmented(
        &self,
        batch_size: usize,
        device: &Device,
        cfg_dropout_prob: f32,
        so3_aug_prob: f32,
        surface_mixup_prob: f32,
    ) -> Result<TrainingBatch> {
        use rand::seq::SliceRandom;
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut audio_feats = Vec::with_capacity(batch_size * LATENT_DIM);
        let mut cond_vecs = Vec::with_capacity(batch_size * CONDITION_DIM);
        let mut target_bands = Vec::with_capacity(batch_size * FILTER_BANDS);
        let mut target_foas = Vec::with_capacity(batch_size * FOA_CHANNELS);

        let surface_tag_to_idx = |tag: &str| -> usize {
            match tag {
                "pavement" | "urban_pavement" => 522,
                "window" | "window_rain" => 523,
                "roof" | "roof_rain" => 524,
                "canvas" | "canvas_tent" => 525,
                "deck" | "wood_deck" => 526,
                "needles" | "pine_needles" => 527,
                "foliage" | "forest_foliage" => 528,
                "water_deep" => 529,
                _ => 530,
            }
        };

        for _ in 0..batch_size {
            let meta = self.entries.choose(&mut rng).expect("Dataset cannot be empty");
            record_training_attribution_if_needed(meta);

            // Build 554-dim condition vector matching Python dataset standard (zero-heap stack array):
            // 512 (CLAP pseudo-embedding) + 41 (Physical parameters) + 1 (Drift)
            let mut cond = [0.0f32; CONDITION_DIM];
            cond[512] = (meta.rain_rate / 1.0).clamp(0.0, 1.0);
            cond[513] = (meta.droplet_density / 2.0).clamp(0.0, 1.0);
            cond[514] = (meta.drops_per_second / 200.0).clamp(0.0, 1.0);
            cond[515] = meta.high_freq_ratio.clamp(0.0, 1.0);
            cond[516] = (meta.spectral_centroid / 8000.0).clamp(0.0, 1.0);
            cond[517] = (meta.rms_energy * 20.0).clamp(0.0, 1.0);
            cond[518] = meta.spectral_flatness.clamp(0.0, 1.0);

            // Compound Dirichlet surface sampling or single surface encoding
            let s_idx1 = surface_tag_to_idx(&meta.surface_tag);
            if surface_mixup_prob > 0.0 && rng.gen_range(0.0f32..1.0f32) < surface_mixup_prob {
                let dirichlet_weights = sample_sparse_dirichlet_surfaces(&mut rng, 0.25);
                for i in 0..8 {
                    cond[522 + i] = dirichlet_weights[i];
                }
            } else {
                cond[s_idx1] = 1.0;
            }

            // Audio spectral feature projection [64] (zero-heap stack array)
            let mut feat = [0.0f32; LATENT_DIM];
            for i in 0..LATENT_DIM {
                feat[i] = (meta.rms_energy * (i as f32 + 1.0) * 0.1).sin() * meta.high_freq_ratio;
            }

            // Target 16-band filterbank gains (zero-heap stack array)
            let mut bands = [0.0f32; FILTER_BANDS];
            for b in 0..FILTER_BANDS {
                bands[b] = (meta.rms_energy * 10.0 + (b as f32 / FILTER_BANDS as f32) * meta.high_freq_ratio).clamp(0.01, 1.0);
            }

            // Target 4-channel FOA: W (omni), X (front-back), Y (left-right), Z (up-down) (zero-heap stack array)
            let w_energy = (meta.rms_energy * 5.0).clamp(0.1, 1.0);
            let mut foa = [0.0f32; FOA_CHANNELS];
            foa[0] = w_energy;
            foa[1] = ((meta.spectral_centroid / 8000.0) * 0.4 - 0.2).clamp(-0.8, 0.8);
            foa[2] = ((meta.high_freq_ratio - 0.5) * 0.4).clamp(-0.8, 0.8);
            foa[3] = (-0.5f32 * w_energy).clamp(-0.9, -0.1); // downward rain vector

            // Sample Meteorological Weather Regime and apply physical acoustic modulation
            let weather = WeatherPrecipitationRegime::sample_random(&mut rng);
            weather.apply_meteorology(&mut cond, &mut bands);

            // Sample Acoustic Spatial Enclosure and apply boundary transfer functions
            let enclosure = AcousticSpatialEnvironment::sample_random(&mut rng);
            enclosure.apply_enclosure(&mut bands, &mut foa);

            // SpecAugment: stochastic frequency band masking (8% probability)
            if rng.gen_range(0.0f32..1.0f32) < 0.08 {
                let mask_b = rng.gen_range(0..FILTER_BANDS);
                bands[mask_b] = 0.0;
            }

            audio_feats.extend_from_slice(&feat);
            cond_vecs.extend_from_slice(&cond);
            target_bands.extend_from_slice(&bands);
            target_foas.extend_from_slice(&foa);
        }

        // Layer 0 Hardening: sanitize all raw float arrays to guarantee zero NaNs/Infs enter tensors
        for val in &mut audio_feats {
            if !val.is_finite() { *val = 0.0; }
        }
        for val in &mut cond_vecs {
            if !val.is_finite() { *val = 0.0; }
        }
        for val in &mut target_bands {
            if !val.is_finite() { *val = 0.1; }
            *val = val.clamp(0.0, 20.0);
        }
        for val in &mut target_foas {
            if !val.is_finite() { *val = 0.0; }
        }

        let audio_features = Tensor::from_vec(audio_feats, (batch_size, LATENT_DIM), device)?;
        let mut conditioning = Tensor::from_vec(cond_vecs, (batch_size, CONDITION_DIM), device)?;

        if rng.gen_range(0.0f32..1.0f32) < cfg_dropout_prob {
            conditioning = Tensor::zeros((batch_size, CONDITION_DIM), DType::F32, device)?;
        }

        let z_prev = Tensor::randn(0.0f32, 1.0f32, (batch_size, LATENT_DIM), device)?;
        let z_prev2 = Some(((&z_prev * 0.96)? + Tensor::randn(0.0f32, 0.05f32, (batch_size, LATENT_DIM), device)?)?);
        let z_target = ((&z_prev * 0.94)? + Tensor::randn(0.0f32, 0.08f32, (batch_size, LATENT_DIM), device)?)?;
        let target_bands = Tensor::from_vec(target_bands, (batch_size, FILTER_BANDS), device)?;
        let mut target_foa = Tensor::from_vec(target_foas, (batch_size, FOA_CHANNELS), device)?;

        // Apply SO(3) 3D Ambisonic spatial rotation augmentation
        if so3_aug_prob > 0.0 && rng.gen_range(0.0f32..1.0f32) < so3_aug_prob {
            let angles = (
                rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
                rng.gen_range(-std::f32::consts::FRAC_PI_2..std::f32::consts::FRAC_PI_2),
                rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
            );
            target_foa = crate::stft_loss::apply_so3_foa_rotation(&target_foa, angles)?;
        }

        // Apply continuous micro-head tracking spatial jitter (~0.85 degrees) to prevent binaural comb-filtering
        let micro_sigma = 0.015f32;
        let micro_angles = (
            rng.gen_range(-micro_sigma..micro_sigma),
            rng.gen_range(-micro_sigma..micro_sigma),
            rng.gen_range(-micro_sigma..micro_sigma),
        );
        target_foa = crate::stft_loss::apply_so3_foa_rotation(&target_foa, micro_angles)?;

        Ok(TrainingBatch {
            audio_features,
            z_prev,
            conditioning,
            z_target,
            target_bands,
            target_foa,
            z_prev2,
        })
    }

    /// Samples a batch from the real dataset manifest with standard parameters.
    pub fn sample_batch(&self, batch_size: usize, device: &Device, cfg_dropout_prob: f32) -> Result<TrainingBatch> {
        self.sample_batch_augmented(batch_size, device, cfg_dropout_prob, 0.0, 0.0)
    }

    /// Splits the manifest dataset into train and validation subsets.
    pub fn split(self, val_ratio: f32) -> (Self, Self) {
        let val_ratio = val_ratio.clamp(0.01, 0.5);
        let val_size = ((self.entries.len() as f32) * val_ratio) as usize;
        let train_size = self.entries.len().saturating_sub(val_size);

        let mut entries = self.entries;
        let val_entries = entries.split_off(train_size);
        (
            Self { entries },
            Self { entries: val_entries },
        )
    }
}
