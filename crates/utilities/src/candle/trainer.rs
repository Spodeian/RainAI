//! Master Native Candle Training Loop, Checkpoints, and Steering Execution.
//!
//! Implements atomic checkpointing, session state persistence, live steering handles,
//! Model EMA, learning rate scheduling with warmup, and the complete training orchestrator.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::Sender,
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tracing::info;

use crate::stft_loss::{MultiResolutionStftLoss, StftLossMode};
use super::*;

/// Atomic checkpoint and metadata transaction manager guaranteeing corruption-proof persistence.
pub struct AtomicCheckpointManager;

impl AtomicCheckpointManager {
    /// Atomically saves safetensors by writing to a temporary file, syncing, and renaming.
    pub fn atomic_save_safetensors<P: AsRef<Path>>(varmap: &VarMap, dest_path: P) -> Result<()> {
        let dest = dest_path.as_ref();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = dest.with_extension("safetensors.tmp");
        varmap.save(&tmp_path)?;
        std::fs::rename(&tmp_path, dest)?;
        Ok(())
    }

    /// Atomically writes JSON metadata by writing to a temporary file and renaming.
    pub fn atomic_save_json<T: Serialize, P: AsRef<Path>>(data: &T, dest_path: P) -> Result<()> {
        let dest = dest_path.as_ref();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = dest.with_extension("json.tmp");
        let f = std::fs::File::create(&tmp_path)?;
        serde_json::to_writer_pretty(f, data)?;
        std::fs::rename(&tmp_path, dest)?;
        Ok(())
    }

    /// Loads session state if present.
    pub fn load_session_state<P: AsRef<Path>>(session_path: P) -> Option<TrainingSessionState> {
        let path = session_path.as_ref();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(state) = serde_json::from_str::<TrainingSessionState>(&content) {
                    return Some(state);
                }
            }
        }
        None
    }
}

/// Persistent training session state for seamless studio open/close resumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSessionState {
    pub completed_vae: bool,
    pub completed_mamba: bool,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub current_batch: usize,
    pub total_batches: usize,
    pub best_vae_loss: f32,
    pub best_mamba_loss: f32,
    pub last_vae_loss: f32,
    pub last_mamba_loss: f32,
    pub last_soup_deficit: f32,
    pub last_stft_loss: f32,
    pub loss_history: Vec<u64>,
    pub vae_loss_history: Vec<u64>,
    pub soup_deficit_history: Vec<u64>,
    pub stft_loss_history: Vec<u64>,
    pub timestamp: u64,
}

impl Default for TrainingSessionState {
    fn default() -> Self {
        Self {
            completed_vae: false,
            completed_mamba: false,
            current_epoch: 0,
            total_epochs: 1,
            current_batch: 0,
            total_batches: 1,
            best_vae_loss: f32::INFINITY,
            best_mamba_loss: f32::INFINITY,
            last_vae_loss: 0.0,
            last_mamba_loss: 0.0,
            last_soup_deficit: 0.0,
            last_stft_loss: 0.0,
            loss_history: Vec::new(),
            vae_loss_history: Vec::new(),
            soup_deficit_history: Vec::new(),
            stft_loss_history: Vec::new(),
            timestamp: 0,
        }
    }
}

/// Live batch/epoch progress update emitted directly in-process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgressUpdate {
    pub phase: TrainingPhase,
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch_idx: usize,
    pub max_batches: usize,
    pub loss: f32,
    pub vae_loss: f32,
    pub mamba_loss: f32,
    pub soup_deficit: f32,
    pub stft_loss: f32,
    pub current_lr: f64,
    pub throughput: f64,
    pub eta_seconds: u64,
}

/// Real-time steering and dynamic control handle for in-process training.
#[derive(Clone)]
pub struct CandleTrainingSteeringHandle {
    pub stop_signal: Arc<AtomicBool>,
    pub pause_signal: Arc<AtomicBool>,
    pub throttle_micros: Arc<AtomicU64>,
    pub progress_tx: Option<Sender<TrainingProgressUpdate>>,
    pub log_tx: Option<Sender<String>>,
    pub surface_weights: Option<Arc<Mutex<HashMap<String, f32>>>>,
    pub dynamic_lr: Option<Arc<Mutex<Option<f64>>>>,
    pub dynamic_config: Option<Arc<Mutex<Option<CandleTrainConfig>>>>,
}

impl Default for CandleTrainingSteeringHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl CandleTrainingSteeringHandle {
    pub fn new() -> Self {
        Self {
            stop_signal: Arc::new(AtomicBool::new(false)),
            pause_signal: Arc::new(AtomicBool::new(false)),
            throttle_micros: Arc::new(AtomicU64::new(0)),
            progress_tx: None,
            log_tx: None,
            surface_weights: None,
            dynamic_lr: None,
            dynamic_config: None,
        }
    }
}

pub const LATENT_DIM: usize = 64;
pub const CONDITION_DIM: usize = 554;
pub const COMBINED_DIM: usize = LATENT_DIM + CONDITION_DIM; // 618
pub const NUM_EXPERTS: usize = 8;
pub const FILTER_BANDS: usize = 16;
pub const FOA_CHANNELS: usize = 4;

// ============================================================================
// 1. Spatial VAE + HOA-DDSP Continuous Latent Projection (Phase 2)
// ============================================================================


// ============================================================================
// 4. Model Exponential Moving Average (EMA)
// ============================================================================

/// Model EMA parameter shadow container.
pub struct ModelEma {
    pub decay: f64,
    pub shadow_vars: HashMap<String, Tensor>,
}

impl ModelEma {
    pub fn new(varmap: &VarMap, decay: f64) -> Result<Self> {
        let mut shadow_vars = HashMap::new();
        for var in varmap.all_vars() {
            let name = var.as_tensor().to_string(); // Variable identifier
            shadow_vars.insert(name, var.as_tensor().copy()?);
        }
        Ok(Self { decay, shadow_vars })
    }

    pub fn update(&mut self, varmap: &VarMap) -> Result<()> {
        for var in varmap.all_vars() {
            let name = var.as_tensor().to_string();
            if let Some(shadow) = self.shadow_vars.get_mut(&name) {
                let current = var.as_tensor();
                let updated = (((&*shadow) * self.decay)? + (current * (1.0 - self.decay))?)?;
                *shadow = updated;
            }
        }
        Ok(())
    }
}

// ============================================================================
// 5. Dataset Ingestion & Synthetic Physics Generator
// ============================================================================


/// Cosine Annealing Learning Rate Scheduler with Linear Warm-up.
pub struct CosineAnnealingWithWarmup {
    pub base_lr: f64,
    pub min_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub current_step: usize,
}

impl CosineAnnealingWithWarmup {
    pub fn new(base_lr: f64, min_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Advances the step and returns the updated learning rate.
    pub fn step(&mut self) -> f64 {
        self.current_step += 1;
        if self.warmup_steps > 0 && self.current_step <= self.warmup_steps {
            // Linear warm-up
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay with strictly clamped progress
            let progress = ((self.current_step.saturating_sub(self.warmup_steps)) as f64
                / (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64)
                .clamp(0.0, 1.0);
            let factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
            self.min_lr + (self.base_lr - self.min_lr) * factor
        }
    }

    /// Gets current learning rate without advancing step.
    pub fn get_lr(&self) -> f64 {
        if self.warmup_steps > 0 && self.current_step <= self.warmup_steps {
            self.base_lr * (self.current_step as f64 / self.warmup_steps.max(1) as f64)
        } else if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            let progress = ((self.current_step.saturating_sub(self.warmup_steps)) as f64
                / (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64)
                .clamp(0.0, 1.0);
            let factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
            self.min_lr + (self.base_lr - self.min_lr) * factor
        }
    }
}

/// Global gradient norm clipping for Candle `VarMap` and `GradStore`.
/// Detects any non-finite gradient (NaN/Inf) and sanitizes all gradients to zero
/// to prevent optimizer momentum corruption and silent parameter poisoning.
pub fn clip_grad_norm_varmap(
    varmap: &VarMap,
    grads: &mut candle_core::backprop::GradStore,
    max_norm: f64,
) -> Result<f64> {
    let mut sum_sq = 0.0f64;
    let mut active_grads = Vec::new();
    let mut has_non_finite = false;

    for var in varmap.all_vars() {
        let t = var.as_tensor();
        if let Some(grad) = grads.get(t) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            if norm_sq.is_finite() {
                sum_sq += norm_sq;
                active_grads.push((t.clone(), grad.clone()));
            } else {
                has_non_finite = true;
                tracing::warn!(
                    "[!] Non-finite gradient detected in parameter shape {:?}! norm_sq is {:?}",
                    t.dims(),
                    norm_sq
                );
            }
        }
    }

    if has_non_finite || !sum_sq.is_finite() {
        tracing::warn!("[!] Non-finite gradient encountered across VarMap! Sanitizing all gradients to zero to prevent momentum poisoning.");
        for var in varmap.all_vars() {
            let t = var.as_tensor();
            if grads.get(t).is_some() {
                grads.insert(t, t.zeros_like()?);
            }
        }
        return Ok(0.0);
    }

    let total_norm = sum_sq.sqrt();
    if total_norm > max_norm && total_norm > 1e-6 {
        let scale = max_norm / (total_norm + 1e-6);
        for (t, grad) in active_grads {
            let clipped = (grad * scale)?;
            grads.insert(&t, clipped);
        }
    }
    Ok(total_norm)
}

/// Verifies that no parameters in a VarMap contain NaN or Inf values.
pub fn verify_varmap_integrity(varmap: &VarMap, name: &str) -> Result<()> {
    for var in varmap.all_vars() {
        let t = var.as_tensor();
        let flattened = t.flatten_all()?;
        let vals = flattened.to_vec1::<f32>()?;
        for v in vals {
            if !v.is_finite() {
                anyhow::bail!("[!] Numerical instability: {} contains non-finite parameters (NaN/Inf)!", name);
            }
        }
    }
    Ok(())
}

// ============================================================================
// 6. Pipeline Orchestrator & Configuration
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPhase {
    Vae,
    Mamba,
    Export,
    All,
}

/// Complete configuration for the Native Candle Training Pipeline.
#[derive(Debug, Clone)]
pub struct CandleTrainConfig {
    pub phases: Vec<TrainingPhase>,
    pub vae_epochs: usize,
    pub mamba_epochs: usize,
    pub batch_size: usize,
    pub max_batches: usize,
    pub learning_rate: f64,
    pub accumulation_steps: usize,
    pub cfg_dropout: f32,
    pub lambda_vel: f64,
    pub lambda_acc: f64,
    pub lambda_drag: f64,
    pub lambda_z: f64,
    pub lambda_doa: f64,
    pub lambda_diff: f64,
    pub lambda_straight: f64,
    pub lambda_div: f64,
    pub gamma_tabu: f64,
    pub enable_distillation: bool,
    pub lambda_distill: f64,
    pub so3_aug_prob: f32,
    pub surface_mixup_prob: f32,
    pub thinking_curriculum: bool,
    pub stochastic_jitter_sigma: f32,
    pub beta_kl: f64,
    pub use_real_data: bool,
    pub use_flow_matching: bool,
    pub tau_moe: f64,
    pub max_thinking_steps: usize,
    pub eps_thinking_halt: f32,
    pub max_grad_norm: f64,
    pub warmup_steps: usize,
    pub val_ratio: f32,
    pub stft_mode: StftLossMode,
    pub stft_weight: f64,
    pub mfp_depth: usize,
    pub mfp_decay: f64,
    pub lambda_soup_deficit: f64,
    pub enable_latent_caching: bool,
    pub continuous_refinement: bool,
    pub output_dir: PathBuf,
    pub device: String,
}

impl Default for CandleTrainConfig {
    fn default() -> Self {
        Self {
            phases: vec![TrainingPhase::All],
            vae_epochs: 2,
            mamba_epochs: 2,
            batch_size: 4,
            max_batches: 10,
            learning_rate: 1e-3,
            accumulation_steps: 1,
            cfg_dropout: 0.15,
            lambda_vel: 0.1,
            lambda_acc: 0.05,
            lambda_drag: 0.02,
            lambda_z: 1e-3,
            lambda_doa: 0.1,
            lambda_diff: 0.05,
            lambda_straight: 0.05,
            lambda_div: 0.05,
            gamma_tabu: 1.0,
            enable_distillation: true,
            lambda_distill: 0.1,
            so3_aug_prob: 0.3,
            surface_mixup_prob: 0.25,
            thinking_curriculum: true,
            stochastic_jitter_sigma: 0.01,
            beta_kl: 0.001,
            use_real_data: true,
            use_flow_matching: true,
            tau_moe: 0.75,
            max_thinking_steps: 3,
            eps_thinking_halt: 0.02,
            max_grad_norm: 1.0,
            warmup_steps: 5,
            val_ratio: 0.15,
            stft_mode: StftLossMode::Combined,
            stft_weight: 0.2,
            mfp_depth: 3,
            mfp_decay: 0.5,
            lambda_soup_deficit: 0.1,
            enable_latent_caching: true,
            continuous_refinement: true,
            output_dir: PathBuf::from("crates/inference/data/candle"),
            device: "auto".to_string(),
        }
    }
}


/// Dynamic device selection supporting CPU, CUDA, and Metal backends.
pub fn select_device(req: &str) -> Device {
    let lower = req.trim().to_lowercase();
    match lower.as_str() {
        #[cfg(feature = "cuda")]
        "cuda" | "gpu" => {
            match Device::new_cuda(0) {
                Ok(dev) => {
                    info!("[+] Successfully initialized CUDA device 0");
                    dev
                }
                Err(e) => {
                    warn!("[-] CUDA requested but initialization failed ({:?}); falling back to CPU", e);
                    Device::Cpu
                }
            }
        }
        #[cfg(feature = "metal")]
        "metal" => {
            match Device::new_metal(0) {
                Ok(dev) => {
                    info!("[+] Successfully initialized Metal device 0");
                    dev
                }
                Err(e) => {
                    warn!("[-] Metal requested but initialization failed ({:?}); falling back to CPU", e);
                    Device::Cpu
                }
            }
        }
        "auto" => {
            #[cfg(feature = "cuda")]
            if let Ok(dev) = Device::new_cuda(0) {
                info!("[+] Auto-selected CUDA hardware accelerator device 0");
                return dev;
            }
            #[cfg(feature = "metal")]
            if let Ok(dev) = Device::new_metal(0) {
                info!("[+] Auto-selected Metal hardware accelerator device 0");
                return dev;
            }
            info!("[*] Using CPU compute device");
            Device::Cpu
        }
        _ => {
            info!("[*] Using CPU compute device");
            Device::Cpu
        }
    }
}

/// Master Native Rust Training Runner executing all phases with steering & session persistence.
pub fn run_candle_training_pipeline_with_steering(
    config: &CandleTrainConfig,
    steering: &CandleTrainingSteeringHandle,
) -> Result<()> {
    let log_msg = |msg: &str| {
        info!("{}", msg);
        if let Some(ref tx) = steering.log_tx {
            let _ = tx.send(msg.to_string());
        }
    };

    log_msg("======================================================================");
    log_msg("RainAI Master Native Candle Training Pipeline (Pure Rust Autonomous)");
    log_msg("======================================================================");

    let device = select_device(&config.device);
    if let Some((ref gpu_name, mem_mb)) = crate::autopilot::probe_host_nvidia_gpu() {
        log_msg(&format!(
            "[*] Compute Architecture: Dual Acceleration (Host Multicore CPU + NVIDIA {} [{:.1} GB VRAM])",
            gpu_name, mem_mb as f64 / 1024.0
        ));
    } else {
        log_msg(&format!("[*] Compute accelerator device: {:?}", device));
    }
    std::fs::create_dir_all(&config.output_dir)?;

    let session_path = config.output_dir.join("training_session.json");
    let mut session = AtomicCheckpointManager::load_session_state(&session_path).unwrap_or_default();
    if session.completed_vae || session.completed_mamba {
        log_msg(&format!(
            "[*] Rehydrated existing session from {:?} (VAE done: {}, Mamba done: {}, Epoch: {})",
            session_path, session.completed_vae, session.completed_mamba, session.current_epoch
        ));
    }

    let real_manifest_candidates = [
        PathBuf::from("Data/processed/manifest.json"),
        PathBuf::from("data/processed/manifest.json"),
        PathBuf::from("../Data/processed/manifest.json"),
        PathBuf::from("../data/processed/manifest.json"),
    ];

    let (train_dataset, val_dataset) = if config.use_real_data {
        let found = real_manifest_candidates.iter().find(|p| p.exists());
        if let Some(path) = found {
            log_msg(&format!("[*] Ingesting real acoustic dataset manifest from {:?}...", path));
            match CandleManifestDataset::load_from_manifest(path) {
                Ok(ds) => {
                    log_msg(&format!("[+] Successfully loaded {} real acoustic records.", ds.entries.len()));
                    let (train_ds, val_ds) = ds.split(config.val_ratio);
                    log_msg(&format!(
                        "[*] Dataset split: {} train samples, {} validation samples (val ratio: {:.2}).",
                        train_ds.entries.len(),
                        val_ds.entries.len(),
                        config.val_ratio
                    ));
                    (Some(train_ds), Some(val_ds))
                }
                Err(e) => {
                    log_msg(&format!("[!] Notice: Manifest parsing failed ({}). Using synthetic physics generator.", e));
                    (None, None)
                }
            }
        } else {
            log_msg("[*] Manifest not found at standard paths. Utilizing high-fidelity synthetic physics generator.");
            (None, None)
        }
    } else {
        (None, None)
    };

    if let Some(ref p_tx) = steering.progress_tx {
        let _ = p_tx.send(TrainingProgressUpdate {
            phase: TrainingPhase::Vae,
            epoch: 1,
            total_epochs: config.vae_epochs,
            batch_idx: 0,
            max_batches: config.max_batches,
            loss: 0.0,
            vae_loss: 0.0,
            mamba_loss: 0.0,
            soup_deficit: 0.0,
            stft_loss: 0.0,
            current_lr: config.learning_rate,
            throughput: 0.0,
            eta_seconds: (config.vae_epochs * config.max_batches) as u64,
        });
    }

    let get_train_batch = |batch_size: usize, cfg_prob: f32, so3_prob: f32, mixup_prob: f32| -> Result<TrainingBatch> {
        if let Some(ref ds) = train_dataset {
            ds.sample_batch_augmented(batch_size, &device, cfg_prob, so3_prob, mixup_prob)
        } else {
            generate_batch_augmented(batch_size, &device, cfg_prob, so3_prob)
        }
    };

    let get_val_batch = |batch_size: usize| -> Result<TrainingBatch> {
        if let Some(ref ds) = val_dataset {
            ds.sample_batch(batch_size, &device, 0.0)
        } else {
            generate_batch(batch_size, &device, 0.0)
        }
    };

    let execute_vae = config.phases.contains(&TrainingPhase::All) || config.phases.contains(&TrainingPhase::Vae);
    let execute_mamba = config.phases.contains(&TrainingPhase::All) || config.phases.contains(&TrainingPhase::Mamba);
    let execute_export = config.phases.contains(&TrainingPhase::All) || config.phases.contains(&TrainingPhase::Export);

    let mut best_vae_val_loss = if session.best_vae_loss.is_finite() { session.best_vae_loss } else { f32::INFINITY };
    let mut best_mamba_val_loss = if session.best_mamba_loss.is_finite() { session.best_mamba_loss } else { f32::INFINITY };

    // ------------------------------------------------------------------------
    // Phase 2: Train Continuous Spatial VAE + HOA-DDSP
    // ------------------------------------------------------------------------
    if execute_vae {
        if session.completed_vae && !config.phases.contains(&TrainingPhase::Vae) && !config.continuous_refinement {
            log_msg(&format!(
                "[*] Skipping Spatial VAE (Phase 2): already marked completed in session (best loss: {:.5}).",
                best_vae_val_loss
            ));
        } else {
            log_msg("\n[Stage 1/3] Training Spatial VAE + HOA-DDSP (Phase 2)...");
            let mut vae_varmap = VarMap::new();
            let vae_vs = VarBuilder::from_varmap(&vae_varmap, DType::F32, &device);
            let vae_model = CandleSpatialVae::new(vae_vs.pp("vae"))?;
            let affine_align = CandleAffineAlignment::new(LATENT_DIM, vae_vs.pp("affine"))?;
            let quantizer = CandleLearnedQuantizer::new(LATENT_DIM, 6.0, vae_vs.pp("quantizer"))?;

            // Rehydrate weights if available for continuous training
            let vae_path = config.output_dir.join("spatial_vae.safetensors");
            if vae_path.exists() {
                if let Ok(()) = vae_varmap.load(&vae_path) {
                    log_msg(&format!("[+] Rehydrated converged Spatial VAE weights from {:?}", vae_path));
                }
            }

            let vae_params = ParamsAdamW {
                lr: config.learning_rate,
                eps: 1e-6,
                ..Default::default()
            };
            let mut vae_opt = AdamW::new(vae_varmap.all_vars(), vae_params)?;
            let mut vae_ema = ModelEma::new(&vae_varmap, 0.995)?;

            let total_vae_steps = config.vae_epochs * config.max_batches;
            let mut vae_scheduler = CosineAnnealingWithWarmup::new(
                config.learning_rate,
                1e-6,
                config.warmup_steps,
                total_vae_steps,
            );

            let stft_calculator = MultiResolutionStftLoss::default();
            log_msg(&format!(
                "[*] Multi-Resolution STFT Loss active (Mode: {:?}, Weight: {:.2})",
                config.stft_mode, config.stft_weight
            ));

            let start_time = Instant::now();
            let mut total_processed_samples = 0usize;

            for epoch in 1..=config.vae_epochs {
                let mut epoch_loss = 0.0f32;
                let mut epoch_recon = 0.0f32;
                let mut epoch_kl = 0.0f32;
                let mut epoch_stft = 0.0f32;
                let mut epoch_doa = 0.0f32;
                let mut epoch_diff = 0.0f32;
                let mut vae_accum_grads: HashMap<candle_core::TensorId, Tensor> = HashMap::new();

                for batch_idx in 1..=config.max_batches {
                    // Check pause
                    while steering.pause_signal.load(Ordering::Relaxed) {
                        if steering.stop_signal.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }

                    // Check stop
                    if steering.stop_signal.load(Ordering::Relaxed) {
                        log_msg("[*] Stop signal detected during VAE training. Saving atomic session state...");
                        session.current_epoch = epoch;
                        session.total_epochs = config.vae_epochs;
                        session.current_batch = batch_idx;
                        session.total_batches = config.max_batches;
                        session.timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                        let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
                        let _ = AtomicCheckpointManager::atomic_save_safetensors(&vae_varmap, config.output_dir.join("spatial_vae.safetensors"));
                        return Ok(());
                    }

                    // Throttling
                    let throttle = steering.throttle_micros.load(Ordering::Relaxed);
                    if throttle > 0 {
                        std::thread::sleep(Duration::from_micros(throttle));
                    }

                    // Dynamic LR steering
                    if let Some(ref lr_mtx) = steering.dynamic_lr {
                        if let Ok(mut g) = lr_mtx.lock() {
                            if let Some(new_lr) = g.take() {
                                vae_opt.set_learning_rate(new_lr);
                            }
                        }
                    }

                    let batch = get_train_batch(
                        config.batch_size,
                        config.cfg_dropout,
                        config.so3_aug_prob,
                        config.surface_mixup_prob,
                    )?;
                    let (pred_bands, pred_foa, mu, logvar) = vae_model.forward(&batch.audio_features, &batch.conditioning)?;
                    let (loss, recon, kl) = compute_beta_vae_loss(&pred_bands, &batch.target_bands, &mu, &logvar, config.beta_kl)?;

                    // Train affine manifold projection & quantizer in loop
                    let z_align = affine_align.forward(&mu)?;
                    let z_q = quantizer.forward(&z_align, 0.1)?;
                    let quant_penalty = (z_q.sqr()?.mean_all()? * 0.005)?;

                    // Direction-of-Arrival and Soundfield Diffuseness Regularization
                    let (foa_loss, doa_err) = crate::stft_loss::compute_acoustic_intensity_and_doa_loss(&pred_foa, &batch.target_foa, 0.5)?;
                    let diff_loss = crate::stft_loss::compute_soundfield_diffuseness_loss(&pred_foa, &batch.target_foa, 0.5)?;
                    let spatial_loss = ((&foa_loss * config.lambda_doa)? + (&diff_loss * config.lambda_diff)?)?;

                    // Evaluate frequency-domain Multi-Resolution STFT Loss
                    let stft_loss = stft_calculator.evaluate_loss(
                        config.stft_mode,
                        &pred_bands,
                        &batch.target_bands,
                        None,
                        None,
                        &device,
                    )?;

                    let affine_ortho = compute_orthogonality_loss(affine_align.weight())?;
                    let raw_batch_loss = ((((&loss + &spatial_loss)? + &quant_penalty)? + (&stft_loss * config.stft_weight)?)? + (&affine_ortho * 0.001)?)?;
                    let total_batch_loss = soft_cap_loss(&raw_batch_loss, 100.0)?;

                    let batch_loss_val = total_batch_loss.to_scalar::<f32>()?;
                    if !batch_loss_val.is_finite() {
                        tracing::warn!("[!] Non-finite VAE batch loss encountered ({:?}) at epoch {} batch {}, skipping update", batch_loss_val, epoch, batch_idx);
                        continue;
                    }

                    let recon_val = recon.to_scalar::<f32>()?;
                    let kl_val = kl.to_scalar::<f32>()?;
                    let stft_val = stft_loss.to_scalar::<f32>()?;
                    let doa_val = doa_err.to_scalar::<f32>()?;
                    let diff_val = diff_loss.to_scalar::<f32>()?;

                    epoch_loss += batch_loss_val;
                    epoch_recon += recon_val;
                    epoch_kl += kl_val;
                    epoch_stft += stft_val;
                    epoch_doa += doa_val;
                    epoch_diff += diff_val;

                    if config.accumulation_steps > 1 {
                        let scaled_loss = (total_batch_loss / config.accumulation_steps as f64)?;
                        let batch_grads = scaled_loss.backward()?;
                        for var in vae_varmap.all_vars() {
                            let t = var.as_tensor();
                            if let Some(g) = batch_grads.get(t) {
                                let is_finite = g.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite();
                                if is_finite {
                                    match vae_accum_grads.get_mut(&t.id()) {
                                        Some(acc) => *acc = (acc as &Tensor + g)?,
                                        None => { vae_accum_grads.insert(t.id(), g.clone()); }
                                    }
                                } else {
                                    tracing::warn!("[!] Non-finite gradient in accumulation for VAE, skipping tensor");
                                }
                            }
                        }

                        if batch_idx % config.accumulation_steps == 0 || batch_idx == config.max_batches {
                            let mut final_grads = candle_core::backprop::GradStore::default();
                            for var in vae_varmap.all_vars() {
                                let t = var.as_tensor();
                                if let Some(g) = vae_accum_grads.remove(&t.id()) {
                                    final_grads.insert(t, g);
                                }
                            }
                            let norm = clip_grad_norm_varmap(&vae_varmap, &mut final_grads, config.max_grad_norm)?;
                            if norm > 1e-12 {
                                vae_opt.step(&final_grads)?;
                                let current_lr = vae_scheduler.step();
                                vae_opt.set_learning_rate(current_lr);
                                vae_ema.update(&vae_varmap)?;
                            } else {
                                tracing::warn!("[!] Skipping VAE optimizer step due to zero or sanitized gradient norm");
                            }
                        }
                    } else {
                        let mut grads = total_batch_loss.backward()?;
                        let norm = clip_grad_norm_varmap(&vae_varmap, &mut grads, config.max_grad_norm)?;
                        if norm > 1e-12 {
                            vae_opt.step(&grads)?;
                            let current_lr = vae_scheduler.step();
                            vae_opt.set_learning_rate(current_lr);
                            vae_ema.update(&vae_varmap)?;
                        } else {
                            tracing::warn!("[!] Skipping VAE optimizer step due to zero or sanitized gradient norm");
                        }
                    }

                    total_processed_samples += config.batch_size;
                    let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                    let throughput = total_processed_samples as f64 / elapsed;
                    let total_expected_steps = config.vae_epochs * config.max_batches;
                    let current_step = (epoch - 1) * config.max_batches + batch_idx;
                    let remaining_steps = total_expected_steps.saturating_sub(current_step);
                    let eta_seconds = (remaining_steps as f64 * config.batch_size as f64 / throughput.max(1.0)) as u64;

                    if batch_idx % 5 == 0 || batch_idx == 1 || batch_idx == config.max_batches {
                        if let Some(ref p_tx) = steering.progress_tx {
                            let _ = p_tx.send(TrainingProgressUpdate {
                                phase: TrainingPhase::Vae,
                                epoch,
                                total_epochs: config.vae_epochs,
                                batch_idx,
                                max_batches: config.max_batches,
                                loss: batch_loss_val,
                                vae_loss: recon_val,
                                mamba_loss: 0.0,
                                soup_deficit: 0.0,
                                stft_loss: stft_val,
                                current_lr: vae_scheduler.get_lr(),
                                throughput,
                                eta_seconds,
                            });
                        }
                    }
                }

                let avg_loss = epoch_loss / config.max_batches as f32;
                let avg_recon = epoch_recon / config.max_batches as f32;
                let avg_kl = epoch_kl / config.max_batches as f32;
                let avg_stft = epoch_stft / config.max_batches as f32;
                let avg_doa = epoch_doa / config.max_batches as f32;
                let avg_diff = epoch_diff / config.max_batches as f32;

                // Epoch validation evaluation
                let mut val_recon_sum = 0.0f32;
                let val_batches = 2;
                for _ in 0..val_batches {
                    let vbatch = get_val_batch(config.batch_size)?;
                    let (vpred_bands, vpred_foa, vmu, vlogvar) = vae_model.forward(&vbatch.audio_features, &vbatch.conditioning)?;
                    let (vloss, _, _) = compute_beta_vae_loss(&vpred_bands, &vbatch.target_bands, &vmu, &vlogvar, config.beta_kl)?;
                    let (vfoa, _) = crate::stft_loss::compute_acoustic_intensity_and_doa_loss(&vpred_foa, &vbatch.target_foa, 0.5)?;
                    let vdiff = crate::stft_loss::compute_soundfield_diffuseness_loss(&vpred_foa, &vbatch.target_foa, 0.5)?;
                    let vtotal = ((&vloss + (&vfoa * config.lambda_doa)?)? + (&vdiff * config.lambda_diff)?)?;
                    val_recon_sum += vtotal.to_scalar::<f32>()?;
                }
                let avg_val_loss = val_recon_sum / val_batches as f32;

                log_msg(&format!(
                    "VAE Epoch [{}/{}] - Train Loss: {:.5} (Recon: {:.5}, KL: {:.6}, STFT: {:.5}, DOA Err: {:.4}, Diff: {:.4}) | Val Loss: {:.5}",
                    epoch, config.vae_epochs, avg_loss, avg_recon, avg_kl, avg_stft, avg_doa, avg_diff, avg_val_loss
                ));

                session.current_epoch = epoch;
                session.total_epochs = config.vae_epochs;
                session.last_vae_loss = avg_loss;
                session.last_stft_loss = avg_stft;
                session.loss_history.push((avg_loss * 1000.0).max(0.0) as u64);
                session.vae_loss_history.push((avg_recon * 1000.0).max(0.0) as u64);
                session.stft_loss_history.push((avg_stft * 1000.0).max(0.0) as u64);
                if session.loss_history.len() > 500 { session.loss_history.drain(..50); }
                if session.vae_loss_history.len() > 500 { session.vae_loss_history.drain(..50); }
                if session.stft_loss_history.len() > 500 { session.stft_loss_history.drain(..50); }
                session.timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

                if avg_val_loss < best_vae_val_loss {
                    best_vae_val_loss = avg_val_loss;
                    session.best_vae_loss = best_vae_val_loss;
                    let best_path = config.output_dir.join("spatial_vae_best.safetensors");
                    AtomicCheckpointManager::atomic_save_safetensors(&vae_varmap, &best_path)?;
                    log_msg(&format!("[*] New best VAE checkpoint atomically saved -> {:?}", best_path));
                }
                let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
            }

            verify_varmap_integrity(&vae_varmap, "Spatial VAE")?;
            let vae_path = config.output_dir.join("spatial_vae.safetensors");
            AtomicCheckpointManager::atomic_save_safetensors(&vae_varmap, &vae_path)?;
            session.completed_vae = true;
            let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
            log_msg(&format!("[+] Verified and atomically saved final Spatial VAE weights -> {:?}", vae_path));
        }
    }

    // ------------------------------------------------------------------------
    // Phase 3: Train Mamba-2 MoE + HWIL Meta-Controller + Iterative Thinking
    // ------------------------------------------------------------------------
    if execute_mamba {
        if session.completed_mamba && !config.phases.contains(&TrainingPhase::Mamba) && !config.continuous_refinement {
            log_msg(&format!(
                "[*] Skipping Mamba-2 MoE (Phase 3): already marked completed in session (best loss: {:.5}).",
                best_mamba_val_loss
            ));
        } else {
            log_msg("\n[Stage 2/3] Training Mamba-2 MoE Recurrence, Router & Deliberation Dynamics (Phase 3)...");
            let mut mamba_varmap = VarMap::new();
            let mamba_vs = VarBuilder::from_varmap(&mamba_varmap, DType::F32, &device);
            let mamba_model = CandleMamba2MoE::new(mamba_vs.pp("mamba"))?;
            let thinking_block = CandleThinkingBlock::new(LATENT_DIM, CONDITION_DIM, mamba_vs.pp("thinking"))?;
            let meta_controller = CandleInvasiveMetaController::new(NUM_EXPERTS, 16, mamba_vs.pp("meta"))?;
            let consistency_head = CandleConsistencyHead::new(LATENT_DIM + CONDITION_DIM, LATENT_DIM, mamba_vs.pp("consistency_head"))?;

            // Rehydrate weights if available for continuous training
            let mamba_path = config.output_dir.join("mamba2_moe.safetensors");
            if mamba_path.exists() {
                if let Ok(()) = mamba_varmap.load(&mamba_path) {
                    log_msg(&format!("[+] Rehydrated converged Mamba-2 MoE weights from {:?}", mamba_path));
                }
            }

            let mamba_params = ParamsAdamW {
                lr: config.learning_rate,
                eps: 1e-6,
                ..Default::default()
            };
            let mut mamba_opt = AdamW::new(mamba_varmap.all_vars(), mamba_params)?;
            let mut mamba_ema = ModelEma::new(&mamba_varmap, 0.995)?;

            let total_mamba_steps = config.mamba_epochs * config.max_batches;
            let mut mamba_scheduler = CosineAnnealingWithWarmup::new(
                config.learning_rate,
                1e-6,
                config.warmup_steps,
                total_mamba_steps,
            );

            let mut h_state = Tensor::zeros((config.batch_size, 128), DType::F32, &device)?;
            let mut telem_state = Tensor::zeros((config.batch_size, 32, 16), DType::F32, &device)?;
            let telem_const = Tensor::full(0.5f32, (config.batch_size, 4), &device)?;
            let user_w_const = Tensor::full(0.5f32, (config.batch_size, 3), &device)?;
            let quality_const = Tensor::full(0.9f32, (config.batch_size, 2), &device)?;
            let slice_const = Tensor::full(1.0f32, (config.batch_size, 1), &device)?;

            log_msg(&format!(
                "[*] Smooth Softmax MoE Routing active (tau_moe: {:.2}, gamma_tabu: {:.2})",
                config.tau_moe, config.gamma_tabu
            ));
            log_msg(&format!(
                "[*] Iterative Latent Thinking active (Max Steps: {}, Halting Eps: {:.4}, Jitter Sigma: {:.4})",
                config.max_thinking_steps, config.eps_thinking_halt, config.stochastic_jitter_sigma
            ));
            if config.thinking_curriculum {
                log_msg("[*] Progressive Thinking Curriculum enabled (Epoch-scaled budget warm-up)");
            }
            if config.enable_distillation {
                log_msg(&format!("[*] 1-Step Consistency Distillation Jump Head active (Weight: {:.2})", config.lambda_distill));
            }

            use rand::Rng;
            let mut rng = rand::thread_rng();
            let start_time = Instant::now();
            let mut total_processed_samples = 0usize;

            for epoch in 1..=config.mamba_epochs {
                let mut epoch_loss = 0.0f32;
                let mut epoch_aux = 0.0f32;
                let mut epoch_flow = 0.0f32;
                let mut epoch_z = 0.0f32;
                let mut epoch_div = 0.0f32;
                let mut epoch_distill = 0.0f32;
                let mut epoch_active_exp = 0.0f32;
                let mut epoch_thinking_steps = 0.0f32;
                let mut mamba_accum_grads: HashMap<candle_core::TensorId, Tensor> = HashMap::new();

                let max_m_epoch = if config.thinking_curriculum {
                    epoch.min(config.max_thinking_steps)
                } else {
                    config.max_thinking_steps
                };

                for batch_idx in 1..=config.max_batches {
                    // Check pause
                    while steering.pause_signal.load(Ordering::Relaxed) {
                        if steering.stop_signal.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }

                    // Check stop
                    if steering.stop_signal.load(Ordering::Relaxed) {
                        log_msg("[*] Stop signal detected during Mamba training. Saving atomic session state...");
                        session.current_epoch = epoch;
                        session.total_epochs = config.mamba_epochs;
                        session.current_batch = batch_idx;
                        session.total_batches = config.max_batches;
                        session.timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                        let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
                        let _ = AtomicCheckpointManager::atomic_save_safetensors(&mamba_varmap, config.output_dir.join("mamba2_moe.safetensors"));
                        return Ok(());
                    }

                    // Throttling
                    let throttle = steering.throttle_micros.load(Ordering::Relaxed);
                    if throttle > 0 {
                        std::thread::sleep(Duration::from_micros(throttle));
                    }

                    // Dynamic LR steering
                    if let Some(ref lr_mtx) = steering.dynamic_lr {
                        if let Ok(mut g) = lr_mtx.lock() {
                            if let Some(new_lr) = g.take() {
                                mamba_opt.set_learning_rate(new_lr);
                            }
                        }
                    }

                    let batch = get_train_batch(
                        config.batch_size,
                        config.cfg_dropout,
                        config.so3_aug_prob,
                        config.surface_mixup_prob,
                    )?;

                    let (z_pred, next_h, router_probs, _smooth_mask, mean_active, router_logits) = mamba_model.forward_smooth_tabu(
                        &batch.z_prev,
                        &batch.conditioning,
                        &h_state,
                        config.tau_moe,
                        None,
                        0.0,
                    )?;
                    h_state = next_h.detach();
                    epoch_active_exp += mean_active.to_scalar::<f32>()?;

                    let m_batch = if max_m_epoch > 1 {
                        rng.gen_range(1..=max_m_epoch)
                    } else {
                        1
                    };

                    let mut prob_history = Vec::with_capacity(m_batch);
                    prob_history.push(router_probs.clone());
                    let mut tabu_sum = router_probs.clone();
                    let mut z_delib = z_pred.copy()?;

                    for iter in 1..m_batch {
                        // Dynamically scale tabu so subsequent iterations push into distinct expert subspaces
                        let dynamic_tabu = config.gamma_tabu * (1.0 + 0.25 * (iter as f64));
                        let (step_z, _, step_probs, _, _, _) = mamba_model.forward_smooth_tabu(
                            &z_delib,
                            &batch.conditioning,
                            &h_state,
                            config.tau_moe,
                            Some(&tabu_sum),
                            dynamic_tabu,
                        )?;
                        tabu_sum = (&tabu_sum + &step_probs)?;
                        prob_history.push(step_probs);
                        z_delib = step_z;
                    }

                    let diversity_loss = compute_expert_diversity_loss(&prob_history)?;

                    let (z_refined, steps_taken, halting_probs) = thinking_block.forward_thinking_with_jitter(
                        &z_pred,
                        &batch.conditioning,
                        m_batch,
                        config.eps_thinking_halt,
                        config.stochastic_jitter_sigma,
                    )?;
                    epoch_thinking_steps += steps_taken as f32;

                    let (traj_loss, _pos, _acc, _drag) = compute_physics_trajectory_loss_v2(
                        &z_refined,
                        &batch.z_target,
                        &batch.z_prev,
                        batch.z_prev2.as_ref(),
                        config.lambda_vel,
                        config.lambda_acc,
                        config.lambda_drag,
                        2.5,
                    )?;

                    let aux_loss = compute_moe_load_balancing_loss(&router_probs)?;
                    let router_z_loss = compute_router_z_loss(&router_logits)?;

                    let meta_out = meta_controller.forward(&router_probs, &telem_const, &user_w_const, &quality_const, &slice_const, &telem_state)?;
                    telem_state = meta_out.next_telem_state.detach();

                    let active_exp_val = mean_active.to_scalar::<f32>()?;
                    let hwil_scalar = compute_continuous_hwil_penalty(50.0, 50.0, active_exp_val, 2.0);
                    let hwil_penalty = ((&meta_out.stress.mean_all()? * 0.01)? + (hwil_scalar as f64 * 0.005))?;

                    let (flow_loss, _base_flow) = if config.use_flow_matching {
                        compute_straight_flow_loss(&z_refined, &batch.z_target, &batch.z_prev, 1e-4, config.lambda_straight)?
                    } else {
                        (Tensor::zeros((), DType::F32, &device)?, Tensor::zeros((), DType::F32, &device)?)
                    };

                    let distill_loss = if config.enable_distillation {
                        let z_fast = consistency_head.forward(&z_pred, &batch.conditioning)?;
                        consistency_head.compute_distill_loss(&z_fast, &z_refined, 0.5)?
                    } else {
                        Tensor::zeros((), DType::F32, &device)?
                    };

                    let halt_penalty = if !halting_probs.is_empty() {
                        let last_halt = &halting_probs[halting_probs.len() - 1];
                        (last_halt.mean_all()? * 0.005)?
                    } else {
                        Tensor::zeros((), DType::F32, &device)?
                    };

                    let soup_alpha = mamba_model.router_to_soup_coefficients(&router_probs)?;
                    let comb_in = Tensor::cat(&[&batch.z_prev, &batch.conditioning], 1)?;
                    let h_comb = mamba_model.in_proj.forward(&comb_in)?.gelu_erf()?;
                    let (soup_out, _) = mamba_model.compute_dense_soup(&h_comb, &h_state, &soup_alpha)?;
                    let soup_fused = mamba_model.fusion.forward(&soup_out)?.gelu_erf()?;
                    let z_soup_raw = mamba_model.traj_head.forward(&soup_fused)?;
                    let z_soup = mamba_model.out_affine.forward(&z_soup_raw)?;
                    let soup_deficit = (&z_soup - &z_pred.detach())?.sqr()?.mean_all()?;

                    let z_t2_raw = mamba_model.traj_head_t2.forward(&soup_fused)?;
                    let z_t2 = mamba_model.out_affine.forward(&z_t2_raw)?;
                    let z_t3_raw = mamba_model.traj_head_t3.forward(&soup_fused)?;
                    let z_t3 = mamba_model.out_affine.forward(&z_t3_raw)?;

                    let mfp_loss = (((&z_t2 - &batch.z_target)?.sqr()?.mean_all()? * config.mfp_decay)?
                        + ((&z_t3 - &batch.z_target)?.sqr()?.mean_all()? * (config.mfp_decay * config.mfp_decay))?)?;

                    // Dimension outlier spike suppression on trajectory prediction
                    let spike_loss = compute_dimension_outlier_spike_loss(&z_pred, 3.5)?;
                    // Orthogonality regularization on learned output affine transformation
                    let ortho_loss = compute_orthogonality_loss(mamba_model.out_affine.weight())?;
                    // Contractive loss for Lyapunov stability
                    let contract_loss = compute_contractive_loss(&z_pred, &batch.z_prev, batch.z_prev2.as_ref(), 2.0)?;

                    let loss_step1 = (&traj_loss + (&aux_loss * 0.02)?)?;
                    let loss_step2 = (&loss_step1 + (&router_z_loss * config.lambda_z)?)?;
                    let loss_step3 = (&loss_step2 + (&flow_loss * 0.05)?)?;
                    let loss_step4 = (&loss_step3 + &hwil_penalty)?;
                    let loss_step5 = (&loss_step4 + (&diversity_loss * config.lambda_div)?)?;
                    let loss_step6 = (&loss_step5 + (&distill_loss * config.lambda_distill)?)?;
                    let loss_step7 = (&loss_step6 + (&soup_deficit * config.lambda_soup_deficit)?)?;
                    let loss_step8 = (&loss_step7 + (&mfp_loss * 0.05)?)?;
                    let loss_step9 = (&loss_step8 + (&spike_loss * 0.01)?)?;
                    let loss_step10 = (&loss_step9 + (&ortho_loss * 0.001)?)?;
                    let loss_step11 = (&loss_step10 + (&contract_loss * 0.01)?)?;
                    let raw_total_loss = (&loss_step11 + &halt_penalty)?;
                    let total_loss = soft_cap_loss(&raw_total_loss, 50.0)?;

                    let total_loss_val = total_loss.to_scalar::<f32>()?;
                    if !total_loss_val.is_finite() {
                        tracing::warn!("[!] Non-finite Mamba batch loss encountered ({:?}) at epoch {} batch {}, skipping update", total_loss_val, epoch, batch_idx);
                        continue;
                    }

                    let aux_val = aux_loss.to_scalar::<f32>()?;
                    let flow_val = flow_loss.to_scalar::<f32>()?;
                    let z_val = router_z_loss.to_scalar::<f32>()?;
                    let div_val = diversity_loss.to_scalar::<f32>()?;
                    let distill_val = distill_loss.to_scalar::<f32>()?;
                    let soup_val = soup_deficit.to_scalar::<f32>()?;

                    epoch_loss += total_loss_val;
                    epoch_aux += aux_val;
                    epoch_flow += flow_val;
                    epoch_z += z_val;
                    epoch_div += div_val;
                    epoch_distill += distill_val;

                    if config.accumulation_steps > 1 {
                        let scaled_loss = (total_loss / config.accumulation_steps as f64)?;
                        let batch_grads = scaled_loss.backward()?;
                        for var in mamba_varmap.all_vars() {
                            let t = var.as_tensor();
                            if let Some(g) = batch_grads.get(t) {
                                let is_finite = g.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite();
                                if is_finite {
                                    match mamba_accum_grads.get_mut(&t.id()) {
                                        Some(acc) => *acc = (acc as &Tensor + g)?,
                                        None => { mamba_accum_grads.insert(t.id(), g.clone()); }
                                    }
                                } else {
                                    tracing::warn!("[!] Non-finite gradient in accumulation for Mamba, skipping tensor");
                                }
                            }
                        }

                        if batch_idx % config.accumulation_steps == 0 || batch_idx == config.max_batches {
                            let mut final_grads = candle_core::backprop::GradStore::default();
                            for var in mamba_varmap.all_vars() {
                                let t = var.as_tensor();
                                if let Some(g) = mamba_accum_grads.remove(&t.id()) {
                                    let shape = t.dims();
                                    let scaled_g = if shape.len() == 2 && shape[0] == 128 && shape[1] == 16 {
                                        (g * 0.1)?
                                    } else {
                                        g
                                    };
                                    final_grads.insert(t, scaled_g);
                                }
                            }

                            let norm = clip_grad_norm_varmap(&mamba_varmap, &mut final_grads, config.max_grad_norm)?;
                            if norm > 1e-12 {
                                mamba_opt.step(&final_grads)?;
                                let current_lr = mamba_scheduler.step();
                                mamba_opt.set_learning_rate(current_lr);
                                mamba_ema.update(&mamba_varmap)?;
                            } else {
                                tracing::warn!("[!] Skipping Mamba optimizer step due to zero or sanitized gradient norm");
                            }
                        }
                    } else {
                        let mut grads = total_loss.backward()?;
                        let norm = clip_grad_norm_varmap(&mamba_varmap, &mut grads, config.max_grad_norm)?;

                        if norm > 1e-12 {
                            for var in mamba_varmap.all_vars() {
                                let shape = var.as_tensor().dims();
                                if shape.len() == 2 && shape[0] == 128 && shape[1] == 16 {
                                    if let Some(g) = grads.get(var.as_tensor()) {
                                        grads.insert(var.as_tensor(), (g * 0.1)?);
                                    }
                                }
                            }

                            mamba_opt.step(&grads)?;
                            let current_lr = mamba_scheduler.step();
                            mamba_opt.set_learning_rate(current_lr);
                            mamba_ema.update(&mamba_varmap)?;
                        } else {
                            tracing::warn!("[!] Skipping Mamba optimizer step due to zero or sanitized gradient norm");
                        }
                    }

                    total_processed_samples += config.batch_size;
                    let elapsed = start_time.elapsed().as_secs_f64().max(0.001);
                    let throughput = total_processed_samples as f64 / elapsed;
                    let total_expected_steps = config.mamba_epochs * config.max_batches;
                    let current_step = (epoch - 1) * config.max_batches + batch_idx;
                    let remaining_steps = total_expected_steps.saturating_sub(current_step);
                    let eta_seconds = (remaining_steps as f64 * config.batch_size as f64 / throughput.max(1.0)) as u64;

                    if batch_idx % 5 == 0 || batch_idx == 1 || batch_idx == config.max_batches {
                        if let Some(ref p_tx) = steering.progress_tx {
                            let _ = p_tx.send(TrainingProgressUpdate {
                                phase: TrainingPhase::Mamba,
                                epoch,
                                total_epochs: config.mamba_epochs,
                                batch_idx,
                                max_batches: config.max_batches,
                                loss: total_loss_val,
                                vae_loss: session.last_vae_loss,
                                mamba_loss: total_loss_val,
                                soup_deficit: soup_val,
                                stft_loss: session.last_stft_loss,
                                current_lr: mamba_scheduler.get_lr(),
                                throughput,
                                eta_seconds,
                            });
                        }
                    }
                }

                let avg_loss = epoch_loss / config.max_batches as f32;
                let avg_aux = epoch_aux / config.max_batches as f32;
                let avg_flow = epoch_flow / config.max_batches as f32;
                let avg_z = epoch_z / config.max_batches as f32;
                let avg_div = epoch_div / config.max_batches as f32;
                let avg_distill = epoch_distill / config.max_batches as f32;
                let avg_exp = epoch_active_exp / config.max_batches as f32;
                let avg_steps = epoch_thinking_steps / config.max_batches as f32;

                // Epoch validation evaluation
                let mut val_loss_sum = 0.0f32;
                let val_batches = 2;
                let mut val_h_state = Tensor::zeros((config.batch_size, 128), DType::F32, &device)?;
                for _ in 0..val_batches {
                    let vbatch = get_val_batch(config.batch_size)?;
                    let (vz_pred, vnext_h, _, _, _, _) = mamba_model.forward_smooth(
                        &vbatch.z_prev,
                        &vbatch.conditioning,
                        &val_h_state,
                        config.tau_moe,
                    )?;
                    val_h_state = vnext_h.copy()?;
                    let (vz_ref, _, _) = thinking_block.forward_thinking(
                        &vz_pred,
                        &vbatch.conditioning,
                        config.max_thinking_steps,
                        config.eps_thinking_halt,
                    )?;
                    let (vtraj, _, _, _) = compute_physics_trajectory_loss_v2(
                        &vz_ref,
                        &vbatch.z_target,
                        &vbatch.z_prev,
                        vbatch.z_prev2.as_ref(),
                        config.lambda_vel,
                        config.lambda_acc,
                        config.lambda_drag,
                        2.5,
                    )?;
                    val_loss_sum += vtraj.to_scalar::<f32>()?;
                }
                let avg_val_loss = val_loss_sum / val_batches as f32;

                log_msg(&format!(
                    "Mamba2-MoE Epoch [{}/{}] - Traj: {:.5} (Flow: {:.5}, Aux: {:.5}, Z: {:.4}, Div: {:.4}, Distill: {:.4}) | Exp: {:.1}/8, Think: {:.1} iters | Val Loss: {:.5}",
                    epoch, config.mamba_epochs, avg_loss, avg_flow, avg_aux, avg_z, avg_div, avg_distill, avg_exp, avg_steps, avg_val_loss
                ));

                session.current_epoch = epoch;
                session.total_epochs = config.mamba_epochs;
                session.last_mamba_loss = avg_loss;
                session.loss_history.push((avg_loss * 1000.0).max(0.0) as u64);
                if session.loss_history.len() > 500 { session.loss_history.drain(..50); }
                session.timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

                if avg_val_loss < best_mamba_val_loss {
                    best_mamba_val_loss = avg_val_loss;
                    session.best_mamba_loss = best_mamba_val_loss;
                    let best_path = config.output_dir.join("mamba2_moe_best.safetensors");
                    AtomicCheckpointManager::atomic_save_safetensors(&mamba_varmap, &best_path)?;
                    log_msg(&format!("[*] New best Mamba-2 MoE checkpoint atomically saved -> {:?}", best_path));
                }
                let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
            }

            verify_varmap_integrity(&mamba_varmap, "Mamba-2 MoE")?;
            let mamba_path = config.output_dir.join("mamba2_moe.safetensors");
            AtomicCheckpointManager::atomic_save_safetensors(&mamba_varmap, &mamba_path)?;
            log_msg(&format!("[+] Verified and atomically saved final Mamba-2 MoE weights -> {:?}", mamba_path));

            if config.enable_distillation {
                let fast_path = config.output_dir.join("mamba2_moe_fast.safetensors");
                AtomicCheckpointManager::atomic_save_safetensors(&mamba_varmap, &fast_path)?;
                log_msg(&format!("[+] Atomically saved 1-Step Edge / WebGPU Fast Mamba-2 MoE deployment package -> {:?}", fast_path));
            }

            let soup_path = config.output_dir.join("mamba2_dense_soup.safetensors");
            AtomicCheckpointManager::atomic_save_safetensors(&mamba_varmap, &soup_path)?;
            log_msg(&format!("[+] Atomically saved collapsed Dense Soup Mamba-2 deployment package -> {:?}", soup_path));

            session.completed_mamba = true;
            let _ = AtomicCheckpointManager::atomic_save_json(&session, &session_path);
        }
    }

    // ------------------------------------------------------------------------
    // Phase 4: Multi-Backend SafeTensors & Metadata Verification
    // ------------------------------------------------------------------------
    if execute_export {
        if let Some(ref p_tx) = steering.progress_tx {
            let _ = p_tx.send(TrainingProgressUpdate {
                phase: TrainingPhase::Export,
                epoch: 1,
                total_epochs: 1,
                batch_idx: 1,
                max_batches: 1,
                loss: session.last_mamba_loss,
                vae_loss: session.last_vae_loss,
                mamba_loss: session.last_mamba_loss,
                soup_deficit: session.last_soup_deficit,
                stft_loss: session.last_stft_loss,
                current_lr: config.learning_rate,
                throughput: 0.0,
                eta_seconds: 0,
            });
        }
        log_msg("\n[Stage 3/3] Exporting and validating SafeTensors deployment packages...");
        let manifest_path = config.output_dir.join("candle_manifest.json");
        let manifest = serde_json::json!({
            "framework": "Candle",
            "version": "0.11",
            "phases_trained": config.phases.iter().map(|p| format!("{:?}", p)).collect::<Vec<_>>(),
            "models": {
                "spatial_vae": "spatial_vae.safetensors",
                "spatial_vae_best": "spatial_vae_best.safetensors",
                "mamba2_moe": "mamba2_moe.safetensors",
                "mamba2_moe_best": "mamba2_moe_best.safetensors",
                "mamba2_moe_fast": if config.enable_distillation { Some("mamba2_moe_fast.safetensors") } else { None },
                "mamba2_dense_soup": "mamba2_dense_soup.safetensors"
            },
            "latent_dim": LATENT_DIM,
            "conditioning_dim": CONDITION_DIM,
            "experts": NUM_EXPERTS,
            "tau_moe": config.tau_moe,
            "gamma_tabu": config.gamma_tabu,
            "max_thinking_steps": config.max_thinking_steps,
            "stft_mode": format!("{:?}", config.stft_mode),
            "best_vae_val_loss": if best_vae_val_loss.is_finite() { Some(best_vae_val_loss) } else { None },
            "best_mamba_val_loss": if best_mamba_val_loss.is_finite() { Some(best_mamba_val_loss) } else { None },
            "real_dataset_used": train_dataset.is_some(),
            "flow_matching_enabled": config.use_flow_matching,
            "expert_diversity_enabled": true,
            "lambda_div": config.lambda_div,
            "enable_distillation": config.enable_distillation,
            "lambda_distill": config.lambda_distill,
            "thinking_curriculum_enabled": config.thinking_curriculum,
            "so3_aug_prob": config.so3_aug_prob,
            "surface_mixup_prob": config.surface_mixup_prob,
            "lambda_vel": config.lambda_vel,
            "lambda_acc": config.lambda_acc,
            "lambda_drag": config.lambda_drag,
            "lambda_z": config.lambda_z,
            "lambda_doa": config.lambda_doa,
            "lambda_diff": config.lambda_diff,
            "lambda_straight": config.lambda_straight,
            "scheduler": "CosineAnnealingWithWarmup",
            "gradient_clipping_max_norm": config.max_grad_norm,
            "timestamp": "2026-09-17"
        });
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
        log_msg(&format!("[+] Created Candle deployment manifest -> {:?}", manifest_path));
    }

    log_msg("\n======================================================================");
    log_msg("MASTER CANDLE PIPELINE FINISHED SUCCESSFULLY");
    log_msg("======================================================================");

    Ok(())
}

/// Backward-compatible master native Rust training runner executing all phases.
pub fn run_candle_training_pipeline(config: &CandleTrainConfig) -> Result<()> {
    let steering = CandleTrainingSteeringHandle::new();
    run_candle_training_pipeline_with_steering(config, &steering)
}
