//! RainAI Utilities & Offline Processing Library.
//!
//! This crate provides production-grade physical acoustics simulation, spatial audio upmixing,
//! acoustic feature extraction, multi-source dataset ingestion, golden vector generation,
//! and the master native Candle training engine.
//!
//! # Submodules
//!
//! - [`candle_train`]: Complete native Hugging Face Candle training pipeline implementing
//!   Spatial VAE, Mamba-2 State Space Duality (SSD) Mixture-of-Experts (MoE), iterative latent
//!   thinking blocks, threshold-coverage routing, temporal tabu anti-repetition penalty,
//!   1-step consistency distillation jump heads, and SafeTensors export.
//! - [`stft_loss`]: Multi-resolution STFT loss suite, 3D acoustic active intensity vector loss,
//!   soundfield diffuseness regularization, transient spectral flux half-wave loss, Bark-scale
//!   psychoacoustic weighting, and 3D SO(3) Ambisonic rotation data augmentations.
//! - [`features`]: Pure-Rust acoustic feature extraction pipeline computing 16 sub-band energies,
//!   First-Order Ambisonic spatial directional metrics, and physical conditioning vector embeddings.
//! - [`ingest`]: Multi-threaded, resume-capable dataset downloader and validator targeting
//!   curated CC0/CC-BY/Public Domain audio sources across environmental acoustic repositories.
//! - [`spatial_upmix`]: Stereo-to-B-format spatial upmixer projecting dual-channel hydrophone
//!   and microphone signals into 4-channel First-Order Ambisonics ($W, Y, Z, X$).
//! - [`golden_vectors`]: Automated regression golden vector generator verifying numerical drift
//!   and kernel execution parity across native, ONNX, and WebGPU backends.
//! - [`synth_rain`]: Procedural physical acoustics synthesizer implementing Pumphrey-Crum bubble
//!   cavitation, Minnaert chirps, and ISO 140-18 structural plate modal vibrations.

pub mod audio_preview;
pub mod autopilot;
pub mod candle_train;
pub mod data_worker;
pub mod features;
pub mod golden_vectors;
pub mod ingest;
pub mod spatial_upmix;
pub mod stft_loss;
pub mod synth_rain;
