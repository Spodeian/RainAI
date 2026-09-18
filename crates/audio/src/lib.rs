//! Real-Time Audio Engine for RainAI.
//!
//! Provides zero-allocation real-time spatial synthesis, procedural DDSP acoustic filterbanks,
//! First-Order Ambisonic (FOA) decoders (binaural HRTF, quadraphonic, 5.1 surround, stereo speakers),
//! continuous hardware-in-the-loop (HWIL) governors, and lossless streaming audio export.
//!
//! # Architecture & Modules
//!
//! - [`engine`]: Real-time audio stream callback driving circular ring buffers, soft clipping limiters,
//!   and sample-accurate dispatch between neural model inference and procedural fallbacks.
//! - [`decoder`]: Ambisonic B-format ($W, Y, Z, X$) decoding matrices projecting 3D spherical soundfields
//!   into stereo headphones (binaural virtual acoustics), stereo monitors, quadraphonic arrays, and 5.1 surround.
//! - [`meta_governor`]: Dynamic runtime hardware governor adjusting quantization tiers (FP32 -> INT16 -> INT8 -> Ternary),
//!   pruning active MoE experts, and triggering emergency procedural cross-fades under buffer underrun stress.
//! - [`procedural`]: High-efficiency procedural rain synthesizer utilizing a 16-band subtractive filterbank
//!   with learned parametric resonant drift profiles for ultra-low-power execution.
//! - [`physical`]: Fluid mechanics and aeroacoustics simulation modules (Ulbrich Gamma DSD, Gunn-Kinzer
//!   terminal velocity aerodynamics, Pumphrey-Crum bubble cavitation, and ISO 140-18 structural plate modes).
//! - [`corruptions`]: Acoustic stress and channel degradation simulation (codec artifacts, packet dropouts,
//!   band-limiting, and thermal noise) for hardware robustness testing.
//! - [`export`]: Chunk-streamed lossless 24-bit/32-bit WAV and raw Ambisonic B-format file rendering.

pub mod corruptions;
pub mod decoder;
pub mod engine;
pub mod export;
pub mod history;
pub mod meta_governor;
pub mod physical;
pub mod procedural;

pub use corruptions::*;
pub use decoder::*;
pub use engine::*;
pub use export::*;
pub use history::*;
pub use meta_governor::*;
pub use physical::*;
pub use procedural::*;

