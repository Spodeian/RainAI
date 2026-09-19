//! Pure Rust Neural Model Training using Hugging Face Candle.
//!
//! Modular architecture:
//! - [`models`]: Neural network architectures (Spatial VAE, Mamba-2 MoE, Latent Attention, Consistency Head)
//! - [`losses`]: Physics-informed trajectory losses, MoE routing losses, Beta-VAE & HWIL penalties
//! - [`dataset`]: Manifest caching, SO(3) Ambisonic rotation, synthetic batch generator, attributions
//! - [`trainer`]: Complete native training loop, atomic checkpoint manager, steering handles, session persistence

pub const LATENT_DIM: usize = 64;
pub const CONDITION_DIM: usize = 554;
pub const COMBINED_DIM: usize = LATENT_DIM + CONDITION_DIM; // 618
pub const NUM_EXPERTS: usize = 8;
pub const FILTER_BANDS: usize = 16;
pub const FOA_CHANNELS: usize = 4;

pub mod dataset;
pub mod losses;
pub mod models;
pub mod trainer;

pub use dataset::*;
pub use losses::*;
pub use models::*;
pub use trainer::*;
