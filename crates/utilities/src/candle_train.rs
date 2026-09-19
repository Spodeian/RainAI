//! Backward-compatible facade re-exporting [`crate::candle`].
//!
//! The monolithic training pipeline has been decomposed into modular submodules:
//! - [`crate::candle::models`]: Neural network architectures
//! - [`crate::candle::losses`]: Physics-informed trajectory and MoE routing losses
//! - [`crate::candle::dataset`]: Manifest dataset, SO(3) augmentations, attributions
//! - [`crate::candle::trainer`]: Native Candle training loop, checkpoints, and steering

pub use crate::candle::*;
