//! Neural Inference Runtime for continuous mixed-precision models.

pub mod asset_manager;
pub mod model;
pub mod runner;

pub use asset_manager::*;
pub use model::*;
pub use runner::*;
