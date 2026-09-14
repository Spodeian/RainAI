//! Audio engine for RainAI: real-time synthesis, procedural DDSP, Ambisonic FOA decoders, and export.

pub mod decoder;
pub mod engine;
pub mod export;
pub mod meta_governor;
pub mod procedural;

pub use decoder::*;
pub use engine::*;
pub use export::*;
pub use meta_governor::*;
pub use procedural::*;
