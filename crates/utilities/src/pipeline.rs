//! Strongly-typed in-process pipeline task abstractions for RainAI Studio.
//!
//! Replaces ad-hoc stringly-typed dispatch with unified compile-time task execution.

use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use crate::features::run_features_pipeline;
use crate::golden_vectors::run_golden_vectors_pipeline;
use crate::ingest::run_ingestion_pipeline;
use crate::spatial_upmix::run_upmix_pipeline;
use crate::synth_rain::run_synth_pipeline;

/// Canonical pipeline execution tasks that can run in-process on background threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineTask {
    Ingest,
    Upmix,
    Features,
    GoldenVectors,
    Synth,
}

impl PipelineTask {
    /// Array containing all supported in-process pipeline tasks.
    pub const ALL: [Self; 5] = [
        Self::Ingest,
        Self::Upmix,
        Self::Features,
        Self::GoldenVectors,
        Self::Synth,
    ];

    /// Canonical short identifier matching CLI commands and TUI triggers.
    pub const fn short_code(self) -> &'static str {
        match self {
            Self::Ingest => "ingest",
            Self::Upmix => "upmix",
            Self::Features => "features",
            Self::GoldenVectors => "golden_vectors",
            Self::Synth => "synth",
        }
    }

    /// User-facing descriptive title.
    pub const fn title(self) -> &'static str {
        match self {
            Self::Ingest => "Multi-Source Audio Ingestion",
            Self::Upmix => "Ambisonic FOA Spatial Upmixer",
            Self::Features => "Acoustic Sub-Band Feature Extraction",
            Self::GoldenVectors => "Golden Vector Numerical Verification",
            Self::Synth => "Gunn-Kinzer Physical Raindrop Synthesis",
        }
    }

    /// Parses a short code or descriptive string into a `PipelineTask`.
    pub fn from_code(code: &str) -> Option<Self> {
        let clean = code.trim().to_lowercase();
        match clean.as_str() {
            "ingest" | "in" | "i" => Some(Self::Ingest),
            "upmix" | "up" | "u" => Some(Self::Upmix),
            "features" | "feat" | "f" => Some(Self::Features),
            "golden_vectors" | "golden" | "v" => Some(Self::GoldenVectors),
            "synth" | "s" => Some(Self::Synth),
            _ => None,
        }
    }

    /// Executes the pipeline task synchronously on the calling thread.
    pub fn execute(
        self,
        target_surfaces: Option<&[String]>,
        stop_flag: Arc<AtomicBool>,
        log_tx: Option<Sender<String>>,
    ) -> Result<usize> {
        match self {
            Self::Ingest => run_ingestion_pipeline(stop_flag, log_tx),
            Self::Upmix => run_upmix_pipeline(
                Path::new("Data/rain"),
                Path::new("Data/processed"),
                stop_flag,
                log_tx,
            ),
            Self::Features => run_features_pipeline(
                Path::new("Data/processed"),
                stop_flag,
                log_tx,
            ),
            Self::GoldenVectors => run_golden_vectors_pipeline(log_tx),
            Self::Synth => run_synth_pipeline(
                Path::new("Data/processed"),
                target_surfaces,
                stop_flag,
                log_tx,
            ),
        }
    }
}

impl std::fmt::Display for PipelineTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.title())
    }
}
