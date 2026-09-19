//! Reusable workspace path resolution and file discovery utilities.
//!
//! Standardizes case-sensitivity and relative path search across root and crate-local executions.

use std::path::{Path, PathBuf};

pub struct WorkspacePaths;

impl WorkspacePaths {
    /// Resolves the first existing path among the provided candidates.
    pub fn resolve_existing<P: AsRef<Path>>(candidates: &[P]) -> Option<PathBuf> {
        candidates
            .iter()
            .map(|p| p.as_ref())
            .find(|p| p.exists())
            .map(PathBuf::from)
    }

    /// Standard candidate paths for the processed dataset manifest (`manifest.json`).
    pub fn manifest_candidates() -> &'static [&'static str] {
        &[
            "Data/processed/manifest.json",
            "data/processed/manifest.json",
            "../Data/processed/manifest.json",
            "../data/processed/manifest.json",
            "../../Data/processed/manifest.json",
            "../../data/processed/manifest.json",
        ]
    }

    /// Resolves the processed dataset manifest if it exists on disk.
    pub fn resolve_manifest() -> Option<PathBuf> {
        Self::resolve_existing(Self::manifest_candidates())
    }

    /// Standard candidate paths for the source catalog (`sources.json`).
    pub fn sources_candidates() -> &'static [&'static str] {
        &[
            "sources.json",
            "../sources.json",
            "../../sources.json",
            "Data/sources.json",
            "data/sources.json",
        ]
    }

    /// Resolves the source catalog file if it exists on disk.
    pub fn resolve_sources() -> Option<PathBuf> {
        Self::resolve_existing(Self::sources_candidates())
    }

    /// Standard candidate paths for data provenance attributions (`ATTRIBUTIONS.txt`).
    pub fn attributions_candidates() -> &'static [&'static str] {
        &[
            "data/rain/ATTRIBUTIONS.txt",
            "Data/rain/ATTRIBUTIONS.txt",
            "../data/rain/ATTRIBUTIONS.txt",
            "../Data/rain/ATTRIBUTIONS.txt",
            "../../data/rain/ATTRIBUTIONS.txt",
            "../../Data/rain/ATTRIBUTIONS.txt",
        ]
    }

    /// Resolves the attributions file if it exists on disk.
    pub fn resolve_attributions() -> Option<PathBuf> {
        Self::resolve_existing(Self::attributions_candidates())
    }

    /// Standard candidate paths for training session state persistence.
    pub fn session_candidates() -> &'static [&'static str] {
        &[
            "checkpoints/candle/training_session.json",
            "crates/inference/data/candle/training_session.json",
            "training_session.json",
            "../checkpoints/candle/training_session.json",
            "../../checkpoints/candle/training_session.json",
        ]
    }

    /// Resolves the training session state file if it exists on disk.
    pub fn resolve_session() -> Option<PathBuf> {
        Self::resolve_existing(Self::session_candidates())
    }
}
