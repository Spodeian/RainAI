//! Asynchronous asset manager and graceful fallback hierarchy.
//!
//! Tracks download states of quantized weight tiers and WebGPU compute pipelines,
//! resolving to the best available fallback whenever a requested target is unready.

use serde::{Deserialize, Serialize};
use shared::rain::QualityTier;

/// Download and initialization state of an engine asset
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AssetState {
    NotDownloaded,
    Downloading { progress: f32 },
    Ready,
    Failed(String),
}

impl AssetState {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    pub fn is_downloading(&self) -> bool {
        matches!(self, Self::Downloading { .. })
    }

    pub fn progress(&self) -> f32 {
        match self {
            Self::Downloading { progress } => *progress,
            Self::Ready => 1.0,
            _ => 0.0,
        }
    }
}

/// Active execution hardware pathway
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionPath {
    #[default]
    CpuProcedural,
    CpuNeural,
    WebGpuNeural,
}

impl ExecutionPath {
    pub fn label(self) -> &'static str {
        match self {
            Self::CpuProcedural => "CPU (Procedural DDSP Fallback)",
            Self::CpuNeural => "CPU (Neural SIMD Float)",
            Self::WebGpuNeural => "WebGPU (Meta-Worker Accelerated)",
        }
    }
}

/// Asynchronous asset manager handling multi-tier weights and hardware pipelines
#[derive(Clone, Debug)]
pub struct AssetManager {
    pub ternary_state: AssetState,
    pub adaptive_state: AssetState,
    pub int16_state: AssetState,
    pub fp32_state: AssetState,
    pub webgpu_pipeline_state: AssetState,
}

impl Default for AssetManager {
    fn default() -> Self {
        Self {
            // Default tiers are embedded and instant
            ternary_state: AssetState::Ready,
            adaptive_state: AssetState::Ready,
            // On-demand tiers start as NotDownloaded
            int16_state: AssetState::NotDownloaded,
            fp32_state: AssetState::NotDownloaded,
            // WebGPU pipeline starts as NotDownloaded / compiling
            webgpu_pipeline_state: AssetState::NotDownloaded,
        }
    }
}

impl AssetManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Query asset state for a given tier
    pub fn tier_state(&self, tier: QualityTier) -> &AssetState {
        match tier {
            QualityTier::Ternary158 => &self.ternary_state,
            QualityTier::AdaptiveMinimum => &self.adaptive_state,
            QualityTier::HighInt16 => &self.int16_state,
            QualityTier::StudioFp32 => &self.fp32_state,
        }
    }

    /// Mutable query for asset state
    pub fn tier_state_mut(&mut self, tier: QualityTier) -> &mut AssetState {
        match tier {
            QualityTier::Ternary158 => &mut self.ternary_state,
            QualityTier::AdaptiveMinimum => &mut self.adaptive_state,
            QualityTier::HighInt16 => &mut self.int16_state,
            QualityTier::StudioFp32 => &mut self.fp32_state,
        }
    }

    /// Triggers an asynchronous background download of the target tier
    pub fn trigger_download(&mut self, tier: QualityTier) {
        let state = self.tier_state_mut(tier);
        if matches!(state, AssetState::NotDownloaded | AssetState::Failed(_)) {
            *state = AssetState::Downloading { progress: 0.05 };
        }
    }

    /// Advances simulated or real background download progress
    pub fn step_download_progress(&mut self, tier: QualityTier, delta_progress: f32) {
        let state = self.tier_state_mut(tier);
        if let AssetState::Downloading { progress } = state {
            let next = *progress + delta_progress;
            if next >= 1.0 {
                *state = AssetState::Ready;
            } else {
                *state = AssetState::Downloading { progress: next };
            }
        }
    }

    /// Set WebGPU pipeline readiness
    pub fn set_webgpu_ready(&mut self, ready: bool) {
        if ready {
            self.webgpu_pipeline_state = AssetState::Ready;
        } else {
            self.webgpu_pipeline_state = AssetState::NotDownloaded;
        }
    }

    /// Resolves the effective quality tier using the graceful fallback hierarchy.
    /// Returns `(effective_tier, is_fallback)`
    pub fn resolve_effective_tier(&self, target_tier: QualityTier) -> (QualityTier, bool) {
        if self.tier_state(target_tier).is_ready() {
            return (target_tier, false);
        }

        // Fallback chain: StudioFp32 -> HighInt16 -> AdaptiveMinimum -> Ternary158
        match target_tier {
            QualityTier::StudioFp32 => {
                if self.int16_state.is_ready() {
                    (QualityTier::HighInt16, true)
                } else if self.adaptive_state.is_ready() {
                    (QualityTier::AdaptiveMinimum, true)
                } else {
                    (QualityTier::Ternary158, true)
                }
            }
            QualityTier::HighInt16 => {
                if self.adaptive_state.is_ready() {
                    (QualityTier::AdaptiveMinimum, true)
                } else {
                    (QualityTier::Ternary158, true)
                }
            }
            QualityTier::AdaptiveMinimum => {
                if self.adaptive_state.is_ready() {
                    (QualityTier::AdaptiveMinimum, false)
                } else {
                    (QualityTier::Ternary158, true)
                }
            }
            QualityTier::Ternary158 => (QualityTier::Ternary158, false),
        }
    }

    /// Resolves the effective hardware execution path.
    /// Returns `(effective_path, is_fallback)`
    pub fn resolve_effective_path(&self, prefer_webgpu: bool) -> (ExecutionPath, bool) {
        if prefer_webgpu {
            if self.webgpu_pipeline_state.is_ready() {
                (ExecutionPath::WebGpuNeural, false)
            } else {
                // Graceful fallback to CPU Neural while GPU compiles/downloads
                (ExecutionPath::CpuNeural, true)
            }
        } else {
            (ExecutionPath::CpuNeural, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_chain_when_unready() {
        let mut mgr = AssetManager::new();
        // Target StudioFp32 (not downloaded)
        let (eff, is_fallback) = mgr.resolve_effective_tier(QualityTier::StudioFp32);
        assert_eq!(eff, QualityTier::AdaptiveMinimum);
        assert!(is_fallback);

        // Step download to completion
        mgr.trigger_download(QualityTier::StudioFp32);
        assert!(mgr.fp32_state.is_downloading());

        mgr.step_download_progress(QualityTier::StudioFp32, 1.0);
        assert!(mgr.fp32_state.is_ready());

        let (eff2, is_fallback2) = mgr.resolve_effective_tier(QualityTier::StudioFp32);
        assert_eq!(eff2, QualityTier::StudioFp32);
        assert!(!is_fallback2);
    }

    #[test]
    fn test_webgpu_cpu_fallback() {
        let mut mgr = AssetManager::new();
        // Request WebGPU when unready -> fallback to CPU
        let (path, is_fallback) = mgr.resolve_effective_path(true);
        assert_eq!(path, ExecutionPath::CpuNeural);
        assert!(is_fallback);

        // Mark WebGPU ready -> successfully returns WebGPU
        mgr.set_webgpu_ready(true);
        let (path2, is_fallback2) = mgr.resolve_effective_path(true);
        assert_eq!(path2, ExecutionPath::WebGpuNeural);
        assert!(!is_fallback2);
    }
}
