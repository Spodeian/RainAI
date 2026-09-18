//! Asynchronous Audio Preview Worker, CPAL Monitor & Human-in-the-Loop Audit Queue.
//!
//! Synthesizes short preview clips asynchronously at training checkpoints without stalling
//! the main pipeline, maintains a persistent review inbox, and commits user ratings/preferences
//! directly to `data/user_feedback.json` and `data/preference_pairs.json`.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// A/B preference choice for Human-in-the-Loop audio evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreferenceChoice {
    PreferA,
    PreferB,
    Tie,
}

/// A synthesized audio clip in the audit review queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditClip {
    pub id: String,
    pub surface_tag: String,
    pub step: usize,
    pub timestamp: u64,
    pub audio_buffer_neural: Vec<f32>,
    pub audio_buffer_baseline: Option<Vec<f32>>,
    pub user_rating: Option<u8>,
    pub user_preference: Option<PreferenceChoice>,
}

/// Serialized feedback log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedbackEntry {
    pub clip_id: String,
    pub surface_tag: String,
    pub step: usize,
    pub timestamp: u64,
    pub rating: Option<u8>,
    pub preference: Option<PreferenceChoice>,
}

/// Pairwise preference record for offline Direct Preference Optimization (DPO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePairRecord {
    pub clip_id: String,
    pub surface_tag: String,
    pub chosen_is_neural: bool,
    pub timestamp: u64,
}

/// Manager for live audio monitoring and the persistent audit queue.
pub struct AudioPreviewManager {
    pub queue: Vec<AuditClip>,
    pub active_idx: usize,
    pub is_muted: bool,
    pub is_hitl_enabled: bool,
    pub feedback_path: PathBuf,
    pub dpo_pairs_path: PathBuf,
}

impl Default for AudioPreviewManager {
    fn default() -> Self {
        Self::new("data/user_feedback.json", "data/preference_pairs.json")
    }
}

impl AudioPreviewManager {
    pub fn new<P: AsRef<Path>>(feedback_path: P, dpo_pairs_path: P) -> Self {
        let mut mgr = Self {
            queue: Vec::new(),
            active_idx: 0,
            is_muted: false,
            is_hitl_enabled: true,
            feedback_path: feedback_path.as_ref().to_path_buf(),
            dpo_pairs_path: dpo_pairs_path.as_ref().to_path_buf(),
        };

        // Seed with a default demo clip if queue is empty
        mgr.seed_demo_clip();
        mgr
    }

    /// Seed a synthetic demo clip so the UI has immediate content to preview.
    pub fn seed_demo_clip(&mut self) {
        if self.queue.is_empty() {
            let sample_len = 48000 * 2; // 2 seconds at 48kHz
            let mut neural_buf = Vec::with_capacity(sample_len);
            let mut base_buf = Vec::with_capacity(sample_len);

            for i in 0..sample_len {
                let t = i as f32 / 48000.0;
                let n_val = (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.15;
                let b_val = (t * 330.0 * 2.0 * std::f32::consts::PI).sin() * 0.15;
                neural_buf.push(n_val);
                base_buf.push(b_val);
            }

            self.queue.push(AuditClip {
                id: "demo_tin_roof_01".into(),
                surface_tag: "tin_roof".into(),
                step: 100,
                timestamp: 1726700000,
                audio_buffer_neural: neural_buf,
                audio_buffer_baseline: Some(base_buf),
                user_rating: None,
                user_preference: None,
            });
        }
    }

    /// Enqueue a newly generated checkpoint clip.
    pub fn enqueue_clip(&mut self, clip: AuditClip) {
        self.queue.push(clip);
    }

    /// Returns count of clips awaiting user review.
    pub fn pending_count(&self) -> usize {
        self.queue.iter().filter(|c| c.user_rating.is_none() && c.user_preference.is_none()).count()
    }

    /// Returns the currently selected audit clip.
    pub fn active_clip(&self) -> Option<&AuditClip> {
        self.queue.get(self.active_idx)
    }

    /// Advances to next clip in queue.
    pub fn next_clip(&mut self) {
        if !self.queue.is_empty() {
            self.active_idx = (self.active_idx + 1) % self.queue.len();
        }
    }

    /// Goes to previous clip in queue.
    pub fn prev_clip(&mut self) {
        if !self.queue.is_empty() {
            self.active_idx = if self.active_idx == 0 {
                self.queue.len() - 1
            } else {
                self.active_idx - 1
            };
        }
    }

    /// Submit a 1..5 star rating for the active clip.
    pub fn rate_active_clip(&mut self, rating: u8) -> Result<()> {
        let entry = if let Some(clip) = self.queue.get_mut(self.active_idx) {
            clip.user_rating = Some(rating.clamp(1, 5));

            Some(UserFeedbackEntry {
                clip_id: clip.id.clone(),
                surface_tag: clip.surface_tag.clone(),
                step: clip.step,
                timestamp: clip.timestamp,
                rating: Some(rating),
                preference: clip.user_preference,
            })
        } else {
            None
        };

        if let Some(entry) = entry {
            self.persist_feedback_entry(&entry)?;
        }
        Ok(())
    }

    /// Submit an A/B preference choice for the active clip.
    pub fn prefer_active_clip(&mut self, choice: PreferenceChoice) -> Result<()> {
        let (entry, maybe_pair) = if let Some(clip) = self.queue.get_mut(self.active_idx) {
            clip.user_preference = Some(choice);

            let entry = UserFeedbackEntry {
                clip_id: clip.id.clone(),
                surface_tag: clip.surface_tag.clone(),
                step: clip.step,
                timestamp: clip.timestamp,
                rating: clip.user_rating,
                preference: Some(choice),
            };

            let maybe_pair = if choice != PreferenceChoice::Tie {
                Some(PreferencePairRecord {
                    clip_id: clip.id.clone(),
                    surface_tag: clip.surface_tag.clone(),
                    chosen_is_neural: choice == PreferenceChoice::PreferB,
                    timestamp: clip.timestamp,
                })
            } else {
                None
            };
            (Some(entry), maybe_pair)
        } else {
            (None, None)
        };

        if let Some(entry) = entry {
            self.persist_feedback_entry(&entry)?;
        }
        if let Some(pair) = maybe_pair {
            self.persist_dpo_pair(&pair)?;
        }
        Ok(())
    }

    /// Append feedback entry to JSON log.
    fn persist_feedback_entry(&self, entry: &UserFeedbackEntry) -> Result<()> {
        if let Some(parent) = self.feedback_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut entries: Vec<UserFeedbackEntry> = if self.feedback_path.exists() {
            let data = fs::read_to_string(&self.feedback_path).unwrap_or_else(|_| "[]".into());
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        entries.push(entry.clone());
        let json_str = serde_json::to_string_pretty(&entries)?;
        fs::write(&self.feedback_path, json_str)?;
        Ok(())
    }

    /// Append DPO preference pair to JSON log.
    fn persist_dpo_pair(&self, pair: &PreferencePairRecord) -> Result<()> {
        if let Some(parent) = self.dpo_pairs_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut pairs: Vec<PreferencePairRecord> = if self.dpo_pairs_path.exists() {
            let data = fs::read_to_string(&self.dpo_pairs_path).unwrap_or_else(|_| "[]".into());
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        pairs.push(pair.clone());
        let json_str = serde_json::to_string_pretty(&pairs)?;
        fs::write(&self.dpo_pairs_path, json_str)?;
        Ok(())
    }

    /// Toggle audio monitor mute.
    pub fn toggle_mute(&mut self) -> bool {
        self.is_muted = !self.is_muted;
        self.is_muted
    }

    /// Toggle Human-in-the-Loop review mode.
    pub fn toggle_hitl(&mut self) -> bool {
        self.is_hitl_enabled = !self.is_hitl_enabled;
        self.is_hitl_enabled
    }
}
