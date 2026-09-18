//! Tests for Studio Services: AudioPreviewManager & DatabaseHealthWorker.

use std::{
    collections::HashMap,
    fs,
};
use utilities::{
    audio_preview::{AuditClip, AudioPreviewManager, PreferenceChoice},
    autopilot::SurfaceEntropyAuditor,
    data_worker::{DataWorkerCommand, DatabaseHealthWorker},
};

#[test]
fn test_audio_preview_manager_workflow() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_preview_test_{}", rand::random::<u64>()));
    let feedback_file = temp_dir.join("user_feedback.json");
    let dpo_file = temp_dir.join("preference_pairs.json");

    let mut manager = AudioPreviewManager::new(feedback_file.clone(), dpo_file.clone());

    // Initially contains 1 seeded demo clip
    assert!(manager.active_clip().is_some());
    assert_eq!(manager.pending_count(), 1);

    // Mute and HITL toggles
    assert!(!manager.is_muted);
    assert!(manager.toggle_mute());
    assert!(manager.is_muted);
    assert!(!manager.toggle_mute());
    assert!(!manager.is_muted);

    assert!(manager.is_hitl_enabled);
    assert!(!manager.toggle_hitl());
    assert!(!manager.is_hitl_enabled);
    assert!(manager.toggle_hitl());
    assert!(manager.is_hitl_enabled);

    // Enqueue clips
    let clip1 = AuditClip {
        id: "clip_001".to_string(),
        step: 500,
        surface_tag: "concrete_urban".to_string(),
        timestamp: 12345678,
        audio_buffer_neural: vec![0.0; 4800],
        audio_buffer_baseline: Some(vec![0.0; 4800]),
        user_rating: None,
        user_preference: None,
    };
    let clip2 = AuditClip {
        id: "clip_002".to_string(),
        step: 1000,
        surface_tag: "foliage_canopy".to_string(),
        timestamp: 12345679,
        audio_buffer_neural: vec![0.0; 4800],
        audio_buffer_baseline: Some(vec![0.0; 4800]),
        user_rating: None,
        user_preference: None,
    };

    manager.enqueue_clip(clip1);
    manager.enqueue_clip(clip2);

    assert_eq!(manager.pending_count(), 3);

    // Navigate to clip_001 (index 1 after seeded demo clip)
    manager.next_clip();
    assert_eq!(manager.active_clip().unwrap().id, "clip_001");

    // Rate active clip (clip_001)
    manager.rate_active_clip(5).expect("Rating failed");
    assert_eq!(manager.active_clip().unwrap().user_rating, Some(5));

    // Choose preference for clip_001
    manager.prefer_active_clip(PreferenceChoice::PreferB).expect("Preference failed");
    assert_eq!(manager.active_clip().unwrap().user_preference, Some(PreferenceChoice::PreferB));

    // Move to next clip (clip_002)
    manager.next_clip();
    assert_eq!(manager.active_clip().unwrap().id, "clip_002");

    // Rate clip_002
    manager.rate_active_clip(3).expect("Rating failed");
    assert_eq!(manager.active_clip().unwrap().user_rating, Some(3));

    // Check persistence to disk
    assert!(feedback_file.exists(), "Feedback JSON file should exist");
    assert!(dpo_file.exists(), "DPO pairs JSON file should exist");

    let feedback_content = fs::read_to_string(&feedback_file).expect("Read feedback failed");
    assert!(feedback_content.contains("clip_001"));
    assert!(feedback_content.contains("concrete_urban"));

    let dpo_content = fs::read_to_string(&dpo_file).expect("Read DPO failed");
    assert!(dpo_content.contains("clip_001"));
    assert!(dpo_content.contains("chosen_is_neural"));

    // Cleanup
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_database_health_worker_lifecycle_and_entropy() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_db_worker_test_{}", rand::random::<u64>()));
    let manifest_file = temp_dir.join("manifest.json");
    let sources_file = temp_dir.join("sources.json");

    // Test SurfaceEntropyAuditor
    let mut counts = HashMap::new();
    counts.insert("pavement".to_string(), 100);
    counts.insert("tin_roof".to_string(), 100);
    counts.insert("glass".to_string(), 5); // Severe deficit

    let (entropy, quotas) = SurfaceEntropyAuditor::audit(&counts);
    assert!(entropy > 0.0 && entropy <= 1.0);
    assert_eq!(quotas.len(), 9);

    let deficit_surfaces: Vec<_> = quotas
        .iter()
        .filter(|q| q.deficit_count > 0 && q.proportion < q.target_proportion * 0.85)
        .collect();
    assert!(!deficit_surfaces.is_empty(), "Deficit surfaces must be detected");

    // Test Worker Spawn & Shutdown
    let mut worker = DatabaseHealthWorker::spawn(&manifest_file, &sources_file);
    let _ = worker.cmd_tx.send(DataWorkerCommand::TriggerAudit);
    let _ = worker.poll_telemetry();
    let _ = worker.cmd_tx.send(DataWorkerCommand::Shutdown);

    let _ = fs::remove_dir_all(&temp_dir);
}
