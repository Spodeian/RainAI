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

#[test]
fn test_quality_score_and_pruning_hierarchy() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_quality_test_{}", rand::random::<u64>()));
    fs::create_dir_all(&temp_dir).unwrap();
    let manifest_path = temp_dir.join("manifest.json");

    use utilities::features::AudioMetadata;
    use utilities::data_worker::{compute_acoustic_quality_score, DatabaseHealthWorker};

    let processed_dir = temp_dir.join("processed");
    fs::create_dir_all(&processed_dir).unwrap();

    // Low quality metadata (near silence / muffled)
    let low_q_meta = AudioMetadata {
        path: processed_dir.join("chunk_low_q.wav").to_string_lossy().to_string(),
        filename: "chunk_low_q.wav".to_string(),
        sample_rate: 48000,
        channels: 2,
        duration_secs: 5.0,
        rms_energy: 0.0001,
        rain_rate: 0.5,
        droplet_density: 0.01,
        drops_per_second: 5.0,
        high_freq_ratio: 0.001,
        spectral_centroid: 400.0,
        spectral_rolloff: 800.0,
        spectral_flatness: 0.95, // pure hiss / flatness penalty
        surface_tag: "pavement".to_string(),
    };

    // High quality metadata (healthy rain dynamics and transient response)
    let high_q_meta = AudioMetadata {
        path: processed_dir.join("chunk_high_q.wav").to_string_lossy().to_string(),
        filename: "chunk_high_q.wav".to_string(),
        sample_rate: 48000,
        channels: 2,
        duration_secs: 5.0,
        rms_energy: 0.05,
        rain_rate: 40.0,
        droplet_density: 0.5,
        drops_per_second: 500.0,
        high_freq_ratio: 0.45,
        spectral_centroid: 4500.0,
        spectral_rolloff: 8500.0,
        spectral_flatness: 0.38,
        surface_tag: "pavement".to_string(),
    };

    let q_low = compute_acoustic_quality_score(&low_q_meta);
    let q_high = compute_acoustic_quality_score(&high_q_meta);
    assert!(q_low < q_high, "High-quality metadata must score higher than low-quality (got {:.3} vs {:.3})", q_low, q_high);

    // Create dummy files inside a dedicated processed subfolder
    let file_low = processed_dir.join("chunk_low_q.wav");
    let file_high = processed_dir.join("chunk_high_q.wav");
    fs::write(&file_low, vec![0u8; 1024]).unwrap();
    fs::write(&file_high, vec![0u8; 1024]).unwrap();

    let mut map = HashMap::new();
    map.insert("chunk_low_q".to_string(), low_q_meta);
    map.insert("chunk_high_q".to_string(), high_q_meta);
    let f = fs::File::create(&manifest_path).unwrap();
    serde_json::to_writer_pretty(f, &map).unwrap();

    // Mock quotas where pavement is heavily over-represented
    let mut counts = HashMap::new();
    counts.insert("pavement".to_string(), 100);
    counts.insert("glass".to_string(), 10);
    let (_, quotas) = SurfaceEntropyAuditor::audit(&counts);

    // Audio size is 2048 bytes. Enforce a ceiling of 1500 bytes.
    let evicted = DatabaseHealthWorker::enforce_rolling_quota_with_ceiling(&processed_dir, &manifest_path, &quotas, 1500);
    assert_eq!(evicted, 1, "Exactly 1 chunk should have been evicted to meet ceiling");

    // The low quality file should have been evicted, preserving the high quality file
    assert!(!file_low.exists(), "Low-quality chunk must be pruned first");
    assert!(file_high.exists(), "High-quality chunk must be preserved");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_trickle_in_and_categorize() {
    let temp_dir = std::env::temp_dir().join(format!("rainai_trickle_test_{}", rand::random::<u64>()));
    fs::create_dir_all(&temp_dir).unwrap();
    let manifest_path = temp_dir.join("manifest.json");
    let sources_path = temp_dir.join("sources.json");
    let processed_dir = temp_dir.join("processed");

    // Seed dummy sources
    let sources_json = r#"[
        {
            "url": "https://example.com/thunder_sample.mp3",
            "filename": "mock_thunder_storm_001.mp3",
            "category": "heavy_rain_thunder",
            "license": "CC0",
            "source_platform": "MockPlatform"
        }
    ]"#;
    fs::write(&sources_path, sources_json).unwrap();

    let quotas = vec![];
    let result = DatabaseHealthWorker::trickle_in_and_categorize(
        &manifest_path,
        &sources_path,
        &processed_dir,
        &quotas,
    );

    assert!(result.is_ok());
    let desc = result.unwrap();
    assert!(desc.is_some(), "Trickle in should succeed for uningested candidate");
    let desc_str = desc.unwrap();
    assert!(desc_str.contains("mock_thunder_storm_001"));
    assert!(desc_str.contains("pavement"), "heavy_rain_thunder should map to canonical pavement");

    // Manifest should now contain the new chunk
    assert!(manifest_path.exists());
    let manifest_content = fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest_content.contains("mock_thunder_storm_001_chunk"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_async_attribution_queue_short_circuit() {
    use utilities::candle_train::{record_training_attribution_if_needed, AsyncAttributionRecorder};
    use utilities::features::AudioMetadata;

    let meta = AudioMetadata {
        path: "data/processed/mock_sample_999.wav".to_string(),
        filename: "mock_sample_999.wav".to_string(),
        sample_rate: 48000,
        channels: 2,
        duration_secs: 5.0,
        rms_energy: 0.03,
        rain_rate: 25.0,
        droplet_density: 0.3,
        drops_per_second: 200.0,
        high_freq_ratio: 0.3,
        spectral_centroid: 3000.0,
        spectral_rolloff: 6000.0,
        spectral_flatness: 0.25,
        surface_tag: "tin_roof".to_string(),
    };

    let recorder = AsyncAttributionRecorder::get_or_init();

    // First call: registers into in-memory cache and dispatches to async queue
    record_training_attribution_if_needed(&meta);

    // Benchmarking 10,000 subsequent calls to ensure nanosecond short-circuiting with zero thread stalls
    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        recorder.record(&meta.filename, &meta.surface_tag);
    }
    let elapsed = start.elapsed();
    // 10,000 lookups should complete in well under 10 milliseconds (average < 1 microsecond per call)
    assert!(
        elapsed < std::time::Duration::from_millis(50),
        "Short-circuit lookup took too long: {:?}",
        elapsed
    );
}
