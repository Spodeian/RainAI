use utilities::ingest::{
    analyze_pcm_samples, compute_file_sha256, CanonicalSurface, DownloadItem, LicenseTier, LicenseVerifier,
    ProvenanceManifest, ProvenanceRecord, SurfaceBalanceQuota,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[test]
fn test_license_verifier_logic() {
    // Approved tiers
    let (ok, tier, _) = LicenseVerifier::verify("CC0 1.0 Universal");
    assert!(ok);
    assert_eq!(tier, LicenseTier::PublicDomain);

    let (ok, tier, _) = LicenseVerifier::verify("Public Domain");
    assert!(ok);
    assert_eq!(tier, LicenseTier::PublicDomain);

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY 4.0");
    assert!(ok);
    assert_eq!(tier, LicenseTier::AttributionOnly);

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY-SA 4.0");
    assert!(ok);
    assert_eq!(tier, LicenseTier::ShareAlike);

    // Rejected tiers (NonCommercial / NoDerivs)
    let (ok, tier, reason) = LicenseVerifier::verify("CC-BY-NC 4.0");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);
    assert!(reason.contains("NonCommercial"));

    let (ok, tier, _) = LicenseVerifier::verify("CC-BY-ND 3.0");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);

    let (ok, tier, _) = LicenseVerifier::verify("All Rights Reserved Proprietary");
    assert!(!ok);
    assert_eq!(tier, LicenseTier::Restricted);
}

#[test]
fn test_canonical_surface_mapping() {
    assert_eq!(CanonicalSurface::from_category_tag("wet_asphalt_traffic"), CanonicalSurface::Asphalt);
    assert_eq!(CanonicalSurface::from_category_tag("surface_pavement"), CanonicalSurface::Pavement);
    assert_eq!(CanonicalSurface::from_category_tag("tin_roof"), CanonicalSurface::TinRoof);
    assert_eq!(CanonicalSurface::from_category_tag("canvas_tent"), CanonicalSurface::CanvasTent);
    assert_eq!(CanonicalSurface::from_category_tag("pine_needles"), CanonicalSurface::Foliage);
    assert_eq!(CanonicalSurface::from_category_tag("forest_canopy"), CanonicalSurface::Foliage);
    assert_eq!(CanonicalSurface::from_category_tag("wood_deck"), CanonicalSurface::WoodDeck);
    assert_eq!(CanonicalSurface::from_category_tag("glass_window"), CanonicalSurface::Glass);
    assert_eq!(CanonicalSurface::from_category_tag("puddle_shallow"), CanonicalSurface::PuddleShallow);
    assert_eq!(CanonicalSurface::from_category_tag("water_deep"), CanonicalSurface::WaterDeep);
}

#[test]
fn test_surface_balance_quota_entropy() {
    let mut quota = SurfaceBalanceQuota::new(2);
    assert_eq!(quota.total_samples(), 0);
    assert_eq!(quota.shannon_entropy(), 0.0);

    // Record evenly across all 9 surfaces
    for s in [
        CanonicalSurface::Asphalt,
        CanonicalSurface::Pavement,
        CanonicalSurface::TinRoof,
        CanonicalSurface::CanvasTent,
        CanonicalSurface::Foliage,
        CanonicalSurface::WoodDeck,
        CanonicalSurface::Glass,
        CanonicalSurface::PuddleShallow,
        CanonicalSurface::WaterDeep,
    ] {
        quota.record(s);
        quota.record(s);
    }

    assert_eq!(quota.total_samples(), 18);
    // When perfectly balanced across 9 classes, normalized diversity is ~1.0
    let norm_div = quota.normalized_diversity();
    assert!((norm_div - 1.0).abs() < 1e-3, "Expected normalized diversity ~1.0, got {}", norm_div);
    assert!(quota.underrepresented_surfaces().is_empty());
}

#[test]
fn test_acoustic_quality_metrics_rain_vs_silence() {
    // 1. Pure silence
    let silence = vec![0.0f32; 48000];
    let metrics_silence = analyze_pcm_samples(&silence, 48000, 1);
    assert_eq!(metrics_silence.rms_energy, 0.0);
    assert!(!metrics_silence.is_valid_rain_texture);

    // 2. Synthetic stochastic rain audio (white noise with random transient droplet spikes)
    let mut rain_sim = Vec::with_capacity(48000);
    let mut seed: u64 = 12345;
    for i in 0..48000 {
        // LCG PRNG
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rand_f = ((seed >> 32) as i32 as f32) / (i32::MAX as f32);
        let noise = rand_f * 0.08;
        // Occasional droplet spike
        let spike = if i % 1200 == 0 { 0.35 } else { 0.0 };
        rain_sim.push(noise + spike);
    }

    let metrics_rain = analyze_pcm_samples(&rain_sim, 48000, 1);
    assert!(metrics_rain.rms_energy > 0.01);
    assert!(metrics_rain.spectral_entropy > 0.6, "Entropy was {}", metrics_rain.spectral_entropy);
    assert!(metrics_rain.clipping_ratio < 0.001);
    assert!(metrics_rain.is_valid_rain_texture);
}

#[test]
fn test_provenance_manifest_serialization() {
    let mut cat_dist = HashMap::new();
    cat_dist.insert("canvas_tent".to_string(), 5);
    cat_dist.insert("tin_roof".to_string(), 6);

    let mut surf_dist = HashMap::new();
    surf_dist.insert("canvas_tent".to_string(), 5);
    surf_dist.insert("tin_roof".to_string(), 6);

    let manifest = ProvenanceManifest {
        generated_at_utc: "2026-09-17T12:00:00Z".to_string(),
        total_sources: 11,
        normalized_surface_diversity: 0.85,
        category_distribution: cat_dist,
        surface_distribution: surf_dist,
        records: vec![ProvenanceRecord {
            filename: "test_rain.wav".to_string(),
            source_url: "https://example.com/test.wav".to_string(),
            source_platform: "TestPlatform".to_string(),
            category: "tin_roof".to_string(),
            canonical_surface: CanonicalSurface::TinRoof,
            license: "CC0".to_string(),
            license_tier: LicenseTier::PublicDomain,
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            file_size_bytes: 48000,
            quality: None,
        }],
    };

    let serialized = serde_json::to_string(&manifest).expect("Serialization failed");
    let deserialized: ProvenanceManifest = serde_json::from_str(&serialized).expect("Deserialization failed");
    assert_eq!(deserialized.total_sources, 11);
    assert_eq!(deserialized.records.len(), 1);
    assert_eq!(deserialized.records[0].canonical_surface, CanonicalSurface::TinRoof);
}

#[test]
fn test_compute_file_sha256() {
    let temp_dir = std::env::temp_dir();
    let test_file = temp_dir.join("rainai_test_sha.txt");
    std::fs::write(&test_file, b"RainAI Audio Ingestion Diversity Test").expect("Failed to write test file");

    let hash = compute_file_sha256(&test_file).expect("Failed to compute SHA256");
    assert_eq!(hash.len(), 64);
    let _ = std::fs::remove_file(test_file);
}

#[test]
fn test_sources_json_catalog_integrity() {
    let candidates = [
        PathBuf::from("sources.json"),
        PathBuf::from("../../sources.json"),
        PathBuf::from("../sources.json"),
    ];
    let path = candidates.iter().find(|p| p.exists()).expect("sources.json not found in search paths");
    let content = std::fs::read_to_string(path).expect("Failed to read sources.json");
    let items: Vec<DownloadItem> = serde_json::from_str(&content).expect("Failed to parse sources.json");

    assert!(items.len() >= 120, "Expected at least 120 sources, found {}", items.len());

    let mut urls = HashSet::new();
    let mut filenames = HashSet::new();
    let mut surface_counts = HashMap::new();

    for item in &items {
        // 1. Uniqueness
        assert!(urls.insert(&item.url), "Duplicate URL in sources.json: {}", item.url);
        assert!(filenames.insert(&item.filename), "Duplicate filename in sources.json: {}", item.filename);

        // 2. Ethical Licensing
        let (approved, tier, reason) = LicenseVerifier::verify(&item.license);
        assert!(
            approved,
            "Unapproved license for item {}: {} ({})",
            item.filename, item.license, reason
        );
        assert_ne!(tier, LicenseTier::Restricted);

        // 3. Surface Mapping
        let surf = CanonicalSurface::from_category_tag(&item.category);
        *surface_counts.entry(surf).or_insert(0) += 1;
    }

    // 4. Verify all 9 canonical surfaces are represented
    assert_eq!(surface_counts.len(), 9, "Expected all 9 canonical surfaces to be represented");
    for (surf, count) in &surface_counts {
        assert!(*count >= 5, "Surface {:?} has only {} sources, expected >= 5", surf, count);
    }
}

