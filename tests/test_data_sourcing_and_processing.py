"""
Unit tests for LicenseVerifier, BigSoundBank / Figshare ingestion logic,
AcousticFeatureExtractor, AudioProcessor, and RainSpatialDataset integration.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.ingest import LicenseVerifier, CURATED_SOURCES, RainAudioIngester
from src.data.audio_processor import AcousticFeatureExtractor, AudioProcessor
from src.data.dataset import RainSpatialDataset, TARGET_SAMPLE_RATE


def test_license_verifier_approved_licenses():
    """Verify that all permissible open licenses pass verification."""
    approved_licenses = [
        "CC0",
        "CC0 1.0 Universal",
        "Creative Commons Zero",
        "Public Domain",
        "CC-BY",
        "CC-BY 4.0",
        "CC-BY-SA 4.0",
        "Creative Commons Attribution 4.0",
        "Unlicense",
        "MIT",
        "Apache-2.0"
    ]
    for lic in approved_licenses:
        valid, reason = LicenseVerifier.verify(lic)
        assert valid is True, f"Expected '{lic}' to be approved, got: {reason}"


def test_license_verifier_rejected_licenses():
    """Verify that restrictive, non-commercial, or derivative licenses are strictly rejected."""
    rejected_licenses = [
        "CC-BY-NC 4.0",
        "CC-BY-NC-SA",
        "CC-BY-NC-ND",
        "CC-BY-ND 3.0",
        "NonCommercial",
        "Creative Commons Non-Commercial",
        "Sampling+",
        "Personal Use Only",
        "All Rights Reserved",
        ""
    ]
    for lic in rejected_licenses:
        valid, reason = LicenseVerifier.verify(lic)
        assert valid is False, f"Expected '{lic}' to be rejected, got: {reason}"


def test_curated_sources_are_all_valid_licenses():
    """Ensure every single item in CURATED_SOURCES is 100% license compliant."""
    for item in CURATED_SOURCES:
        valid, reason = LicenseVerifier.verify(item.license)
        assert valid is True, f"Curated source {item.target_filename} has invalid license: {item.license} ({reason})"


def test_acoustic_feature_extractor_synthetic():
    """Test feature extraction on synthetic high-frequency droplet noise vs low drone."""
    extractor = AcousticFeatureExtractor(sample_rate=48000)

    # 1. High frequency droplet sound (clicks + noise above 3kHz)
    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    high_freq_audio = np.random.randn(len(t)) * 0.05
    # Add sharp clicks
    click_indices = np.arange(1000, len(t) - 1000, 2400)  # ~20 clicks/sec
    high_freq_audio[click_indices] = 0.9

    feats_hf = extractor.compute_features(high_freq_audio)
    assert feats_hf["duration_sec"] == pytest.approx(2.0, abs=0.01)
    assert feats_hf["drops_per_second"] > 10.0
    assert feats_hf["droplet_density"] > 0.1
    assert "tin" in feats_hf["surface_tag"] or "glass" in feats_hf["surface_tag"] or "foliage" in feats_hf["surface_tag"]
    assert len(feats_hf["physical_params"]) == 41

    # 2. Low drone (muffled low pass)
    low_drone = np.sin(2 * np.pi * 150 * t) * 0.2
    feats_lf = extractor.compute_features(low_drone)
    assert feats_lf["spectral_centroid"] < 1500.0
    assert feats_lf["high_freq_ratio"] < 0.20


def test_acoustic_feature_extractor_compound_surfaces():
    """Verify that compound multi-surface soundscapes populate continuous Dirichlet parameter distributions."""
    extractor = AcousticFeatureExtractor(sample_rate=48000)
    audio = np.random.randn(48000 * 2).astype(np.float32) * 0.1

    # Balcony compound: pavement 0.50, glass 0.35, tin 0.15
    feats_balcony = extractor.compute_features(audio, filename="synth_compound_urban_balcony_01.flac")
    assert feats_balcony["surface_tag"] == "compound_urban_balcony"
    # surfaces slice is [10:19]
    surfaces = feats_balcony["physical_params"][10:19]
    assert surfaces[0] == pytest.approx(0.15, abs=1e-4)  # tin
    assert surfaces[3] == pytest.approx(0.50, abs=1e-4)  # pavement
    assert surfaces[7] == pytest.approx(0.35, abs=1e-4)  # glass
    assert sum(surfaces) == pytest.approx(1.0, abs=1e-4)

    # Camp compound: foliage 0.45, canvas 0.35, pine_needles 0.20
    feats_camp = extractor.compute_features(audio, filename="synth_compound_forest_camp_02.flac")
    assert feats_camp["surface_tag"] == "compound_forest_camp"
    surfaces_camp = feats_camp["physical_params"][10:19]
    assert surfaces_camp[1] == pytest.approx(0.45, abs=1e-4)  # leaves/foliage
    assert surfaces_camp[2] == pytest.approx(0.20, abs=1e-4)  # pine
    assert surfaces_camp[6] == pytest.approx(0.35, abs=1e-4)  # canvas
    assert sum(surfaces_camp) == pytest.approx(1.0, abs=1e-4)


def test_audio_processor_resampling_and_channel_standardization(tmp_path: Path):
    """Test converting 44.1kHz stereo to 48kHz 4-channel FOA."""
    processor = AudioProcessor(target_sample_rate=48000, chunk_duration_sec=2.0)

    # Create synthetic 44.1 kHz stereo audio
    orig_sr = 44100
    duration = 3.0
    samples = int(orig_sr * duration)
    stereo = np.random.randn(2, samples).astype(np.float32) * 0.1

    foa = processor.resample_and_standardize(stereo, orig_sr=orig_sr)
    assert foa.shape[0] == 4  # 4-channel FOA (W, Y, Z, X)
    expected_samples = int(duration * 48000)
    assert abs(foa.shape[1] - expected_samples) <= 2


def test_audio_processor_slicing_and_manifest(tmp_path: Path):
    """Test slicing an audio file into standardized chunks and building a manifest."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    manifest_file = tmp_path / "rain_corpus_manifest.json"
    raw_dir.mkdir()

    # Generate a 6-second test WAV file
    sr = 48000
    dur = 6.0
    audio_data = np.random.randn(int(sr * dur)).astype(np.float32) * 0.2
    test_wav = raw_dir / "test_downpour.wav"
    sf.write(str(test_wav), audio_data, sr)

    processor = AudioProcessor(
        target_sample_rate=sr,
        chunk_duration_sec=2.0,
        overlap_sec=0.5
    )
    manifest = processor.build_dataset_manifest(raw_dir, processed_dir, manifest_file)

    assert manifest_file.exists()
    assert len(manifest) > 0

    first_chunk_key = list(manifest.keys())[0]
    chunk_meta = manifest[first_chunk_key]
    assert "rain_rate" in chunk_meta
    assert "droplet_density" in chunk_meta
    assert "surface_tag" in chunk_meta
    assert len(chunk_meta["physical_params"]) == 41


def test_dataset_manifest_integration(tmp_path: Path):
    """Test RainSpatialDataset correctly reads metadata from rain_corpus_manifest.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create dummy 4-channel 48kHz audio chunk
    sr = 48000
    dur = 5.0
    num_samples = int(sr * dur)
    foa = np.random.randn(num_samples, 4).astype(np.float32) * 0.05
    chunk_wav = data_dir / "sample_slice000.wav"
    sf.write(str(chunk_wav), foa, sr)

    # Create manifest
    manifest_data = {
        "sample_slice000": {
            "rain_rate": 0.85,
            "droplet_density": 0.72,
            "surface_tag": "tin",
            "surface_idx": 0,
            "high_freq_ratio": 0.68,
            "physical_params": [0.85] + [0.2] * 40
        }
    }
    manifest_path = data_dir / "rain_corpus_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f)

    # Initialize dataset
    dataset = RainSpatialDataset(data_dir=data_dir, manifest_path=manifest_path, chunk_samples=num_samples)
    assert len(dataset) == 1

    sample = dataset[0]
    assert sample["audio"].shape == (4, num_samples)
    assert sample["conditioning"].shape == (554,)
    assert sample["rain_rate"].item() == pytest.approx(0.85, abs=1e-4)
    assert sample["droplet_density"].item() == pytest.approx(0.72, abs=1e-4)
    assert sample["surface_idx"].item() == 0
    assert sample["high_freq_ratio"].item() == pytest.approx(0.68, abs=1e-4)
