"""
Unit tests for LicenseVerifier, AcousticFeatureExtractor, and RainSpatialDataset integration.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.ingest import LicenseVerifier, CURATED_SOURCES
from src.data.audio_processor import AcousticFeatureExtractor, AudioProcessor
from src.data.dataset import RainSpatialDataset


def test_license_verifier_approved_licenses():
    approved_licenses = ["CC0", "Creative Commons Zero", "Public Domain", "CC-BY 4.0", "MIT"]
    for lic in approved_licenses:
        valid, reason = LicenseVerifier.verify(lic)
        assert valid is True


def test_license_verifier_rejected_licenses():
    rejected_licenses = ["CC-BY-NC 4.0", "CC-BY-NC-SA", "NonCommercial", "All Rights Reserved"]
    for lic in rejected_licenses:
        valid, reason = LicenseVerifier.verify(lic)
        assert valid is False


def test_curated_sources_are_all_valid_licenses():
    for item in CURATED_SOURCES:
        valid, reason = LicenseVerifier.verify(item.license)
        assert valid is True


def test_acoustic_feature_extractor_synthetic():
    extractor = AcousticFeatureExtractor(sample_rate=48000)
    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    high_freq_audio = np.random.randn(len(t)) * 0.05
    click_indices = np.arange(1000, len(t) - 1000, 2400)
    high_freq_audio[click_indices] = 0.9

    feats_hf = extractor.compute_features(high_freq_audio)
    assert feats_hf["duration_sec"] == pytest.approx(2.0, abs=0.01)


def test_acoustic_feature_extractor_compound_surfaces():
    extractor = AcousticFeatureExtractor(sample_rate=48000)
    audio = np.random.randn(48000 * 2).astype(np.float32) * 0.1

    feats_balcony = extractor.compute_features(audio, filename="synth_compound_urban_balcony_01.flac")
    assert feats_balcony["surface_tag"] == "compound_urban_balcony"
    surfaces = feats_balcony["physical_params"][10:19]
    assert sum(surfaces) == pytest.approx(1.0, abs=1e-4)


def test_audio_processor_resampling_and_channel_standardization():
    processor = AudioProcessor(target_sample_rate=48000, chunk_duration_sec=2.0)
    orig_sr = 44100
    duration = 3.0
    samples = int(orig_sr * duration)
    stereo = np.random.randn(2, samples).astype(np.float32) * 0.1

    foa = processor.resample_and_standardize(stereo, orig_sr=orig_sr)
    assert foa.shape[0] == 4


def test_dataset_manifest_integration(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    sr = 48000
    dur = 5.0
    num_samples = int(sr * dur)
    foa = np.random.randn(num_samples, 4).astype(np.float32) * 0.05
    chunk_wav = data_dir / "sample_slice000.wav"
    sf.write(str(chunk_wav), foa, sr)

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

    dataset = RainSpatialDataset(data_dir=data_dir, manifest_path=manifest_path, chunk_samples=num_samples)
    assert len(dataset) == 1
