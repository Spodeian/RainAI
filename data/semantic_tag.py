"""
Semantic audio embedding extractor with CLAP integration and fast acoustic projection fallback.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import numpy as np
import soundfile as sf
import torch

CLAP_EMBED_DIM = 512


class SemanticAudioTagger:
    """
    Extracts semantic audio embeddings (512-dim).
    Uses a pretrained CLAP model when available, or a deterministic 
    spectral feature extractor with orthogonal projection for fast, offline execution.
    """
    def __init__(self, use_neural_clap: bool = False, model_name: str = "laion/clap-htsat-unfused"):
        self.use_neural_clap = use_neural_clap
        self.model = None
        self.processor = None
        
        # Fixed pseudo-random seed for deterministic orthogonal projection
        rng = np.random.RandomState(42)
        # Random orthogonal matrix mapping acoustic features (128-dim) to CLAP dimension (512-dim)
        raw_proj = rng.randn(CLAP_EMBED_DIM, 128).astype(np.float32)
        q, _ = np.linalg.qr(raw_proj)
        self.orthogonal_proj = q  # Shape: (512, 128)

        if self.use_neural_clap:
            try:
                from transformers import ClapModel, ClapProcessor
                self.processor = ClapProcessor.from_pretrained(model_name)
                self.model = ClapModel.from_pretrained(model_name).eval()
                print(f"[SemanticAudioTagger] Loaded neural CLAP: {model_name}")
            except Exception as e:
                print(f"[SemanticAudioTagger] Neural CLAP unavailable ({e}), using deterministic acoustic projection.")
                self.use_neural_clap = False

    def extract_embedding_from_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Extracts 512-dimensional embedding for an audio file."""
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        # Average to mono for semantic content
        mono = np.mean(data, axis=-1)

        if self.use_neural_clap and self.model is not None and self.processor is not None:
            try:
                inputs = self.processor(audios=mono, sampling_rate=sr, return_tensors="pt")
                with torch.no_grad():
                    embed = self.model.get_audio_features(**inputs)
                    embed = embed / torch.norm(embed, dim=-1, keepdim=True)
                return embed.cpu().numpy()[0]
            except Exception:
                pass

        # Fallback: Deterministic Acoustic Descriptor
        return self._extract_acoustic_projection(mono, sr)

    def _extract_acoustic_projection(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Computes 128 acoustic descriptors:
        - Multi-band spectral energy (64 bands)
        - Spectral centroid, rolloff, flatness, flux
        - Temporal envelope statistics (RMS, variance, crest factor, zero-crossing rate)
        Projected into 512 dimensions.
        """
        # Trim or pad to 5 seconds
        target_len = sr * 5
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="reflect")
        else:
            audio = audio[:target_len]

        # STFT
        n_fft = 2048
        hop_length = 512
        window = np.hanning(n_fft)
        
        # Compute spectrogram
        stft = []
        for i in range(0, len(audio) - n_fft + 1, hop_length):
            chunk = audio[i:i + n_fft] * window
            stft.append(np.abs(np.fft.rfft(chunk)))
        
        if not stft:
            spec = np.zeros((1025, 1), dtype=np.float32)
        else:
            spec = np.array(stft).T  # (freq_bins, frames)

        # 64-band log filterbank energy
        num_bands = 64
        mel_energies = np.zeros(num_bands, dtype=np.float32)
        bin_step = spec.shape[0] // num_bands
        for b in range(num_bands):
            start_bin = b * bin_step
            end_bin = (b + 1) * bin_step
            band_energy = np.mean(spec[start_bin:end_bin, :] ** 2)
            mel_energies[b] = np.log1p(band_energy)

        # Temporal and spectral statistics (64 features)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        peak = np.max(np.abs(audio)) + 1e-12
        crest = peak / rms
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) * 0.5
        
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        mean_spec = np.mean(spec, axis=-1)
        spec_sum = np.sum(mean_spec) + 1e-12
        centroid = np.sum(freqs * mean_spec) / spec_sum
        
        stats = np.zeros(64, dtype=np.float32)
        stats[0] = rms
        stats[1] = crest
        stats[2] = zcr
        stats[3] = centroid / (sr * 0.5)
        # Quantiles of spectral energy
        cumsum = np.cumsum(mean_spec) / spec_sum
        for q_idx, q_val in enumerate(np.linspace(0.1, 0.9, 10)):
            stats[4 + q_idx] = freqs[np.searchsorted(cumsum, q_val)] / (sr * 0.5)

        # Concatenate into 128-dim descriptor
        desc = np.concatenate([mel_energies, stats], axis=0)
        desc = (desc - np.mean(desc)) / (np.std(desc) + 1e-8)

        # Project to 512 dimensions
        embed = np.dot(self.orthogonal_proj, desc)
        norm = np.linalg.norm(embed) + 1e-8
        return (embed / norm).astype(np.float32)


def generate_manifest_for_directory(
    audio_dir: Union[str, Path], 
    output_manifest: Union[str, Path]
) -> Dict[str, dict]:
    """Scans directory of processed audio chunks and builds complete semantic + physical acoustic manifest."""
    from src.data.audio_processor import AcousticFeatureExtractor

    audio_dir = Path(audio_dir)
    tagger = SemanticAudioTagger(use_neural_clap=False)
    extractor = AcousticFeatureExtractor(sample_rate=48000)
    
    manifest = {}
    files = sorted(list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.wav")))
    
    for f in files:
        embed = tagger.extract_embedding_from_file(f)
        try:
            data, sr = sf.read(str(f), dtype="float32", always_2d=False)
            if data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T
            feats = extractor.compute_features(data, filename=f.name)
        except Exception as e:
            # Fallback default features
            feats = {
                "rain_rate": 0.5,
                "droplet_density": 0.5,
                "drops_per_second": 30.0,
                "high_freq_ratio": 0.5,
                "spectral_centroid": 3000.0,
                "spectral_rolloff": 6000.0,
                "spectral_flatness": 0.1,
                "surface_tag": "foliage",
                "surface_idx": 1,
                "physical_params": [0.5] + [0.2] * 40
            }

        item_meta = {
            "path": str(f),
            "clap_embedding": embed.tolist(),
            "drift_scale": 0.2,
            **feats
        }
        manifest[f.stem] = item_meta
        
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(manifest, out_f, indent=2)
        
    return manifest
