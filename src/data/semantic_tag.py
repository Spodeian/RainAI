"""
Semantic Audio Feature Extraction & CLAP Projection for RainAI.
"""

from typing import Optional
import numpy as np

class SemanticAudioTagger:
    def __init__(self, use_neural_clap: bool = False, device: str = "cpu"):
        self.use_neural_clap = use_neural_clap
        self.device = device
        self.embedding_dim = 512

    def _extract_acoustic_projection(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """
        Extracts acoustic features and projects them into a 512-dim unit-normalized vector.
        """
        if audio.ndim > 1:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio

        # Deterministic pseudo-spectral summary
        n_samples = len(mono)
        stride = max(1, n_samples // self.embedding_dim)
        features = np.zeros(self.embedding_dim, dtype=np.float32)

        for i in range(self.embedding_dim):
            start = (i * stride) % max(1, n_samples - 64)
            chunk = mono[start : start + 64]
            features[i] = float(np.mean(chunk ** 2) if len(chunk) > 0 else 0.01) + 0.01 * np.sin(i * 0.1)

        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
        else:
            features[0] = 1.0

        return features

    def get_embedding(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        return self._extract_acoustic_projection(audio, sr=sr)
