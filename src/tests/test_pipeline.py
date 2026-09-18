"""
Comprehensive Unit and Integration Test Suite for RainAI Architecture.
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial_upmix import stereo_or_mono_to_foa
from src.data.semantic_tag import SemanticAudioTagger
from src.models.diff_autoencoder import SpatialAudioEncoder, LearnedMixedPrecisionQuantizer
from src.models.ddsp import ContinuousParametricFilter
from src.models.meta_controller import InvasiveMetaController
from src.models.mamba2_moe import Mamba2MoETrajectory


class TestRainAIPipeline(unittest.TestCase):

    def test_01_spatial_upmixing(self):
        """Verify stereo to 4-channel FOA (B-format) upmixing."""
        sample_rate = 48000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo = np.stack([left, right], axis=0)
        
        foa = stereo_or_mono_to_foa(stereo, default_elevation_deg=65.0)
        self.assertEqual(foa.shape, (4, samples))
        self.assertTrue(np.all(np.isfinite(foa)))
        
        energy_w = np.mean(foa[0] ** 2)
        self.assertGreater(energy_w, 0.0)

    def test_02_semantic_tagger(self):
        """Verify semantic projection outputs 512-dim normalized embedding."""
        tagger = SemanticAudioTagger(use_neural_clap=False)
        fake_audio = np.random.randn(48000 * 2).astype(np.float32)
        embed = tagger._extract_acoustic_projection(fake_audio, sr=48000)
        
        self.assertEqual(embed.shape, (512,))
        norm = np.linalg.norm(embed)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_03_diff_autoencoder_and_quantization(self):
        """Verify VAE encoder, affine alignment, quality critic, and smooth QAT."""
        batch_size = 2
        samples = 48000
        x = torch.randn(batch_size, 4, samples)
        u = torch.randn(batch_size, 554)
        
        encoder = SpatialAudioEncoder()
        z_q, z_align, quality_score = encoder(x, u, tau=0.5)
        
        self.assertEqual(z_q.shape[0], batch_size)
        self.assertEqual(z_q.shape[1], 64)
        self.assertEqual(z_align.shape, z_q.shape)
        self.assertEqual(quality_score.shape, (batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(z_q)))
        self.assertTrue(torch.all(quality_score >= 0.0) and torch.all(quality_score <= 1.0))

    def test_04_ddsp_ambisonics(self):
        """Verify DDSP subtractive filtering, trainable drift, and spherical harmonics gains."""
        batch_size = 2
        seq_len = 100
        audio_samples = 48000
        
        z = torch.randn(batch_size, 64, seq_len)
        noise = torch.randn(batch_size, 1, audio_samples)
        
        ddsp = ContinuousParametricFilter(num_filters=8)
        self.assertEqual(ddsp.physics_drift.shape, (8,))
        
        foa_audio, params = ddsp(z, noise, ambisonic_order=1)
        self.assertEqual(foa_audio.shape, (batch_size, 4, audio_samples))
        
        sparsity = ddsp.get_sparsity_loss(params["gains"])
        self.assertTrue(sparsity >= 0.0)
        
        mono_audio, _ = ddsp(z, noise, ambisonic_order=0)
        self.assertEqual(mono_audio.shape, (batch_size, 1, audio_samples))

    def test_05_meta_controller_panic_adaptation(self):
        """Verify Meta-Controller sheds experts under buffer and telemetry starvation."""
        controller = InvasiveMetaController(num_experts=8)
        
        logits = torch.randn(1, 8)
        telemetry_healthy = torch.tensor([[50.0, 1.0, 1.0, 10.0]])
        weights = torch.tensor([[0.5, 0.5, 40.0]])
        decision_healthy = controller(logits, telemetry_healthy, weights)
        active_healthy = torch.sum(decision_healthy["expert_mask"]).item()
        
        telemetry_starved = torch.tensor([[8.0, 0.5, 0.2, 35.0]])
        decision_starved = controller(logits, telemetry_starved, weights)
        active_starved = torch.sum(decision_starved["expert_mask"]).item()
        
        self.assertLessEqual(active_starved, active_healthy)
        self.assertGreater(decision_starved["panic_factor"].item(), 0.0)
        self.assertGreater(decision_starved["jitter_factor"].item(), 0.0)
        self.assertIn("synthesis_blend", decision_starved)
        self.assertIn("telemetry_state", decision_starved)

    def test_06_mamba2_trajectory(self):
        """Verify Mamba-2 autoregressive state trajectory prediction with quality critic."""
        batch_size = 2
        seq_len = 20
        z = torch.randn(batch_size, seq_len, 64)
        u = torch.randn(batch_size, 554)
        mask = torch.ones(batch_size, 8)
        tau = torch.tensor([[1.0]])
        
        mamba = Mamba2MoETrajectory()
        mu, sigma, logits, quality_score, log_var = mamba(z, u, expert_mask=mask, tau_moe=tau, drift_scale=0.3)
        
        self.assertEqual(mu.shape, (batch_size, seq_len, 64))
        self.assertEqual(sigma.shape, (batch_size, seq_len, 64))
        self.assertEqual(logits.shape, (batch_size, 8))
        self.assertEqual(quality_score.shape, (batch_size, 1))
        self.assertEqual(log_var.shape, (batch_size, seq_len, 64))
        self.assertTrue(torch.all(sigma > 0))

    def test_07_unified_continuous_quantizer(self):
        """Verify Box-Cox homotopy, capacity staircase, and gradient flow without STE."""
        quantizer = LearnedMixedPrecisionQuantizer(dim=16, initial_bits=1.58)
        z = torch.randn(2, 16, 50, requires_grad=True)
        
        out = quantizer(z, tau=0.1)
        self.assertEqual(out.shape, z.shape)
        self.assertTrue(torch.all(torch.isfinite(out)))
        
        loss = torch.sum(out ** 2)
        loss.backward()
        
        self.assertIsNotNone(z.grad)
        self.assertIsNotNone(quantizer.beta.grad)
        self.assertIsNotNone(quantizer.lambda_raw.grad)
        self.assertIsNotNone(quantizer.delta_prune_raw.grad)


if __name__ == "__main__":
    unittest.main()
