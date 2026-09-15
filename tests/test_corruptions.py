"""
Unit Tests for Acoustic Corruptions and 8 Hardware Stress Profiles.
"""

import sys
from pathlib import Path
import unittest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corruptions import AcousticCorruptionPipeline
from src.models.meta_controller import InvasiveMetaController


class TestAcousticCorruptionsAndHardwareProfiles(unittest.TestCase):

    def setUp(self):
        self.pipeline = AcousticCorruptionPipeline(sample_rate=48000)
        self.batch_size = 2
        self.channels = 4
        self.samples = 48000
        self.audio = torch.randn(self.batch_size, self.channels, self.samples)

    def test_01_temporal_gaussian_blur(self):
        blurred = self.pipeline.temporal_gaussian_blur(self.audio, max_kernel_size=21, sigma_range=(2.0, 4.0))
        self.assertEqual(blurred.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(blurred)))
        
        diff_orig = torch.diff(self.audio, dim=-1)
        diff_blur = torch.diff(blurred, dim=-1)
        self.assertLess(torch.var(diff_blur).item(), torch.var(diff_orig).item())

    def test_02_spectrogram_tf_blur(self):
        blurred = self.pipeline.spectrogram_tf_blur(self.audio, kernel_size=5, sigma=1.5)
        self.assertEqual(blurred.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(blurred)))

    def test_03_diffuse_reverb_smear(self):
        blurred = self.pipeline.diffuse_reverb_smear(self.audio, rt60_range=(0.1, 0.3))
        self.assertEqual(blurred.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(blurred)))

    def test_04_lossy_codec_emulation(self):
        codec_audio = self.pipeline.lossy_codec_emulation(self.audio)
        self.assertEqual(codec_audio.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(codec_audio)))

    def test_05_bit_depth_crushing(self):
        crushed = self.pipeline.bit_depth_crushing(self.audio, bits_range=(4, 6))
        self.assertEqual(crushed.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(crushed)))
        unique_vals = len(torch.unique(torch.round(crushed[0, 0, :1000] * 100)))
        self.assertLess(unique_vals, 200)

    def test_06_decimation_aliasing(self):
        aliased = self.pipeline.decimation_aliasing(self.audio, factor_range=(3, 4))
        self.assertEqual(aliased.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(aliased)))

    def test_07_packet_loss_dropouts(self):
        dropped = self.pipeline.packet_loss_dropouts(self.audio, max_dropouts=3, dropout_ms=(15.0, 25.0))
        self.assertEqual(dropped.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(dropped)))

    def test_08_pipeline_stochastic_composition(self):
        corrupted = self.pipeline(self.audio)
        self.assertEqual(corrupted.shape, self.audio.shape)
        self.assertTrue(torch.all(torch.isfinite(corrupted)))
        self.assertLessEqual(torch.max(torch.abs(corrupted)).item(), 1.05)

    def test_09_all_8_hardware_stress_profiles(self):
        controller = InvasiveMetaController(num_experts=8)
        logits = torch.randn(8, 8)
        user_weights = torch.tensor([[0.7, 0.3, 40.0]]).repeat(8, 1)
        quality_scores = torch.tensor([[0.9, 0.9]]).repeat(8, 1)
        
        telemetry = torch.tensor([
            [45.0, 1.00, 1.00, 10.0],
            [12.0, 0.35, 0.35, 28.0],
            [15.0, 0.90, 0.90, 42.0],
            [35.0, 0.55, 0.30, 12.0],
            [25.0, 0.80, 0.40, 25.0],
            [140.0, 0.95, 0.95, 15.0],
            [18.0, 0.15, 0.15, 12.0],
            [30.0, 0.25, 1.00, 11.0],
        ])
        
        decision = controller(logits, telemetry, user_weights, quality_scores=quality_scores)
        
        self.assertIn("expert_mask", decision)
        self.assertIn("safety_reserve_ms", decision)
        self.assertIn("synthesis_blend", decision)
        
        panic_p0 = decision["panic_factor"][0].item()
        panic_p1 = decision["panic_factor"][1].item()
        panic_p5 = decision["panic_factor"][5].item()
        self.assertGreater(panic_p1, panic_p0)
        self.assertEqual(panic_p5, 0.0)
        
        jitter_p2 = decision["jitter_factor"][2].item()
        jitter_p0 = decision["jitter_factor"][0].item()
        self.assertGreater(jitter_p2, jitter_p0)


if __name__ == "__main__":
    unittest.main()
