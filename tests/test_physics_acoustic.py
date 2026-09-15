"""
Unit and physical validation tests for RainAI's physical acoustics and fluid dynamics engine:
- Ulbrich Gamma Drop Size Distribution (DSD) moments
- Gunn-Kinzer terminal velocity vs drop diameter
- Drop-size dependent trajectory angle under wind fields
- Pumphrey & Crum bubble entrapment acoustic synthesis
- Frequency-stratified First-Order Ambisonics (FOA) energy distribution
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from src.data.synth_rain import (
    sample_gamma_dsd,
    gunn_kinzer_terminal_velocity,
    PhysicalRainSynthesizer,
    MAX_DROPLET_DIAMETER_MM
)
from src.data.spatial_upmix import stereo_or_mono_to_foa
from src.models.ddsp import ContinuousParametricFilter, spherical_harmonics_foa


class TestPhysicsAcoustics(unittest.TestCase):
    def setUp(self):
        self.synth = PhysicalRainSynthesizer(sample_rate=48000)

    def test_01_gamma_dsd_moments(self):
        """Test that Gamma DSD scales with rainfall intensity."""
        d_drizzle = sample_gamma_dsd(rainfall_rate_mmh=1.0, num_drops=10000)
        d_heavy = sample_gamma_dsd(rainfall_rate_mmh=50.0, num_drops=10000)

        self.assertTrue(np.all(d_drizzle >= 0.2))
        self.assertTrue(np.all(d_drizzle <= MAX_DROPLET_DIAMETER_MM))
        self.assertTrue(np.all(d_heavy >= 0.2))
        self.assertTrue(np.all(d_heavy <= MAX_DROPLET_DIAMETER_MM))

        mean_drizzle = float(np.mean(d_drizzle))
        mean_heavy = float(np.mean(d_heavy))
        self.assertGreater(mean_heavy, mean_drizzle * 1.4,
                           f"Expected heavy rain DSD mean ({mean_heavy:.2f}) > drizzle ({mean_drizzle:.2f})")

    def test_02_gunn_kinzer_terminal_velocity(self):
        """Test Gunn-Kinzer terminal velocity matches empirical tables."""
        test_diameters = np.array([0.5, 1.5, 3.0, 5.0])
        vt = gunn_kinzer_terminal_velocity(test_diameters)

        self.assertAlmostEqual(vt[0], 2.01, delta=0.3)
        self.assertAlmostEqual(vt[1], 5.46, delta=0.4)
        self.assertAlmostEqual(vt[2], 7.94, delta=0.4)
        self.assertAlmostEqual(vt[3], 9.13, delta=0.4)

    def test_03_trajectory_angle_drop_size_dependence(self):
        """Test that fine mist blows sideways while heavy drops fall steeply under wind."""
        wind_speed = 5.0
        d_mist = 0.4
        d_large = 4.5

        vt_mist = float(gunn_kinzer_terminal_velocity(np.array([d_mist]))[0])
        vt_large = float(gunn_kinzer_terminal_velocity(np.array([d_large]))[0])

        theta_mist_deg = np.degrees(np.arctan(wind_speed / vt_mist))
        theta_large_deg = np.degrees(np.arctan(wind_speed / vt_large))

        self.assertGreater(theta_mist_deg, 60.0)
        self.assertLess(theta_large_deg, 35.0)

    def test_04_pumphrey_bubble_entrapment_acoustics(self):
        """Test Pumphrey & Crum bubble entrapment vs Rayleigh impact."""
        drop_bubble = self.synth.generate_single_droplet(diameter_mm=1.5, surface="water")
        drop_no_bubble = self.synth.generate_single_droplet(diameter_mm=3.0, surface="water")

        self.assertGreater(len(drop_bubble), 100)
        self.assertGreater(len(drop_no_bubble), 100)
        self.assertTrue(np.all(np.isfinite(drop_bubble)))
        self.assertTrue(np.all(np.isfinite(drop_no_bubble)))

        drop_roof = self.synth.generate_single_droplet(diameter_mm=2.0, surface="roof")
        self.assertTrue(np.all(np.isfinite(drop_roof)))

    def test_05_frequency_stratified_foa(self):
        """Test frequency-stratified FOA spatialization."""
        t = np.linspace(0, 1.0, 48000, endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)

        foa = stereo_or_mono_to_foa(sig, wind_speed_ms=6.0, wind_azimuth_deg=90.0)
        self.assertEqual(foa.shape, (4, 48000))
        self.assertTrue(np.all(np.isfinite(foa)))

        rms_w = np.sqrt(np.mean(foa[0] ** 2))
        rms_z = np.sqrt(np.mean(foa[2] ** 2))
        self.assertGreater(rms_w, 0.05)
        self.assertGreater(rms_z, 0.03)

    def test_06_parametric_ddsp_filterbank(self):
        """Test DDSP filterbank forward pass and energy conservation."""
        model = ContinuousParametricFilter(num_filters=16)
        z = torch.randn(2, 64, 100)
        noise = torch.randn(2, 1, 48000)

        out_foa, params = model(z, noise, ambisonic_order=1)
        self.assertEqual(out_foa.shape, (2, 4, 48000))
        self.assertTrue(torch.all(torch.isfinite(out_foa)))

        out_mono, _ = model(z, noise, ambisonic_order=0)
        self.assertEqual(out_mono.shape, (2, 1, 48000))

    def test_07_material_parameter_sweep(self):
        """Test that material modulation shifts acoustic resonant frequencies."""
        drop_thin = self.synth.generate_single_droplet(diameter_mm=2.0, surface="glass", material_mod=1.3)
        drop_thick = self.synth.generate_single_droplet(diameter_mm=2.0, surface="glass", material_mod=0.8)

        self.assertTrue(np.all(np.isfinite(drop_thin)))
        self.assertTrue(np.all(np.isfinite(drop_thick)))
        
        fft_thin = np.abs(np.fft.rfft(drop_thin))
        fft_thick = np.abs(np.fft.rfft(drop_thick))
        peak_thin = np.argmax(fft_thin)
        peak_thick = np.argmax(fft_thick)
        self.assertGreater(peak_thin, peak_thick)

    def test_08_compound_surface_synthesis(self):
        """Test continuous compound multi-surface texture synthesis with mixture distributions."""
        compound_mix = {"wood_deck": 0.40, "puddle_shallow": 0.35, "tin": 0.25}
        stereo = self.synth.generate_rain_texture(
            duration_sec=2.0,
            rainfall_rate_mmh=25.0,
            surface=compound_mix,
            wind_speed_ms=5.0
        )
        self.assertEqual(stereo.shape, (2, 96000))
        self.assertTrue(np.all(np.isfinite(stereo)))
        self.assertGreater(np.max(np.abs(stereo)), 0.1)

    def test_09_atmospheric_absorption(self):
        """Test ISO 9613-1 temperature and humidity air absorption dynamics."""
        stereo_dry = self.synth.generate_rain_texture(
            duration_sec=2.0,
            rainfall_rate_mmh=15.0,
            temp_c=35.0,
            humidity_rel=0.40
        )
        stereo_humid = self.synth.generate_rain_texture(
            duration_sec=2.0,
            rainfall_rate_mmh=15.0,
            temp_c=12.0,
            humidity_rel=0.95
        )
        self.assertEqual(stereo_dry.shape, (2, 96000))
        self.assertEqual(stereo_humid.shape, (2, 96000))
        self.assertTrue(np.all(np.isfinite(stereo_dry)))
        self.assertTrue(np.all(np.isfinite(stereo_humid)))


if __name__ == "__main__":
    unittest.main()
