"""
Physical Rain Sound Synthesizer and Fluid Dynamics Acoustics.
Matches native Rust rainai_synth engine equations.
"""

from typing import Union, Optional, Dict
import numpy as np

MAX_DROPLET_DIAMETER_MM = 5.5

def gunn_kinzer_terminal_velocity(d_mm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Empirical Gunn-Kinzer terminal velocity: v_t = 9.65 - 10.3 * exp(-0.6 * d).
    """
    d = np.clip(np.asarray(d_mm, dtype=np.float32), 0.1, 5.5)
    vt = np.maximum(9.65 - 10.3 * np.exp(-0.6 * d), 0.8)
    if np.isscalar(d_mm):
        return float(vt)
    return vt

def sample_gamma_dsd(rainfall_rate_mmh: float = 1.0, num_drops: int = 1000) -> np.ndarray:
    """
    Ulbrich Gamma Drop Size Distribution (DSD).
    """
    r = max(float(rainfall_rate_mmh), 0.05)
    lambda_val = 4.1 * (r ** -0.21)
    shape = 3.0
    scale = max(1.0 / lambda_val, 0.1)
    samples = np.random.gamma(shape, scale, size=num_drops)
    return np.clip(samples, 0.2, MAX_DROPLET_DIAMETER_MM)

class PhysicalRainSynthesizer:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_single_droplet(
        self,
        diameter_mm: float = 1.5,
        surface: str = "water",
        material_mod: float = 1.0,
        wind_speed_ms: float = 0.0
    ) -> np.ndarray:
        vt = gunn_kinzer_terminal_velocity(diameter_mm)
        duration_sec = min(max(0.006 + 0.005 * diameter_mm, 0.006), 0.035)
        n_samples = max(int(self.sample_rate * duration_sec), 128)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        radius_m = (diameter_mm * 0.5) * 1e-3
        f0 = min(max((3.26 / max(radius_m, 1e-4)) * material_mod, 250.0), 14000.0)

        # Bubble resonance for water surfaces
        if "water" in surface.lower():
            has_bubble = (0.8 <= diameter_mm <= 2.2)
            damp = 180.0 if has_bubble else 450.0
            decay = np.exp(-damp * t)
            droplet = np.sin(2 * np.pi * f0 * t) * decay * (diameter_mm ** 2)
        elif "roof" in surface.lower() or "tin" in surface.lower():
            decay = np.exp(-220.0 * t)
            droplet = (np.sin(2 * np.pi * (1250.0 * material_mod) * t) * 0.5 + np.sin(2 * np.pi * (2550.0 * material_mod) * t) * 0.25) * decay
        else:
            decay = np.exp(-450.0 * t)
            droplet = np.sin(2 * np.pi * f0 * t) * decay * 0.3

        return droplet.astype(np.float32)

    def generate_rain_texture(
        self,
        duration_sec: float = 2.0,
        rainfall_rate_mmh: float = 15.0,
        surface: Union[str, Dict[str, float]] = "water",
        wind_speed_ms: float = 0.0,
        temp_c: float = 20.0,
        humidity_rel: float = 0.5,
        **kwargs
    ) -> np.ndarray:
        total_samples = int(duration_sec * self.sample_rate)
        stereo = np.zeros((2, total_samples), dtype=np.float32)

        # Generate dense droplet impulse texture
        num_droplets = int(min(rainfall_rate_mmh * 80.0 * duration_sec, 800))
        diameters = sample_gamma_dsd(rainfall_rate_mmh, num_drops=num_droplets)

        if isinstance(surface, dict):
            surf_names = list(surface.keys())
            surf_probs = np.array(list(surface.values()), dtype=np.float64)
            surf_probs = surf_probs / np.sum(surf_probs)
        else:
            surf_names = [surface]
            surf_probs = [1.0]

        for d in diameters:
            chosen_surf = np.random.choice(surf_names, p=surf_probs)
            drop_audio = self.generate_single_droplet(float(d), surface=chosen_surf, wind_speed_ms=wind_speed_ms)
            drop_len = len(drop_audio)
            if drop_len >= total_samples:
                continue

            start = int(np.random.randint(0, total_samples - drop_len))
            pan = float(np.random.uniform(0.1, 0.9))
            stereo[0, start : start + drop_len] += drop_audio * pan
            stereo[1, start : start + drop_len] += drop_audio * (1.0 - pan)

        # ISO 9613-1 air absorption attenuation
        absorb = max(1.0 - (temp_c / 100.0) * (1.0 - humidity_rel * 0.5), 0.5)
        stereo *= absorb

        # Normalize and ensure minimum amplitude
        peak = float(np.max(np.abs(stereo)))
        if peak > 0.0:
            target_peak = max(min(rainfall_rate_mmh / 50.0, 0.8), 0.2)
            stereo = stereo * (target_peak / peak)

        return stereo
