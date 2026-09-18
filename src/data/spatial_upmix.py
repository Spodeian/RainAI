"""
High-Throughput Spatial Audio Upmixing for RainAI.
Converts Mono (1ch) and Stereo (2ch) field recordings to 4-channel First-Order Ambisonics (FOA)
in AmbiX (ACN/SN3D: W, Y, Z, X) format with elevation and wind steering.
"""

from typing import Union
import numpy as np
import torch

SQRT2_INV = 0.7071067811865475

def stereo_or_mono_to_foa(
    audio: Union[np.ndarray, torch.Tensor],
    default_elevation_deg: float = 65.0,
    wind_speed_ms: float = 0.0,
    wind_azimuth_deg: float = 0.0,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Upmixes Mono or Stereo audio to First-Order Ambisonics (FOA) B-format (W, Y, Z, X).
    """
    is_torch = isinstance(audio, torch.Tensor)
    if is_torch:
        device = audio.device
        dtype = audio.dtype
        x = audio.cpu().numpy()
    else:
        x = np.asarray(audio, dtype=np.float32)

    if x.ndim == 1:
        x = x[np.newaxis, :]

    channels, samples = x.shape
    foa = np.zeros((4, samples), dtype=np.float32)

    elev_rad = np.radians(default_elevation_deg)
    sin_elev = float(np.sin(elev_rad))
    cos_elev = float(np.cos(elev_rad))

    wind_rad = np.radians(wind_azimuth_deg)
    wind_factor = min(wind_speed_ms / 15.0, 1.0)
    wind_x = float(np.cos(wind_rad)) * wind_factor
    wind_y = float(np.sin(wind_rad)) * wind_factor

    if channels == 1:
        m = x[0]
        foa[0, :] = m * SQRT2_INV
        foa[1, :] = m * wind_y * 0.5
        foa[2, :] = m * sin_elev * SQRT2_INV
        foa[3, :] = m * (cos_elev + wind_x * 0.5) * SQRT2_INV
    else:
        l = x[0]
        r = x[1]
        mid = (l + r) * 0.5
        side = (l - r) * 0.5
        foa[0, :] = (l + r) * SQRT2_INV
        foa[1, :] = side + mid * wind_y * 0.5
        foa[2, :] = mid * sin_elev * SQRT2_INV
        foa[3, :] = mid * (cos_elev + wind_x * 0.5) * SQRT2_INV

    if is_torch:
        return torch.from_numpy(foa).to(device=device, dtype=dtype)
    return foa


def apply_spatial_rir_convolution(
    foa_audio: np.ndarray,
    enclosure: float = 0.0,
    distance: float = 0.0,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    Simulates spatial Room Impulse Response (RIR) convolution across Ambisonic channels.
    """
    if enclosure <= 1e-4 and distance <= 1e-4:
        return foa_audio

    wet = foa_audio.copy()
    delay = max(int(sample_rate * 0.02 * distance), 1)
    decay = float(np.exp(-3.0 * (1.0 - enclosure * 0.5)))
    for ch in range(wet.shape[0]):
        if delay < wet.shape[1]:
            wet[ch, delay:] += foa_audio[ch, :-delay] * (enclosure * decay)
    return wet
