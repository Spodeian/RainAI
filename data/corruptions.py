"""
Acoustic Blurring and Digital Codec Corruption Suite for Denoising Autoencoder (DAE) Training.
Provides tensor-native augmentations simulating realistic field recording degradations,
lossy compression, acoustic absorption, and hardware capture flaws.
"""

from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticCorruptionPipeline:
    """
    Stochastic composition of acoustic blurring, lossy codec artifacts,
    reverberation smearing, and digital transmission dropouts.
    """
    def __init__(self, sample_rate: int = 48000, p_corrupt: float = 0.85):
        self.sample_rate = sample_rate
        self.p_corrupt = p_corrupt

    def temporal_gaussian_blur(self, x: torch.Tensor, max_kernel_size: int = 31, sigma_range: Tuple[float, float] = (1.5, 6.0)) -> torch.Tensor:
        """
        1D temporal Gaussian smoothing (simulates acoustic low-pass filtering,
        muffled double-glazing, or waterproof protective microphone casing).
        x: (batch, channels, samples)
        """
        batch, channels, samples = x.shape
        sigma = float(np.random.uniform(*sigma_range))
        kernel_size = max_kernel_size if max_kernel_size % 2 == 1 else max_kernel_size + 1
        
        # Build 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - (kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-4)) ** 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1) # Depthwise conv over all channels
        
        padded = F.pad(x, (kernel_size // 2, kernel_size // 2), mode="reflect")
        return F.conv1d(padded, kernel, groups=channels)

    def spectrogram_tf_blur(self, x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.8) -> torch.Tensor:
        """
        2D Gaussian blur over STFT magnitude (smears sharp droplet transients across time and frequency).
        x: (batch, channels, samples)
        """
        batch, channels, samples = x.shape
        x_flat = x.view(batch * channels, samples)
        
        n_fft = 1024
        hop_len = 256
        window = torch.hann_window(n_fft, device=x.device)
        
        stft = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_len, window=window, return_complex=True)
        mag = torch.abs(stft) + 1e-8
        phase = torch.angle(stft)
        
        # 2D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g2d = torch.outer(g1d, g1d)
        g2d = (g2d / torch.sum(g2d)).view(1, 1, kernel_size, kernel_size)
        
        # Pad and convolve magnitude
        mag_4d = mag.unsqueeze(1) # (B*C, 1, F, T)
        pad = kernel_size // 2
        mag_blurred = F.conv2d(F.pad(mag_4d, (pad, pad, pad, pad), mode="replicate"), g2d).squeeze(1)
        
        # Reconstruct with original phase
        blurred_stft = torch.polar(mag_blurred, phase)
        blurred_audio = torch.istft(blurred_stft, n_fft=n_fft, hop_length=hop_len, window=window, length=samples)
        return blurred_audio.view(batch, channels, samples)

    def diffuse_reverb_smear(self, x: torch.Tensor, rt60_range: Tuple[float, float] = (0.2, 0.8)) -> torch.Tensor:
        """
        Diffuse exponential decay reverberation tail (smears transient droplet attacks into continuous drone).
        """
        batch, channels, samples = x.shape
        rt60 = float(np.random.uniform(*rt60_range))
        decay_samples = int(self.sample_rate * min(rt60, 0.5))
        
        t = torch.linspace(0, min(rt60, 0.5), decay_samples, device=x.device)
        # Decay envelope: -60dB at rt60
        decay_env = torch.exp(-6.91 * t / max(rt60, 1e-3))
        noise_ir = torch.randn(decay_samples, device=x.device) * decay_env
        noise_ir[0] = 1.0 # Direct path pulse
        noise_ir = noise_ir / (torch.norm(noise_ir) + 1e-8)
        
        # Fast FFT Convolution instead of time-domain 24k-tap conv1d
        fft_len = samples + decay_samples - 1
        fft_n = 1 << (fft_len - 1).bit_length()
        x_fft = torch.fft.rfft(x, n=fft_n, dim=-1)
        ir_fft = torch.fft.rfft(noise_ir, n=fft_n, dim=-1).view(1, 1, -1)
        out = torch.fft.irfft(x_fft * ir_fft, n=fft_n, dim=-1)
        return out[..., :samples]

    def lossy_codec_emulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Emulates MP3 / Opus / Vorbis lossy compression:
        - High-frequency brickwall band-limiting (11 kHz - 14 kHz)
        - MDCT spectral hole punching (zeroing low-energy subbands)
        """
        batch, channels, samples = x.shape
        x_flat = x.view(batch * channels, samples)
        
        n_fft = 2048
        hop_len = 512
        if not hasattr(self, "_cached_win_2048") or self._cached_win_2048.device != x.device:
            self._cached_win_2048 = torch.hann_window(n_fft, device=x.device)
        window = self._cached_win_2048
        
        stft = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_len, window=window, return_complex=True)
        num_freqs = stft.shape[-2]
        
        # 1. High-frequency brickwall cutoff (simulate 128kbps MP3 / 64kbps Opus)
        cutoff_bin = int(num_freqs * np.random.uniform(0.55, 0.75)) # ~13.2 kHz to 18 kHz
        stft[:, cutoff_bin:, :] = 0.0
        
        # 2. Spectral hole punching: randomly drop 15% of middle subbands
        mid_mask = torch.rand(1, num_freqs, 1, device=x.device) > 0.18
        stft = stft * mid_mask
        
        corrupted = torch.istft(stft, n_fft=n_fft, hop_length=hop_len, window=window, length=samples)
        return corrupted.view(batch, channels, samples)

    def bit_depth_crushing(self, x: torch.Tensor, bits_range: Tuple[int, int] = (4, 8)) -> torch.Tensor:
        """
        Non-linear bit-depth reduction and dynamic range crushing.
        """
        bits = int(np.random.randint(bits_range[0], bits_range[1] + 1))
        steps = float(2 ** (bits - 1))
        
        # Mu-law companding curve before quantization
        mu = 255.0
        x_companded = torch.sign(x) * (torch.log1p(mu * torch.abs(x)) / math.log1p(mu))
        
        # Quantize to discrete steps
        quantized = torch.round(x_companded * steps) / steps
        
        # Invert mu-law
        x_expanded = torch.sign(quantized) * ((1.0 / mu) * ((1.0 + mu) ** torch.abs(quantized) - 1.0))
        return torch.clamp(x_expanded, -1.0, 1.0)

    def decimation_aliasing(self, x: torch.Tensor, factor_range: Tuple[int, int] = (2, 5)) -> torch.Tensor:
        """
        Downsamples without an anti-aliasing filter and upsamples with zero-order hold,
        folding ultrasonic splash frequencies back into the audible spectrum.
        """
        factor = int(np.random.randint(factor_range[0], factor_range[1] + 1))
        # Simple sub-sampling without filtering
        subsampled = x[:, :, ::factor]
        # Zero-order hold interpolation (nearest neighbor upsample)
        aliased = F.interpolate(subsampled, size=x.shape[-1], mode="nearest")
        return aliased

    def packet_loss_dropouts(self, x: torch.Tensor, max_dropouts: int = 4, dropout_ms: Tuple[float, float] = (10.0, 30.0)) -> torch.Tensor:
        """
        Simulates network packet loss / WebRTC buffer underrun with zero-order hold PLC.
        """
        out = x.clone()
        samples = x.shape[-1]
        num_drops = np.random.randint(1, max_dropouts + 1)
        
        for _ in range(num_drops):
            drop_len = int(self.sample_rate * (np.random.uniform(*dropout_ms) / 1000.0))
            if drop_len >= samples:
                continue
            start = int(np.random.randint(0, samples - drop_len))
            # Zero-order hold freeze (freeze last sample)
            freeze_val = out[:, :, start:start+1]
            out[:, :, start:start + drop_len] = freeze_val
            
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastically applies a chain of 1 to 3 distinct corruption operators.
        x: (batch, channels, samples) pristine audio
        """
        if np.random.uniform(0.0, 1.0) > self.p_corrupt:
            # Clean passthrough with slight microphone jitter
            corrupted = x + 0.01 * torch.randn_like(x)
        else:
            corrupted = x.clone()
            operators = [
                self.temporal_gaussian_blur,
                self.spectrogram_tf_blur,
                self.diffuse_reverb_smear,
                self.lossy_codec_emulation,
                self.bit_depth_crushing,
                self.decimation_aliasing,
                self.packet_loss_dropouts
            ]
            
            # Randomly choose 1 to 3 operators
            num_ops = int(np.random.randint(1, 4))
            chosen_ops = np.random.choice(operators, size=num_ops, replace=False)
            
            for op in chosen_ops:
                try:
                    corrupted = op(corrupted)
                except Exception:
                    # Fallback on numerical safety
                    pass
                    
        # Final gain normalization to avoid clipping
        max_val = torch.max(torch.abs(corrupted))
        if max_val > 1.0:
            corrupted = corrupted / (max_val + 1e-6)
            
        return corrupted
