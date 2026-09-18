"""
Differentiable DSP (DDSP) Subtractive Synthesis and Higher-Order Ambisonics (HOA) Engine.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SAMPLE_RATE = 48000


def spherical_harmonics_foa(azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
    """
    Computes First-Order Ambisonics (FOA) Spherical Harmonics gains (B-format, ACN/SN3D).
    azimuth (theta): in radians [-pi, pi]
    elevation (phi): in radians [-pi/2, pi/2]
    
    Returns Tensor of shape (batch, 4, seq_len):
        Channel 0: W = 1 / sqrt(2)
        Channel 1: Y = sin(theta) * cos(phi)
        Channel 2: Z = sin(phi)
        Channel 3: X = cos(theta) * cos(phi)
    """
    cos_phi = torch.cos(elevation)
    sin_phi = torch.sin(elevation)
    cos_theta = torch.cos(azimuth)
    sin_theta = torch.sin(azimuth)
    
    w = torch.full_like(azimuth, 1.0 / np.sqrt(2.0))
    y = sin_theta * cos_phi
    z = sin_phi
    x = cos_theta * cos_phi
    
    stack_dim = azimuth.dim() - 1
    return torch.stack([w, y, z, x], dim=stack_dim)


class ContinuousParametricFilter(nn.Module):
    """
    Continuous Parametric Subtractive Filter bank with Sim2Real Parameter Drift
    and Hybrid Tied/Untied Bands.
    - 12 Tied Bands: Grounded in micro-meteorological DSD & structural plate modes.
    - 4 Untied Bands: Freely learned residual acoustic textures (e.g. microphone noise, foliage rattle).
    """
    def __init__(self, num_filters: int = 16, num_tied: int = 12, latent_dim: int = 64):
        super().__init__()
        self.num_filters = num_filters
        self.num_tied = num_tied
        self.num_untied = num_filters - num_tied
        
        # Trainable bounded parameter drift delta in [-0.15, 0.15]
        # Bridges ideal mathematical physics with real-world material variance (tension, age, dampening)
        self.physics_drift = nn.Parameter(torch.zeros(num_filters))
        
        # Predicts [log_fc, log_Q, gain, azimuth, elevation] for each filter
        # 5 parameters per filter -> num_filters * 5
        self.param_net = nn.Sequential(
            nn.Conv1d(latent_dim, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, num_filters * 5, kernel_size=1)
        )

    def get_sparsity_loss(self, gains: torch.Tensor) -> torch.Tensor:
        """
        L1 sparsity penalty on filter gains. Encourages shutting down unneeded
        filters during light drizzle, saving hardware DSP ops.
        """
        return torch.mean(torch.abs(gains))

    def get_material_dampening_loss(self, q: torch.Tensor, max_damped_q: float = 8.0) -> torch.Tensor:
        """
        Physical material dampening penalty: penalizes resonances that exceed realistic bounds (Q > 8.0).
        Raindrops on wet surfaces (leaves, canvas, wood, puddle) are heavily damped; excessive Q produces artificial bell-like ringing.
        """
        excess_q = F.relu(q - max_damped_q)
        return F.smooth_l1_loss(excess_q, torch.zeros_like(excess_q), beta=1.0)


    def forward(
        self, 
        z: torch.Tensor, 
        noise: torch.Tensor,
        ambisonic_order: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: Latents of shape (batch, latent_dim, seq_len)
        noise: Tunable noise vector of shape (batch, 1, audio_samples)
        ambisonic_order: 0 (mono, 1 ch) or 1 (FOA, 4 ch)
        
        Returns:
            audio_dsp: (batch, num_channels, audio_samples)
            filter_params: Dictionary of predicted physical trajectories
        """
        batch_size = z.shape[0]
        params = self.param_net(z)  # (batch, num_filters * 5, seq_len)
        params = params.view(batch_size, self.num_filters, 5, -1)
        
        # Unpack continuous physical parameters
        # Center frequencies bounded to audio spectrum [20 Hz, 20000 Hz]
        log_fc = params[:, :, 0, :]
        raw_fc = torch.exp(torch.clamp(log_fc, min=np.log(20.0), max=np.log(20000.0)))
        
        # Apply bounded sim2real parameter drift delta: f_learned = f_raw * exp(clamp(drift, -0.15, 0.15))
        drift_factor = torch.exp(torch.clamp(self.physics_drift, -0.15, 0.15)).view(1, self.num_filters, 1)
        fc = raw_fc * drift_factor
        
        # Q factor (resonance/bandwidth) in [0.5, 20.0]
        log_q = params[:, :, 1, :]
        q = torch.exp(torch.clamp(log_q, min=np.log(0.5), max=np.log(20.0)))
        
        # Filter gains in [0, 1]
        gains = torch.sigmoid(params[:, :, 2, :])
        
        # 3D Spatial Angles
        azimuth = torch.tanh(params[:, :, 3, :]) * np.pi          # [-pi, pi]
        elevation = torch.tanh(params[:, :, 4, :]) * (np.pi / 2.0) # [-pi/2, pi/2]
        
        # Upsample parameters from latent rate (100 Hz) to audio sample rate (48000 Hz)
        audio_len = noise.shape[-1]
        fc_up = F.interpolate(fc, size=audio_len, mode="linear", align_corners=False)
        gains_up = F.interpolate(gains, size=audio_len, mode="linear", align_corners=False)
        azimuth_up = F.interpolate(azimuth, size=audio_len, mode="linear", align_corners=False)
        elevation_up = F.interpolate(elevation, size=audio_len, mode="linear", align_corners=False)
        
        # Frequency grid for spectral filtering
        fft_len = audio_len
        freqs = torch.fft.rfftfreq(fft_len, d=1.0/SAMPLE_RATE, device=noise.device).view(1, 1, -1) # (1, 1, num_freqs)
        noise_fft = torch.fft.rfft(noise, n=fft_len, dim=-1) # (batch, 1, num_freqs)
        
        # Mean center frequency and Q for spectral transfer function
        fc_mean = torch.mean(fc, dim=-1, keepdim=True) # (batch, num_filters, 1)
        q_mean = torch.mean(q, dim=-1, keepdim=True)   # (batch, num_filters, 1)
        
        # Continuous Parametric Resonant Filter transfer function:
        # |H(f)| = 1.0 / sqrt(1 + Q^2 * (f/fc - fc/f)^2)
        f_ratio = (freqs + 1.0) / (fc_mean + 1.0)
        denom = torch.sqrt(1.0 + (q_mean ** 2) * ((f_ratio - (1.0 / f_ratio)) ** 2))
        h_mag = 1.0 / torch.clamp(denom, min=1e-3, max=1e4) # (batch, num_filters, num_freqs)
        
        # Filter noise in spectral domain and invert to time domain
        filtered_noise_fft = noise_fft * h_mag # (batch, num_filters, num_freqs)
        filtered_noise = torch.fft.irfft(filtered_noise_fft, n=fft_len, dim=-1) # (batch, num_filters, audio_len)
        
        # Modulate filtered noise with time-varying gain envelope
        sources_tensor = filtered_noise * gains_up # (batch, num_filters, audio_samples)
        
        if ambisonic_order == 0:
            # Mono output (Order 0)
            audio_dsp = torch.sum(sources_tensor, dim=1, keepdim=True)
        else:
            # Order 1 (FOA, 4 channels: W, Y, Z, X)
            # Vectorized multi-band Ambisonic spherical harmonics panning
            # azimuth_up and elevation_up are (batch, num_filters, audio_len)
            sh_gains = spherical_harmonics_foa(azimuth_up, elevation_up)  # (batch, num_filters, 4, audio_len)
            panned = sources_tensor.unsqueeze(2) * sh_gains             # (batch, num_filters, 4, audio_len)
            audio_dsp = torch.sum(panned, dim=1)                        # (batch, 4, audio_len)
            
        return audio_dsp, {
            "fc": fc,
            "q": q,
            "gains": gains,
            "azimuth": azimuth,
            "elevation": elevation
        }


class DifferentiableReverbEngine(nn.Module):
    """
    Differentiable First-Order Ambisonic (FOA) Reverberator.
    Synthesizes physical room impulse responses (RIR) parameterized by:
    - rt60: Reverberation decay time in seconds [0.1, 4.0]
    - damping: High-frequency absorption factor [0.0, 1.0]
    - wet_dry: Wet/Dry reverberation mix ratio [0.0, 1.0]
    
    The entire convolution is performed in the frequency domain with torch.fft,
    enabling gradient backpropagation directly into physical enclosure and room size parameters.
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE, ir_duration: float = 0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.ir_samples = int(sample_rate * ir_duration)
        
        # Time axis for exponential energy decay envelope (e^(-6.91 * t / RT60))
        t = torch.linspace(0.0, ir_duration, self.ir_samples).view(1, 1, -1)
        self.register_buffer("time_axis", t)
        
        # Fixed orthogonal pseudo-random noise patterns for 4 FOA channels [W, Y, Z, X]
        generator = torch.Generator().manual_seed(1337)
        noise = torch.randn(1, 4, self.ir_samples, generator=generator)
        noise = noise / (torch.norm(noise, dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("diffuse_noise", noise)

    def forward(
        self,
        audio: torch.Tensor,
        rt60: Optional[torch.Tensor] = None,
        damping: Optional[torch.Tensor] = None,
        wet_dry: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        audio: (batch, 4, samples) FOA audio
        rt60: (batch, 1, 1) or float (default 0.8s)
        damping: (batch, 1, 1) or float (default 0.3)
        wet_dry: (batch, 1, 1) or float (default 0.25)
        """
        batch, channels, samples = audio.shape
        device = audio.device
        
        if rt60 is None:
            rt60 = torch.full((batch, 1, 1), 0.8, device=device)
        elif rt60.dim() == 1:
            rt60 = rt60.view(batch, 1, 1)
        elif rt60.dim() == 2:
            rt60 = rt60.unsqueeze(-1)
            
        if damping is None:
            damping = torch.full((batch, 1, 1), 0.3, device=device)
        elif damping.dim() == 1:
            damping = damping.view(batch, 1, 1)
        elif damping.dim() == 2:
            damping = damping.unsqueeze(-1)
            
        if wet_dry is None:
            wet_dry = torch.full((batch, 1, 1), 0.25, device=device)
        elif wet_dry.dim() == 1:
            wet_dry = wet_dry.view(batch, 1, 1)
        elif wet_dry.dim() == 2:
            wet_dry = wet_dry.unsqueeze(-1)
            
        # 1. Synthesize differentiable decay envelope: e^(-6.91 * t / rt60)
        # 6.91 corresponds to -60 dB decay
        decay_rate = 6.91 / torch.clamp(rt60, min=0.05, max=5.0)
        decay_env = torch.exp(-decay_rate * self.time_axis) # (batch, 1, ir_samples)
        
        # 2. Damping: lowpass attenuation over time
        # High frequencies decay faster when damping is high
        damp_env = torch.exp(-decay_rate * (1.0 + damping * 2.0) * self.time_axis)
        combined_env = (1.0 - damping) * decay_env + damping * damp_env
        
        # 3. Formulate RIR: initial direct delta impulse + diffuse decay tail
        rir = self.diffuse_noise * combined_env # (batch, 4, ir_samples)
        
        # Inject direct path peak at sample 0 safely without in-place modification
        delta = torch.zeros_like(rir)
        delta[:, :, 0] = 1.0
        rir = rir + delta
        
        # Normalize RIR energy
        rir = rir / (torch.norm(rir, dim=-1, keepdim=True) + 1e-8)
        
        # 4. FFT Convolution in frequency domain
        fft_len = samples + self.ir_samples - 1
        n_fft = 2 ** int(np.ceil(np.log2(fft_len)))
        
        audio_fft = torch.fft.rfft(audio, n=n_fft, dim=-1)
        rir_fft = torch.fft.rfft(rir, n=n_fft, dim=-1)
        
        convolved_fft = audio_fft * rir_fft
        convolved = torch.fft.irfft(convolved_fft, n=n_fft, dim=-1)[:, :, :samples]
        
        # 5. Wet/Dry mix
        output = (1.0 - wet_dry) * audio + wet_dry * convolved
        return output