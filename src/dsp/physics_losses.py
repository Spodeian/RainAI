"""
Physics-Informed Loss Functions for RainAI:
1. MultiScaleAmbisonicPhysicsLoss:
   - Mel-Scaled Multi-Scale Spectral Convergence & Log-Magnitude
   - Acoustic Energy Density Conservation (E = W^2 + 1/3*(X^2 + Y^2 + Z^2))
   - Active Acoustic Intensity Vector (DoA direction-of-arrival: I = p * v)
   - High-Frequency Transient Envelope (Hilbert/Analytic droplet sharpness)
2. PhysicsTrajectoryLoss:
   - Log-Variance Formulated Gaussian Negative Log-Likelihood
   - Turbulent Kinetic Energy / 1/f Kolmogorov Power-Law Cascade
3. SliceAwareHWILPenalty:
   - Compute-density weighted hardware penalty based on active quantization slice level
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_WINDOW_CACHE: Dict[Tuple[int, str, Optional[int]], torch.Tensor] = {}

def get_cached_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    """Retrieves or creates a cached Hann window tensor on the specified device."""
    key = (n_fft, device.type, device.index)
    win = _WINDOW_CACHE.get(key)
    if win is None or win.device != device:
        win = torch.hann_window(n_fft, device=device)
        _WINDOW_CACHE[key] = win
    return win


class InstantaneousPhaseLoss(nn.Module):
    """
    Instantaneous Phase and Group Delay Difference Loss for Ambisonic Audio:
    Phase alignment in First Order Ambisonics (FOA) governs spatial localization cues.
    Computes smooth cosine phase difference and instantaneous frequency (group delay)
    along frequency and time axes without 2pi wrap discontinuities.
    """
    def __init__(self, fft_sizes: List[int] = [512, 1024]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = x_pred.shape
        device = x_pred.device
        x_p = x_pred.view(batch * channels, samples)
        x_t = x_true.view(batch * channels, samples)
        
        total_phase_loss = torch.tensor(0.0, device=device)
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = get_cached_hann_window(n_fft, device=device)
            stft_p = torch.stft(x_p, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            stft_t = torch.stft(x_t, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            
            ang_p = torch.angle(stft_p)
            ang_t = torch.angle(stft_t)
            
            # Smooth cosine distance on phase angles (0 when identical, 2 when anti-phase)
            cos_phase_diff = 1.0 - torch.cos(ang_p - ang_t)
            
            # Group delay / instantaneous frequency differences (frequency derivative)
            gd_p = ang_p[:, 1:, :] - ang_p[:, :-1, :]
            gd_t = ang_t[:, 1:, :] - ang_t[:, :-1, :]
            cos_gd_diff = 1.0 - torch.cos(gd_p - gd_t)
            
            total_phase_loss = total_phase_loss + torch.mean(cos_phase_diff) + 0.5 * torch.mean(cos_gd_diff)
            
        return total_phase_loss / len(self.fft_sizes)


class MultiScaleAmbisonicPhysicsLoss(nn.Module):
    """
    Complete Physics-Informed Ambisonic Audio Loss:
    Integrates spectral convergence, physical energy density, intensity vector (DoA),
    transient envelope conservation, and instantaneous phase alignment with dynamic annealing.
    """
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        sample_rate: int = 48000,
        warmup_steps: int = 2000
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.sample_rate = sample_rate
        self.warmup_steps = warmup_steps
        self.phase_criterion = InstantaneousPhaseLoss(fft_sizes=[512, 1024])

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        current_step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        x_pred: Predicted audio of shape (batch, 4, samples) [W, Y, Z, X]
        x_true: Pristine ground truth audio of shape (batch, 4, samples) [W, Y, Z, X]
        current_step: Step index for dynamic loss weight annealing
        """
        batch, channels, samples = x_pred.shape
        device = x_pred.device
        
        # Calculate dynamic annealing scale for secondary physics constraints
        if current_step is not None and self.warmup_steps > 0:
            anneal_scale = min(1.0, float(current_step) / float(self.warmup_steps))
        else:
            anneal_scale = 1.0
        
        # ---------------------------------------------------------------------
        # 1. Multi-Scale Spectral Convergence and Log-Magnitude Loss
        # ---------------------------------------------------------------------
        spectral_loss = torch.tensor(0.0, device=device)
        x_pred_flat = x_pred.view(batch * channels, samples)
        x_true_flat = x_true.view(batch * channels, samples)
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = get_cached_hann_window(n_fft, device=device)
            
            stft_pred = torch.stft(x_pred_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            stft_true = torch.stft(x_true_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            
            mag_pred = torch.abs(stft_pred) + 1e-7
            mag_true = torch.abs(stft_true) + 1e-7
            
            # Linear spectral convergence + Log magnitude
            sc = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-7)
            log_mag = F.l1_loss(torch.log(mag_pred), torch.log(mag_true))
            spectral_loss = spectral_loss + sc + log_mag
            
        spectral_loss = spectral_loss / len(self.fft_sizes)
        
        # ---------------------------------------------------------------------
        # 2. Ambisonic Acoustic Energy Density Conservation:
        # E = W^2 + (1/3) * (X^2 + Y^2 + Z^2)
        # Channel order: W=0, Y=1, Z=2, X=3
        # ---------------------------------------------------------------------
        w_p, y_p, z_p, x_p = x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], x_pred[:, 3]
        w_t, y_t, z_t, x_t = x_true[:, 0], x_true[:, 1], x_true[:, 2], x_true[:, 3]
        
        energy_pred = w_p ** 2 + (1.0 / 3.0) * (x_p ** 2 + y_p ** 2 + z_p ** 2)
        energy_true = w_t ** 2 + (1.0 / 3.0) * (x_t ** 2 + y_t ** 2 + z_t ** 2)
        
        # Short-time average energy envelope (smoothed over 256 samples ~ 5.3ms)
        pool = 256
        e_pred_env = F.avg_pool1d(energy_pred.unsqueeze(1), kernel_size=pool, stride=pool // 2)
        e_true_env = F.avg_pool1d(energy_true.unsqueeze(1), kernel_size=pool, stride=pool // 2)
        energy_density_loss = F.mse_loss(torch.log(e_pred_env + 1e-6), torch.log(e_true_env + 1e-6))
        
        # ---------------------------------------------------------------------
        # 3. Active Acoustic Intensity Vector Loss (3D Direction of Arrival DoA):
        # I = [W*X, W*Y, W*Z] (acoustic pressure * particle velocity)
        # ---------------------------------------------------------------------
        i_pred = torch.stack([w_p * x_p, w_p * y_p, w_p * z_p], dim=1) # (batch, 3, samples)
        i_true = torch.stack([w_t * x_t, w_t * y_t, w_t * z_t], dim=1) # (batch, 3, samples)
        
        # Time-averaged intensity vectors over frames
        i_pred_mean = F.avg_pool1d(i_pred, kernel_size=pool, stride=pool // 2) # (batch, 3, frames)
        i_true_mean = F.avg_pool1d(i_true, kernel_size=pool, stride=pool // 2) # (batch, 3, frames)
        
        # Cosine directional similarity + magnitude alignment
        cos_sim = F.cosine_similarity(i_pred_mean, i_true_mean, dim=1, eps=1e-6)
        intensity_dir_loss = torch.mean(1.0 - cos_sim)
        
        mag_pred_i = torch.norm(i_pred_mean, p=2, dim=1)
        mag_true_i = torch.norm(i_true_mean, p=2, dim=1)
        intensity_mag_loss = F.l1_loss(torch.log(mag_pred_i + 1e-6), torch.log(mag_true_i + 1e-6))
        
        doa_intensity_loss = intensity_dir_loss + 0.3 * intensity_mag_loss
        
        # ---------------------------------------------------------------------
        # 4. Transient Envelope Loss (High-frequency impact sharpness):
        # Raindrops are sharp micro-transients that STFT windows smear in time.
        # Analytic envelope via smoothed high-frequency rectification.
        # ---------------------------------------------------------------------
        hp_pred = x_pred[:, :, 1:] - x_pred[:, :, :-1]
        hp_true = x_true[:, :, 1:] - x_true[:, :, :-1]
        
        env_pred = F.avg_pool1d(hp_pred.abs().view(batch * channels, 1, -1), kernel_size=64, stride=32)
        env_true = F.avg_pool1d(hp_true.abs().view(batch * channels, 1, -1), kernel_size=64, stride=32)
        transient_envelope_loss = F.l1_loss(env_pred, env_true)
        
        # ---------------------------------------------------------------------
        # 5. Instantaneous Phase Alignment Loss
        # ---------------------------------------------------------------------
        phase_loss = self.phase_criterion(x_pred, x_true)
        
        # Combined weighted composite loss with dynamic annealing
        total_loss = (
            1.0 * spectral_loss +
            anneal_scale * (
                0.4 * energy_density_loss +
                0.3 * doa_intensity_loss +
                0.3 * transient_envelope_loss +
                0.2 * phase_loss
            )
        )
        
        return {
            "total_loss": total_loss,
            "spectral_loss": spectral_loss,
            "energy_density_loss": energy_density_loss,
            "doa_intensity_loss": doa_intensity_loss,
            "transient_envelope_loss": transient_envelope_loss,
            "phase_loss": phase_loss,
            "anneal_scale": torch.tensor(anneal_scale, device=device)
        }



class PhysicsTrajectoryLoss(nn.Module):
    """
    Physics-Informed Latent Trajectory Loss for Mamba SSD:
    1. Numerically Stable Log-Variance Gaussian NLL:
       NLL = 0.5 * (log_var + (target - mu)^2 * exp(-log_var))
    2. Fluid Turbulence 1/f Kolmogorov Power-Law Cascade:
       Penalizes unnatural high-frequency trajectory oscillations.
    """
    def __init__(self, target_alpha: float = 1.0):
        super().__init__()
        self.target_alpha = target_alpha  # 1.0 = 1/f pink noise cascade

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        mu: (batch, seq_len, latent_dim)
        log_var: (batch, seq_len, latent_dim)
        target: (batch, seq_len, latent_dim)
        """
        device = mu.device
        
        # 1. Numerically Stable Gaussian NLL with Huber transition
        clamped_log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        inv_var = torch.exp(-clamped_log_var)
        diff = target - mu
        # Huber transition: exact Gaussian NLL for |diff| < 1.0, robust linear for outliers
        smooth_diff = torch.where(torch.abs(diff) < 1.0, 0.5 * diff ** 2, torch.abs(diff) - 0.5)
        nll = torch.mean(0.5 * clamped_log_var + smooth_diff * inv_var).clamp(max=100.0)
        
        # 2. Turbulent Kinetic Energy & Temporal Velocity Loss
        # Prevents high-frequency robotic stepping artifacts across frames
        vel_pred = mu[:, 1:, :] - mu[:, :-1, :]
        vel_targ = target[:, 1:, :] - target[:, :-1, :]
        vel_loss = F.mse_loss(vel_pred, vel_targ).clamp(max=50.0)
        
        # 3. 1/f Power Spectrum Decay on Latent Fluctuations
        # Fluid turbulence states exhibit power spectrum P(f) proportional to 1/f^alpha
        seq_len = mu.shape[1]
        if seq_len >= 16:
            # 1D FFT over temporal dimension of latent trajectory (float32 for cuFFT stability)
            fft_traj = torch.fft.rfft(mu.float(), dim=1) # (batch, num_freqs, latent_dim)
            power_pred = torch.abs(fft_traj) ** 2 # (batch, num_freqs, latent_dim)
            
            num_freqs = power_pred.shape[1]
            freqs = torch.linspace(1.0, float(num_freqs), num_freqs, device=device).view(1, -1, 1)
            # Ideal 1/f decay: 1 / (f^alpha)
            ideal_decay = 1.0 / (freqs ** self.target_alpha)
            
            # Normalize power across frequency to compare spectral shape
            power_norm = power_pred / (torch.sum(power_pred, dim=1, keepdim=True) + 1e-6)
            ideal_norm = ideal_decay / torch.sum(ideal_decay, dim=1, keepdim=True)
            
            log_power = torch.log(power_norm.clamp(min=1e-5, max=1.0))
            spectral_turbulence_loss = F.kl_div(
                log_power,
                ideal_norm.expand_as(power_norm),
                reduction="batchmean"
            ).clamp(min=0.0, max=50.0)
        else:
            spectral_turbulence_loss = torch.tensor(0.0, device=device)
            
        total_traj_loss = nll + 0.5 * vel_loss + 0.1 * spectral_turbulence_loss
        
        return {
            "total_traj_loss": total_traj_loss,
            "nll_loss": nll,
            "velocity_loss": vel_loss,
            "turbulence_loss": spectral_turbulence_loss
        }


def compute_slice_aware_hwil_penalty(
    expert_mask: torch.Tensor,
    buf_ms: torch.Tensor,
    gpu_h: torch.Tensor,
    active_slice_level: int = 4
) -> torch.Tensor:
    """
    Calculates hardware-in-the-loop performance penalty, weighted by the active quantization slice level.
    Ternary (slice 0) costs less FLOP/memory bandwidth than Full Precision (slice 4).
    """
    # Relative compute cost multiplier: Slice 0 (Ternary)=1.0x, Slice 2 (INT8)=1.8x, Slice 4 (FP32)=2.5x
    slice_cost_table = [1.0, 1.4, 1.8, 2.2, 2.5]
    clamped_k = min(max(active_slice_level, 0), 4)
    cost_multiplier = slice_cost_table[clamped_k]
    
    num_active = torch.sum(expert_mask, dim=-1, keepdim=True)  # (B, 1)
    buffer_deficit = torch.clamp(25.0 - buf_ms, min=0.0)
    gpu_deficit = torch.clamp(1.0 - gpu_h, min=0.0)
    
    hwil_cost = 0.04 * cost_multiplier * torch.mean(num_active * (buffer_deficit + 15.0 * gpu_deficit))
    return hwil_cost


class STFTSubDiscriminator(nn.Module):
    """
    2D Spectrogram Sub-Discriminator operating on complex STFT representations (Real + Imag).
    Captures multi-scale phase consistency and high-frequency micro-transient textures.
    """
    def __init__(self, n_fft: int = 1024, hop_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        
        # Convolutions process 2 channels (Real, Imag) from FOA flattened channels
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (batch, 4, samples)
        Returns:
            score: (batch * 4, 1, F_down, T_down)
            fmaps: List of intermediate activation tensors for feature matching
        """
        batch, channels, samples = x.shape
        device = x.device
        flat_x = x.view(batch * channels, samples)
        win = get_cached_hann_window(self.n_fft, device=device)
        stft = torch.stft(flat_x, n_fft=self.n_fft, hop_length=self.hop_length, window=win, return_complex=True)
        
        # Real + Imaginary 2-channel representation: (batch * channels, 2, freq_bins, time_frames)
        feat = torch.stack([stft.real, stft.imag], dim=1)
        
        fmaps = []
        for layer in self.convs[:-1]:
            feat = layer(feat)
            fmaps.append(feat)
        score = self.convs[-1](feat)
        return score, fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator operating across window sizes [512, 1024, 2048]
    to evaluate both sharp temporal transient detail and broad harmonic spectral distributions.
    """
    def __init__(self, fft_sizes: List[int] = [512, 1024, 2048]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([
            STFTSubDiscriminator(n_fft=s) for s in fft_sizes
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores = []
        fmaps = []
        for disc in self.sub_discriminators:
            s, f = disc(x)
            scores.append(s)
            fmaps.append(f)
        return scores, fmaps


def discriminator_hinge_loss(
    real_scores: List[torch.Tensor],
    fake_scores: List[torch.Tensor]
) -> torch.Tensor:
    """Computes Multi-Scale Hinge Loss for the Discriminator."""
    total_loss = 0.0
    for r, f in zip(real_scores, fake_scores):
        total_loss += torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return total_loss / len(real_scores)


def generator_adversarial_loss(
    fake_scores: List[torch.Tensor]
) -> torch.Tensor:
    """Computes Multi-Scale Hinge Loss for the Generator."""
    total_loss = 0.0
    for f in fake_scores:
        total_loss += -torch.mean(f)
    return total_loss / len(fake_scores)


def feature_matching_loss(
    real_fmaps: List[List[torch.Tensor]],
    fake_fmaps: List[List[torch.Tensor]]
) -> torch.Tensor:
    """Computes L1 feature matching loss across all discriminator layers."""
    total_fm = 0.0
    count = 0
    for r_scale, f_scale in zip(real_fmaps, fake_fmaps):
        for r_feat, f_feat in zip(r_scale, f_scale):
            total_fm += F.l1_loss(f_feat, r_feat.detach())
            count += 1
    return total_fm / max(count, 1)


class MoELoadBalancingLoss(nn.Module):
    """
    Auxiliary load-balancing and entropy loss for Mixture of Experts (MoE) routing.
    Prevents router expert collapse and ensures full utilization of all 8 Mamba experts.
    """
    def __init__(self, num_experts: int = 8, load_weight: float = 0.01, entropy_weight: float = 0.005):
        super().__init__()
        self.num_experts = num_experts
        self.load_weight = load_weight
        self.entropy_weight = entropy_weight

    def forward(self, gating_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        gating_probs: (batch, num_experts) or (batch, seq_len, num_experts) routing probabilities
        """
        if gating_probs.dim() == 3:
            probs = gating_probs.view(-1, self.num_experts)
        else:
            probs = gating_probs
            
        # P_i: Average probability assigned to expert i
        p_mean = torch.mean(probs, dim=0) # (num_experts,)
        
        # f_i: Fraction of inputs routed to expert i (using argmax or softmax proxy)
        # Using softmax expectation as differentiable load estimate
        f_mean = p_mean
        
        # Load balancing loss: E * sum(f_i * P_i)
        load_loss = self.num_experts * torch.sum(f_mean * p_mean)
        
        # Entropy penalty: encourage uniform routing exploration
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        entropy_loss = -entropy  # Minimize negative entropy = maximize entropy
        
        total = self.load_weight * load_loss + self.entropy_weight * entropy_loss
        return {
            "moe_aux_loss": total,
            "load_loss": load_loss,
            "entropy": entropy
        }


class KnowledgeDistillationLoss(nn.Module):
    """
    Teacher-to-Student Distillation Loss:
    Transfers knowledge from an unconstrained high-precision FP32 Teacher model
    to a resource-constrained, quantized Student model.
    """
    def __init__(self, temperature: float = 2.0, feat_weight: float = 1.0, logit_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.feat_weight = feat_weight
        self.logit_weight = logit_weight

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        student_pred: Optional[torch.Tensor] = None,
        teacher_pred: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Feature representation matching (e.g. latent sequence MSE)
        feat_loss = F.mse_loss(student_feats, teacher_feats.detach())
        
        # Trajectory prediction distribution matching if available
        if student_pred is not None and teacher_pred is not None:
            p_s = F.log_softmax(student_pred / self.temperature, dim=-1)
            p_t = F.softmax(teacher_pred.detach() / self.temperature, dim=-1)
            distill_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        else:
            distill_loss = torch.tensor(0.0, device=student_feats.device)
            
        total = self.feat_weight * feat_loss + self.logit_weight * distill_loss
        return {
            "distillation_loss": total,
            "feat_distill_loss": feat_loss,
            "pred_distill_loss": distill_loss
        }


class BetaVAEDisentanglementLoss(nn.Module):
    """
    Beta-VAE Latent Disentanglement & Mutual Information Regularization:
    Forces the continuous latent space to mathematically separate independent acoustic factors
    (e.g., rainfall rate, wind speed, resonance materials) along orthogonal dimensions.
    """
    def __init__(self, beta: float = 2.0, tc_weight: float = 0.1):
        super().__init__()
        self.beta = beta
        self.tc_weight = tc_weight

    def forward(self, z_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_latents: (batch, latent_dim, seq_len) or (batch, seq_len, latent_dim)
        """
        device = z_latents.device
        if z_latents.dim() == 3 and z_latents.shape[1] == 64:
            # (batch, 64, seq_len) -> (batch * seq_len, 64)
            z_flat = z_latents.permute(0, 2, 1).reshape(-1, z_latents.shape[1]).float()
        else:
            z_flat = z_latents.reshape(-1, z_latents.shape[-1]).float()
            
        # Numerically robust bounding to prevent AMP overflow (float16 max 65504)
        z_flat = torch.clamp(z_flat, min=-20.0, max=20.0)
        
        # Standard unit Gaussian prior penalty on latent variance and mean (in float32 for AMP stability)
        mean = torch.mean(z_flat, dim=0)
        var = torch.clamp(torch.var(z_flat, dim=0, unbiased=False), min=1e-4, max=1e4)
        kl_prior = 0.5 * torch.sum(var + mean ** 2 - 1.0 - torch.log(var)) / z_flat.shape[-1]
        
        # Total Correlation (TC) / Dimension Independence penalty:
        # Off-diagonal elements of correlation matrix should be zero
        z_centered = z_flat - torch.mean(z_flat, dim=0, keepdim=True)
        cov = torch.matmul(z_centered.T, z_centered) / (z_flat.shape[0] - 1 + 1e-6)
        diag = torch.diag(torch.diagonal(cov))
        off_diag = cov - diag
        tc_loss = torch.norm(off_diag, p="fro") / (torch.norm(diag, p="fro").clamp(min=1e-4))
        
        total = self.beta * kl_prior + self.tc_weight * tc_loss
        return {
            "disentangle_loss": total,
            "kl_prior": kl_prior,
            "tc_loss": tc_loss
        }

