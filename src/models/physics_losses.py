"""
Physics-Informed Loss Functions for RainAI (Hardened against NaNs/Infs):
1. MultiScaleAmbisonicPhysicsLoss:
   - Mel-Scaled Multi-Scale Spectral Convergence & Log-Magnitude with NaN guards
   - Acoustic Energy Density Conservation & DoA Intensity Vector Loss
2. PhysicsTrajectoryLoss:
   - Numerically stable clamped Log-Variance Gaussian NLL
   - Turbulent Kinetic Energy / 1/f Kolmogorov Power-Law Cascade with nan_to_num
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_WINDOW_CACHE: Dict[Tuple[int, str, Optional[int]], torch.Tensor] = {}

def get_cached_hann_window(n_fft: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, device.type, device.index)
    win = _WINDOW_CACHE.get(key)
    if win is None or win.device != device:
        win = torch.hann_window(n_fft, device=device)
        _WINDOW_CACHE[key] = win
    return win


class InstantaneousPhaseLoss(nn.Module):
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
            
            # Add epsilon to prevent torch.angle NaN gradients at exactly 0.0 + 0.0j
            stft_p = torch.stft(x_p, n_fft=n_fft, hop_length=hop, window=win, return_complex=True) + 1e-7
            stft_t = torch.stft(x_t, n_fft=n_fft, hop_length=hop, window=win, return_complex=True) + 1e-7
            
            ang_p = torch.angle(stft_p)
            ang_t = torch.angle(stft_t)
            
            cos_phase_diff = 1.0 - torch.cos(ang_p - ang_t)
            gd_p = ang_p[:, 1:, :] - ang_p[:, :-1, :]
            gd_t = ang_t[:, 1:, :] - ang_t[:, :-1, :]
            cos_gd_diff = 1.0 - torch.cos(gd_p - gd_t)
            
            total_phase_loss = total_phase_loss + torch.mean(cos_phase_diff) + 0.5 * torch.mean(cos_gd_diff)
            
        return torch.nan_to_num(total_phase_loss / len(self.fft_sizes), nan=0.0, posinf=10.0, neginf=0.0)


class MultiScaleAmbisonicPhysicsLoss(nn.Module):
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
        batch, channels, samples = x_pred.shape
        device = x_pred.device
        
        if current_step is not None and self.warmup_steps > 0:
            anneal_scale = min(1.0, float(current_step) / float(self.warmup_steps))
        else:
            anneal_scale = 1.0
        
        spectral_loss = torch.tensor(0.0, device=device)
        x_pred_flat = x_pred.view(batch * channels, samples)
        x_true_flat = x_true.view(batch * channels, samples)
        
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            win = get_cached_hann_window(n_fft, device=device)
            
            stft_pred = torch.stft(x_pred_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            stft_true = torch.stft(x_true_flat, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            
            mag_pred = torch.abs(stft_pred) + 1e-6
            mag_true = torch.abs(stft_true) + 1e-6
            
            # Replaced torch.norm with explicit sum+sqrt to protect against NaN gradients when mag_pred == mag_true
            diff_sq = (mag_true - mag_pred) ** 2
            sc = torch.sqrt(torch.sum(diff_sq) + 1e-7) / (torch.sqrt(torch.sum(mag_true ** 2)) + 1e-7)
            log_mag = F.l1_loss(torch.log(mag_pred), torch.log(mag_true))
            spectral_loss = spectral_loss + sc + log_mag
            
        spectral_loss = torch.nan_to_num(spectral_loss / len(self.fft_sizes), nan=1.0, posinf=10.0, neginf=0.0)
        
        w_p, y_p, z_p, x_p = x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], x_pred[:, 3]
        w_t, y_t, z_t, x_t = x_true[:, 0], x_true[:, 1], x_true[:, 2], x_true[:, 3]
        
        energy_pred = w_p ** 2 + (1.0 / 3.0) * (x_p ** 2 + y_p ** 2 + z_p ** 2)
        energy_true = w_t ** 2 + (1.0 / 3.0) * (x_t ** 2 + y_t ** 2 + z_t ** 2)
        
        pool = 256
        e_pred_env = F.avg_pool1d(energy_pred.unsqueeze(1), kernel_size=pool, stride=pool // 2)
        e_true_env = F.avg_pool1d(energy_true.unsqueeze(1), kernel_size=pool, stride=pool // 2)
        energy_density_loss = F.mse_loss(torch.log(e_pred_env + 1e-6), torch.log(e_true_env + 1e-6))
        energy_density_loss = torch.nan_to_num(energy_density_loss, nan=1.0, posinf=10.0, neginf=0.0)
        
        i_pred = torch.stack([w_p * x_p, w_p * y_p, w_p * z_p], dim=1)
        i_true = torch.stack([w_t * x_t, w_t * y_t, w_t * z_t], dim=1)
        
        i_pred_mean = F.avg_pool1d(i_pred, kernel_size=pool, stride=pool // 2)
        i_true_mean = F.avg_pool1d(i_true, kernel_size=pool, stride=pool // 2)
        
        cos_sim = F.cosine_similarity(i_pred_mean, i_true_mean, dim=1, eps=1e-6)
        intensity_dir_loss = torch.mean(1.0 - cos_sim)
        
        # Protected intensity magnitudes
        mag_pred_i = torch.sqrt(torch.sum(i_pred_mean ** 2, dim=1) + 1e-7)
        mag_true_i = torch.sqrt(torch.sum(i_true_mean ** 2, dim=1) + 1e-7)
        intensity_mag_loss = F.l1_loss(torch.log(mag_pred_i + 1e-6), torch.log(mag_true_i + 1e-6))
        
        doa_intensity_loss = torch.nan_to_num(intensity_dir_loss + 0.3 * intensity_mag_loss, nan=1.0, posinf=10.0, neginf=0.0)
        
        hp_pred = x_pred[:, :, 1:] - x_pred[:, :, :-1]
        hp_true = x_true[:, :, 1:] - x_true[:, :, :-1]
        
        env_pred = F.avg_pool1d(hp_pred.abs().view(batch * channels, 1, -1), kernel_size=64, stride=32)
        env_true = F.avg_pool1d(hp_true.abs().view(batch * channels, 1, -1), kernel_size=64, stride=32)
        transient_envelope_loss = torch.nan_to_num(F.l1_loss(env_pred, env_true), nan=1.0, posinf=10.0, neginf=0.0)
        
        phase_loss = self.phase_criterion(x_pred, x_true)
        
        total_loss = (
            1.0 * spectral_loss +
            anneal_scale * (
                0.4 * energy_density_loss +
                0.3 * doa_intensity_loss +
                0.3 * transient_envelope_loss +
                0.2 * phase_loss
            )
        )
        total_loss = torch.nan_to_num(total_loss, nan=5.0, posinf=20.0, neginf=0.0)
        
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
    def __init__(self, target_alpha: float = 1.0):
        super().__init__()
        self.target_alpha = target_alpha

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        device = mu.device
        
        # Hardened strict log-variance bounds to prevent exp overflow
        clamped_log_var = torch.clamp(log_var, min=-10.0, max=4.0)
        inv_var = torch.exp(-clamped_log_var)
        diff = target - mu
        smooth_diff = torch.where(torch.abs(diff) < 1.0, 0.5 * diff ** 2, torch.abs(diff) - 0.5)
        nll = torch.mean(0.5 * clamped_log_var + smooth_diff * inv_var).clamp(min=0.0, max=50.0)
        nll = torch.nan_to_num(nll, nan=1.0, posinf=50.0, neginf=0.0)
        
        vel_pred = mu[:, 1:, :] - mu[:, :-1, :]
        vel_targ = target[:, 1:, :] - target[:, :-1, :]
        vel_loss = F.mse_loss(vel_pred, vel_targ).clamp(min=0.0, max=25.0)
        vel_loss = torch.nan_to_num(vel_loss, nan=1.0, posinf=25.0, neginf=0.0)
        
        seq_len = mu.shape[1]
        if seq_len >= 16:
            fft_traj = torch.fft.rfft(mu.float(), dim=1)
            power_pred = torch.abs(fft_traj) ** 2
            
            num_freqs = power_pred.shape[1]
            freqs = torch.linspace(1.0, float(num_freqs), num_freqs, device=device).view(1, -1, 1)
            ideal_decay = 1.0 / (freqs ** self.target_alpha)
            
            power_norm = power_pred / (torch.sum(power_pred, dim=1, keepdim=True) + 1e-6)
            ideal_norm = ideal_decay / torch.sum(ideal_decay, dim=1, keepdim=True)
            
            log_power = torch.log(power_norm.clamp(min=1e-6, max=1.0))
            spectral_turbulence_loss = F.kl_div(
                log_power,
                ideal_norm.expand_as(power_norm),
                reduction="batchmean"
            ).clamp(min=0.0, max=25.0)
            spectral_turbulence_loss = torch.nan_to_num(spectral_turbulence_loss, nan=0.0, posinf=25.0, neginf=0.0)
        else:
            spectral_turbulence_loss = torch.tensor(0.0, device=device)
            
        total_traj_loss = nll + 0.5 * vel_loss + 0.1 * spectral_turbulence_loss
        total_traj_loss = torch.nan_to_num(total_traj_loss, nan=2.0, posinf=50.0, neginf=0.0)
        
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
    active_slice_level: int | torch.Tensor = 4
) -> torch.Tensor:
    slice_cost_table = [1.0, 1.4, 1.8, 2.2, 2.5]
    if isinstance(active_slice_level, torch.Tensor):
        k_val = int(active_slice_level.item())
    else:
        k_val = int(active_slice_level)
        
    clamped_k = min(max(k_val, 0), 4)
    cost_multiplier = slice_cost_table[clamped_k]
    
    num_active = torch.sum(expert_mask, dim=-1, keepdim=True)
    buffer_deficit = torch.clamp(25.0 - buf_ms, min=0.0)
    gpu_deficit = torch.clamp(1.0 - gpu_h, min=0.0)
    
    hwil_cost = 0.04 * cost_multiplier * torch.mean(num_active * (buffer_deficit + 15.0 * gpu_deficit))
    return torch.nan_to_num(hwil_cost, nan=0.0, posinf=10.0, neginf=0.0)


class STFTSubDiscriminator(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 4)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(2, 2), padding=(1, 4)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.LeakyReLU(0.2, inplace=True)),
            nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch, channels, samples = x.shape
        device = x.device
        flat_x = x.view(batch * channels, samples)
        win = get_cached_hann_window(self.n_fft, device=device)
        stft = torch.stft(flat_x, n_fft=self.n_fft, hop_length=self.hop_length, window=win, return_complex=True)
        feat = torch.stack([stft.real, stft.imag], dim=1)
        
        fmaps = []
        for layer in self.convs[:-1]:
            feat = layer(feat)
            fmaps.append(feat)
        score = self.convs[-1](feat)
        return score, fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, fft_sizes: List[int] = [512, 1024, 2048]):
        super().__init__()
        self.sub_discriminators = nn.ModuleList([STFTSubDiscriminator(n_fft=s) for s in fft_sizes])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        scores, fmaps = [], []
        for disc in self.sub_discriminators:
            s, f = disc(x)
            scores.append(s)
            fmaps.append(f)
        return scores, fmaps


def discriminator_hinge_loss(real_scores: List[torch.Tensor], fake_scores: List[torch.Tensor]) -> torch.Tensor:
    total_loss = 0.0
    for r, f in zip(real_scores, fake_scores):
        total_loss += torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return torch.nan_to_num(total_loss / len(real_scores), nan=0.0, posinf=10.0, neginf=0.0)


def generator_adversarial_loss(fake_scores: List[torch.Tensor]) -> torch.Tensor:
    total_loss = 0.0
    for f in fake_scores:
        total_loss += -torch.mean(f)
    return torch.nan_to_num(total_loss / len(fake_scores), nan=0.0, posinf=10.0, neginf=0.0)


def feature_matching_loss(real_fmaps: List[List[torch.Tensor]], fake_fmaps: List[List[torch.Tensor]]) -> torch.Tensor:
    total_fm, count = 0.0, 0
    for r_scale, f_scale in zip(real_fmaps, fake_fmaps):
        for r_feat, f_feat in zip(r_scale, f_scale):
            total_fm += F.l1_loss(f_feat, r_feat.detach())
            count += 1
    return torch.nan_to_num(total_fm / max(count, 1), nan=0.0, posinf=10.0, neginf=0.0)


class MoELoadBalancingLoss(nn.Module):
    def __init__(self, num_experts: int = 8, load_weight: float = 0.01, entropy_weight: float = 0.005):
        super().__init__()
        self.num_experts = num_experts
        self.load_weight = load_weight
        self.entropy_weight = entropy_weight

    def forward(self, gating_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if gating_probs.dim() == 3:
            probs = gating_probs.view(-1, self.num_experts)
        else:
            probs = gating_probs
            
        p_mean = torch.mean(probs, dim=0)
        f_mean = p_mean
        load_loss = self.num_experts * torch.sum(f_mean * p_mean)
        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
        entropy_loss = -entropy
        
        total = self.load_weight * load_loss + self.entropy_weight * entropy_loss
        return {
            "moe_aux_loss": torch.nan_to_num(total, nan=0.0, posinf=1.0, neginf=0.0),
            "load_loss": load_loss,
            "entropy": entropy
        }


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, feat_weight: float = 1.0, logit_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.feat_weight = feat_weight
        self.logit_weight = logit_weight

    def forward(self, student_feats: torch.Tensor, teacher_feats: torch.Tensor, student_pred: Optional[torch.Tensor] = None, teacher_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat_loss = F.mse_loss(student_feats, teacher_feats.detach())
        if student_pred is not None and teacher_pred is not None:
            p_s = F.log_softmax(student_pred / self.temperature, dim=-1)
            p_t = F.softmax(teacher_pred.detach() / self.temperature, dim=-1)
            distill_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        else:
            distill_loss = torch.tensor(0.0, device=student_feats.device)
            
        total = self.feat_weight * feat_loss + self.logit_weight * distill_loss
        return {"distillation_loss": torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=0.0), "feat_distill_loss": feat_loss, "pred_distill_loss": distill_loss}


class BetaVAEDisentanglementLoss(nn.Module):
    def __init__(self, beta: float = 2.0, tc_weight: float = 0.1):
        super().__init__()
        self.beta = beta
        self.tc_weight = tc_weight

    def forward(self, z_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_latents.dim() == 3 and z_latents.shape[1] == 64:
            z_flat = z_latents.permute(0, 2, 1).reshape(-1, z_latents.shape[1]).float()
        else:
            z_flat = z_latents.reshape(-1, z_latents.shape[-1]).float()
            
        z_flat = torch.clamp(z_flat, min=-15.0, max=15.0)
        mean = torch.mean(z_flat, dim=0)
        var = torch.clamp(torch.var(z_flat, dim=0, unbiased=False), min=1e-4, max=1e4)
        kl_prior = 0.5 * torch.sum(var + mean ** 2 - 1.0 - torch.log(var)) / z_flat.shape[-1]
        
        z_centered = z_flat - torch.mean(z_flat, dim=0, keepdim=True)
        cov = torch.matmul(z_centered.T, z_centered) / (z_flat.shape[0] - 1 + 1e-6)
        diag = torch.diag(torch.diagonal(cov))
        off_diag = cov - diag
        
        # Protected correlation calculation using explicit sum/sqrt
        tc_loss = torch.sqrt(torch.sum(off_diag ** 2) + 1e-7) / torch.sqrt(torch.sum(diag ** 2) + 1e-7).clamp(min=1e-4)
        
        total = self.beta * kl_prior + self.tc_weight * tc_loss
        return {
            "disentangle_loss": torch.nan_to_num(total, nan=0.0, posinf=20.0, neginf=0.0),
            "kl_prior": kl_prior,
            "tc_loss": tc_loss
        }