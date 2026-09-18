"""
Continuous Diffusion / Flow-Matching Autoencoder with Affine Latent Alignment and Learned Bit-Width QAT.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 64
CONDITION_DIM = 554  # 512 (CLAP) + 41 (physical parameters) + 1 (drift)


class HierarchicalMultiResLoss(nn.Module):
    """
    Evaluates audio reconstruction loss hierarchically across multiple sampling resolutions:
    - 16 kHz: Macro atmospheric rumble, heavy thunder envelope, low-frequency pressure
    - 32 kHz: Mid-frequency wind turbulence, branch rustling, splash bodies
    - 48 kHz: Micro-transient droplet impacts, crisp HF clicks, fine acoustic air absorption
    """
    def __init__(self, target_rates: list[int] = [16000, 32000, 48000]):
        super().__init__()
        self.target_rates = target_rates

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, channels, samples = pred.shape
        device = pred.device
        total = torch.tensor(0.0, device=device)
        losses_by_rate = {}
        for rate in self.target_rates:
            if rate == 48000:
                p_sub, t_sub = pred, true
            else:
                down_factor = 48000 // rate
                p_sub = F.avg_pool1d(pred, kernel_size=down_factor, stride=down_factor)
                t_sub = F.avg_pool1d(true, kernel_size=down_factor, stride=down_factor)
                
            l1 = F.l1_loss(p_sub, t_sub)
            l1 = torch.clamp(l1, min=0.0, max=100.0)
            total = total + l1
            losses_by_rate[f"loss_{rate//1000}k"] = l1
            
        losses_by_rate["hierarchical_loss"] = total / len(self.target_rates)
        return losses_by_rate


class AffineAlignment(nn.Module):
    """
    Trainable Affine alignment layer: z_align = W * z + b
    """
    def __init__(self, dim: int = LATENT_DIM):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            return torch.einsum("ij,bjt->bit", self.weight, z) + self.bias.unsqueeze(-1)
        return F.linear(z, self.weight, self.bias)

    def inverse(self, z_align: torch.Tensor) -> torch.Tensor:
        inv_w = torch.inverse(self.weight + 1e-6 * torch.eye(self.weight.shape[0], device=self.weight.device))
        if z_align.ndim == 3:
            centered = z_align - self.bias.unsqueeze(-1)
            return torch.einsum("ij,bjt->bit", inv_w, centered)
        centered = z_align - self.bias
        return F.linear(centered, inv_w)


class LearnedMixedPrecisionQuantizer(nn.Module):
    """
    Unified Continuous Quantizer Formulation (Box-Cox Homotopy + Smooth Capacity Staircase).
    """
    def __init__(self, dim: int = LATENT_DIM, initial_bits: float = 6.0, k_max: int = 16):
        super().__init__()
        self.dim = dim
        self.k_max = k_max
        self.beta = nn.Parameter(torch.full((dim,), float(initial_bits)))
        self.lambda_raw = nn.Parameter(torch.full((dim,), -1.0))
        self.delta_prune_raw = nn.Parameter(torch.full((dim,), -3.0))
        self.register_buffer("k_indices", torch.arange(1, k_max + 1, dtype=torch.float32))

    @property
    def lambda_param(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_raw)

    @property
    def delta_prune(self) -> torch.Tensor:
        return F.softplus(self.delta_prune_raw)

    def _box_cox_forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        p = (1.0 - lam).clamp(min=-1.0, max=1.0)
        x_safe = x.clamp(min=1e-3)
        log_x = torch.log(x_safe)
        near_one = p.abs() < 1e-3
        p_safe = torch.where(near_one, torch.ones_like(p), p)
        res_linear = (x_safe.pow(p_safe) - 1.0) / p_safe
        res_log = log_x * (1.0 + 0.5 * p * log_x)
        return torch.where(near_one, res_log, res_linear)

    def _box_cox_inverse(self, y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        p = (1.0 - lam).clamp(min=-1.0, max=1.0)
        near_one = p.abs() < 1e-3
        p_safe = torch.where(near_one, torch.ones_like(p), p)
        arg = (p_safe * y + 1.0).clamp(min=1e-3)
        res_linear = arg.pow(1.0 / p_safe)
        res_log = torch.exp(y.clamp(max=10.0))
        return torch.where(near_one, res_log, res_linear)

    def forward(self, z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        orig_ndim = z.ndim
        if orig_ndim == 2:
            z = z.unsqueeze(-1)
            
        b, d, s = z.shape
        lam = self.lambda_param.view(1, d, 1)
        b_cap = torch.clamp(self.beta, 0.0, 8.0).view(1, d, 1)
        delta_p = self.delta_prune.view(1, d, 1)
        
        # Differentiable trace-safe temperature clamping
        if isinstance(tau, torch.Tensor):
            tau_safe = torch.clamp(tau, min=1e-3)
        else:
            tau_safe = max(float(tau), 1e-3)
        
        beta_gate = 1.0 / tau_safe
        mag_z = z.abs()
        prune_gate = torch.sigmoid(beta_gate * (mag_z - delta_p))
        
        phi_y = self._box_cox_forward(mag_z, lam)
        k = self.k_indices.view(1, 1, 1, self.k_max)
        d0 = 0.12
        gamma = 4.0
        gate_k = torch.sigmoid(gamma * (b_cap.unsqueeze(-1) - 1.0 - torch.log2(k)))
        beta_quant = 2.0 / tau_safe
        stair_term = 0.5 * (1.0 + torch.tanh(beta_quant * (phi_y.unsqueeze(-1) - k * d0)))
        
        warped_quant = d0 * torch.sum(gate_k * stair_term, dim=-1)
        unwarped_mag = self._box_cox_inverse(warped_quant, lam)
        
        smooth_sign = torch.tanh(z / tau_safe)
        quantized = smooth_sign * prune_gate * unwarped_mag
        
        if orig_ndim == 2:
            quantized = quantized.squeeze(-1)
        return quantized


from src.models.residual_quant import ResidualLinear, ResidualConv1d


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional State Space Block for Non-Causal Spatial VAE Encoder.
    """
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.conv_fwd = ResidualConv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.conv_bwd = ResidualConv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.fuse = ResidualLinear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        fwd = F.silu(self.conv_fwd(x, active_level=active_level))
        x_rev = torch.flip(x, dims=[-1])
        bwd_rev = F.silu(self.conv_bwd(x_rev, active_level=active_level))
        bwd = torch.flip(bwd_rev, dims=[-1])
        cat = torch.cat([fwd, bwd], dim=1).transpose(1, 2)
        out = self.fuse(cat, active_level=active_level).transpose(1, 2)
        return x + out


class SpatialAudioEncoder(nn.Module):
    """
    Encodes 4-channel 48kHz FOA audio into continuous latent sequence z.
    """
    def __init__(self, in_channels: int = 4, latent_dim: int = LATENT_DIM, cond_dim: int = CONDITION_DIM):
        super().__init__()
        self.cond_proj = ResidualLinear(cond_dim, 512)
        self.conv1 = ResidualConv1d(in_channels, 64, kernel_size=15, stride=4, padding=7)
        self.conv2 = ResidualConv1d(64, 128, kernel_size=15, stride=4, padding=7)
        self.conv3 = ResidualConv1d(128, 256, kernel_size=15, stride=5, padding=7)
        self.conv4 = ResidualConv1d(256, 512, kernel_size=15, stride=6, padding=7)
        self.bi_mamba = BidirectionalMambaBlock(d_model=512)
        self.res_conv1 = ResidualConv1d(512, 512, kernel_size=3, padding=1)
        self.res_conv2 = ResidualConv1d(512, 512, kernel_size=3, padding=1)
        self.out_proj = ResidualConv1d(512, latent_dim, kernel_size=3, padding=1)
        self.affine_align = AffineAlignment(latent_dim)
        self.quantizer = LearnedMixedPrecisionQuantizer(latent_dim)
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self, 
        x: torch.Tensor, 
        u: torch.Tensor, 
        tau: float = 0.1,
        active_level: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.silu(self.conv1(x, active_level=active_level))
        h = F.silu(self.conv2(h, active_level=active_level))
        h = F.silu(self.conv3(h, active_level=active_level))
        h = F.silu(self.conv4(h, active_level=active_level))
        h = self.bi_mamba(h, active_level=active_level)
        res = F.silu(self.res_conv1(h, active_level=active_level))
        h = h + self.res_conv2(res, active_level=active_level)
        quality_score = self.quality_head(h)
        cond = self.cond_proj(u, active_level=active_level).unsqueeze(-1)
        h = h + cond
        z_raw = self.out_proj(h, active_level=active_level)
        z_align = self.affine_align(z_raw)
        z_q = self.quantizer(z_align, tau=tau)
        return z_q, z_align, quality_score