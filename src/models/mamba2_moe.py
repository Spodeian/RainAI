"""
Mamba-2 State Space Duality (SSD) Continuous Latent Trajectory Model with Dynamic Threshold MoE
and Multi-Slice Residual Quantization (W = sum_{i=0}^{N-1} S_i).
"""

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_quant import ResidualLinear, ResidualConv1d

LATENT_DIM = 64
CONDITION_DIM = 554  # 512 (CLAP) + 41 (physical parameters) + 1 (drift)
NUM_EXPERTS = 8


class MambaSSDBlock(nn.Module):
    """
    Simplified State Space Duality (SSD) block with Residual Quantization.
    Combines 1D causal temporal convolutions with state-space recurrence:
    h_{t} = A h_{t-1} + B x_t
    y_t = C h_t + D x_t
    """
    def __init__(self, d_model: int = 128, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Dense and Conv projections are multi-slice residual quantized
        self.in_proj = ResidualLinear(d_model, d_model * 2)
        self.conv1d = ResidualConv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        # State space recurrent stability parameters (Kept as full FP32 continuous parameters)
        # decay = exp(-exp(A_log)) in (0, 1) strictly guarantees stable non-exploding recurrence
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 0.5))
        self.B_proj = ResidualLinear(d_model, d_state)
        self.C_proj = ResidualLinear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = ResidualLinear(d_model, d_model)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        proj = self.in_proj(x, active_level=active_level)
        u, gate = proj.chunk(2, dim=-1)
        
        # Causal 1D conv over time
        u_conv = self.conv1d(u.transpose(1, 2), active_level=active_level)[:, :, :seq_len].transpose(1, 2)
        u_act = F.silu(u_conv)
        
        # State space recurrence
        B = self.B_proj(u_act, active_level=active_level)
        C = self.C_proj(u_act, active_level=active_level)
        
        # Clamp A_log to 8.0 to prevent FP16 inf * 0 gradients
        A_log_safe = torch.clamp(self.A_log, max=8.0)
        decay = torch.exp(-torch.exp(A_log_safe)).unsqueeze(0)  # (1, d_model, d_state)
        
        # Fast prefix scan / recurrence
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            u_t = u_act[:, t, :].unsqueeze(-1)       # (batch, d_model, 1)
            b_t = B[:, t, :].unsqueeze(1)            # (batch, 1, d_state)
            c_t = C[:, t, :].unsqueeze(1)            # (batch, 1, d_state)
            
            # Stable Recurrence: h_t = decay * h_{t-1} + u_t * b_t
            h = decay * h + (u_t * b_t)
            # Output: y_t = sum(h_t * c_t) + D * u_t
            y_t = torch.sum(h * c_t, dim=-1) + self.D * u_act[:, t, :]
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_model)
        y = y * F.silu(gate)
        return self.out_proj(y, active_level=active_level)


class JambaSelfAttentionBlock(nn.Module):
    """
    Hybrid Attention-SSM injection layer (Jamba style):
    Provides exact associative long-term recall to complement Mamba-2's linear SSM recurrence.
    """
    def __init__(self, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attn_out)


class Mamba2MoETrajectory(nn.Module):
    """
    Continuous Latent Trajectory Model with 8 Mamba-2 Experts, Jamba Hybrid Self-Attention,
    Dynamic Token Dropping, Flow Matching, and Multi-Slice Residual Quantization.
    Controlled by the Invasive Meta-Controller.
    """
    def __init__(
        self, 
        latent_dim: int = LATENT_DIM, 
        cond_dim: int = CONDITION_DIM, 
        num_experts: int = NUM_EXPERTS,
        d_model: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Multi-slice residual projections
        self.latent_proj = ResidualLinear(latent_dim, d_model)
        self.cond_proj = ResidualLinear(cond_dim, d_model)
        
        # Gating router: predicts unnormalized logits L for each expert
        self.router_linear1 = ResidualLinear(d_model, 64)
        self.router_linear2 = ResidualLinear(64, num_experts)
        
        # 8 Specialized Mamba-2 Experts
        self.experts = nn.ModuleList([
            MambaSSDBlock(d_model=d_model) for _ in range(num_experts)
        ])
        
        # Hybrid Jamba Attention layer for long-term associative memory
        self.attention_fusion = JambaSelfAttentionBlock(d_model=d_model, num_heads=4)
        
        # Dynamic Token Dropping Gate: identifies stationary temporal latent frames (e.g. steady hum)
        self.stationarity_gate = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Output distribution heads: predict mean mu and log_variance sigma of next latent
        self.mu_head = ResidualLinear(d_model, latent_dim)
        self.log_var_head = ResidualLinear(d_model, latent_dim)
        
        # Shared-weight critic: predicts the physical plausibility / expected quality of the trajectory
        self.quality_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def get_activation_sparsity_loss(self) -> torch.Tensor:
        """Computes structured L1 activation and parameter sparsity penalty on experts."""
        l1_reg = torch.tensor(0.0)
        for expert in self.experts:
            l1_reg = l1_reg + torch.mean(torch.abs(expert.D))
        return 0.001 * l1_reg

    def compute_flow_matching_loss(
        self,
        z_target: torch.Tensor,
        u: torch.Tensor,
        expert_mask: torch.Tensor,
        tau_moe: torch.Tensor,
        sigma_min: float = 1e-4
    ) -> Dict[str, torch.Tensor]:
        """
        Continuous Normalizing Flow Matching Loss:
        Interpolates: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        Target velocity: u_t = x_1 - (1 - sigma_min) * x_0
        """
        batch, seq_len, dim = z_target.shape
        device = z_target.device
        
        t = torch.rand(batch, 1, 1, device=device)
        x_0 = torch.randn_like(z_target)
        x_1 = z_target
        
        x_t = (1.0 - (1.0 - sigma_min) * t) * x_0 + t * x_1
        target_v = x_1 - (1.0 - sigma_min) * x_0
        
        pred_v, _, _, _, _ = self.forward(x_t, u, expert_mask=expert_mask, tau_moe=tau_moe)
        flow_loss = F.mse_loss(pred_v, target_v)
        
        return {
            "flow_matching_loss": flow_loss,
            "pred_velocity": pred_v,
            "target_velocity": target_v
        }

    def forward(
        self, 
        z_seq: torch.Tensor, 
        u: torch.Tensor, 
        expert_mask: torch.Tensor, 
        tau_moe: torch.Tensor,
        drift_scale: float = 0.2,
        active_level: Optional[int] = None,
        return_stationarity: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        z_seq: (batch, seq_len, latent_dim)
        u: (batch, cond_dim)
        expert_mask: (batch, num_experts) from Meta-Controller
        tau_moe: (batch, 1) routing temperature from Meta-Controller
        active_level: Truncation depth k in [0, 4] for progressive multi-slice evaluation
        
        Returns:
            mu_next: (batch, seq_len, latent_dim)
            sigma_next: (batch, seq_len, latent_dim)
            moe_logits: (batch, num_experts)
            quality_score: (batch, 1) self-reported quality estimation
            log_var: (batch, seq_len, latent_dim) log variance
            (optional: stationarity_scores)
        """
        batch, seq_len, _ = z_seq.shape
        
        # Project inputs
        z_emb = self.latent_proj(z_seq, active_level=active_level)
        u_emb = self.cond_proj(u, active_level=active_level).unsqueeze(1)  # (batch, 1, d_model)
        h = z_emb + u_emb
        
        # Compute dynamic stationarity score for token dropping
        stat_scores = self.stationarity_gate(h)
        
        # Compute MoE router logits on the last latent state
        r_h = F.silu(self.router_linear1(h[:, -1, :], active_level=active_level))
        moe_logits = self.router_linear2(r_h, active_level=active_level)  # (batch, num_experts)
        
        # Smooth continuous routing softmax modulated by continuous expert_mask
        raw_weights = F.softmax(moe_logits / torch.clamp(tau_moe, min=0.05), dim=-1)
        gated_weights = raw_weights * expert_mask
        weights = gated_weights / (torch.sum(gated_weights, dim=-1, keepdim=True) + 1e-8)
        
        # Compute active expert trajectories and blend
        blended_h = torch.zeros_like(h)
        for k in range(self.num_experts):
            w_k = weights[:, k].view(batch, 1, 1)
            expert_out = self.experts[k](h, active_level=active_level)
            blended_h = blended_h + w_k * expert_out

        # Hybrid Jamba Attention: inject exact long-range associative memory
        blended_h = self.attention_fusion(blended_h)

        # Predict continuous Gaussian trajectory parameters
        mu = self.mu_head(blended_h, active_level=active_level)
        log_var = torch.clamp(self.log_var_head(blended_h, active_level=active_level), min=-20.0, max=20.0)
        
        # Upcast to float32 for exponential safety, then return to original dtype
        sigma = (torch.exp(0.5 * log_var.float()) * (1.0 + drift_scale * 2.0) + 1e-6).to(log_var.dtype)
        
        # Predict trajectory quality score from the latest blended state
        quality_score = self.quality_head(blended_h[:, -1, :])
        
        if return_stationarity:
            return mu, sigma, moe_logits, quality_score, log_var, stat_scores
            
        return mu, sigma, moe_logits, quality_score, log_var

