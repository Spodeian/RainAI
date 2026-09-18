"""
Mamba-2 State Space Duality (SSD) Continuous Latent Trajectory Model (Hardened).
Incorporates DeepSeek-inspired efficiency techniques:
- Multi-Head Latent Attention (MLA) with low-rank KV compression
- Engram Conditional Memory Bank (O(1) physical prior lookup)
- Conditioning Compression Gateway (temporal caching without MoE starvation)
- Auxiliary-Loss-Free Router with detached external load balancing
- DeepSeek-style Invariant Shared Base Expert + Additive Residual Experts
- Router-Derived Dynamic Dense Soup of Experts with deficit distillation loss
"""

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.residual_quant import ResidualLinear, ResidualConv1d
from src.models.engram import EngramBank

LATENT_DIM = 64
CONDITION_DIM = 554
NUM_EXPERTS = 8


class MambaSSDBlock(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = ResidualLinear(d_model, d_model * 2)
        self.conv1d = ResidualConv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 0.5))
        self.B_proj = ResidualLinear(d_model, d_state)
        self.C_proj = ResidualLinear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = ResidualLinear(d_model, d_model)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        proj = self.in_proj(x, active_level=active_level)
        u, gate = proj.chunk(2, dim=-1)
        
        u_conv = self.conv1d(u.transpose(1, 2), active_level=active_level)[:, :, :seq_len].transpose(1, 2)
        u_act = F.silu(u_conv)
        
        B = self.B_proj(u_act, active_level=active_level)
        C = self.C_proj(u_act, active_level=active_level)
        
        A_log_safe = torch.clamp(self.A_log, max=6.0) # Tighter cap to prevent exp overflow
        decay = torch.exp(-torch.exp(A_log_safe)).unsqueeze(0)
        
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        ys = []
        for t in range(seq_len):
            u_t = u_act[:, t, :].unsqueeze(-1)
            b_t = B[:, t, :].unsqueeze(1)
            c_t = C[:, t, :].unsqueeze(1)
            
            # Stable recurrence with strict state magnitude clamping
            h = decay * h + (u_t * b_t)
            h = torch.clamp(h, min=-1e4, max=1e4)
            
            y_t = torch.sum(h * c_t, dim=-1) + self.D * u_act[:, t, :]
            y_t = torch.nan_to_num(y_t, nan=0.0, posinf=1e4, neginf=-1e4)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1)
        y = y * F.silu(gate)
        return self.out_proj(y, active_level=active_level)


class LatentAttentionBlock(nn.Module):
    """
    Multi-Head Latent Attention (MLA) — DeepSeek-style low-rank KV compression.
    Instead of caching full K, V in R^{d x n_heads}, compresses into a low-rank
    latent c_kv in R^{d_compress} and reconstructs K, V on the fly.
    """
    def __init__(self, d_model: int = 128, num_heads: int = 4, d_compress: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_compress = d_compress
        self.head_dim = d_model // num_heads

        self.q_proj = ResidualLinear(d_model, d_model)
        # KV compression: x -> c_kv
        self.kv_down = ResidualLinear(d_model, d_compress)
        # KV reconstruction: c_kv -> K, V
        self.k_up = ResidualLinear(d_compress, d_model)
        self.v_up = ResidualLinear(d_compress, d_model)

        self.out_proj = ResidualLinear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x, active_level=active_level).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Low-rank KV compression
        c_kv = self.kv_down(x, active_level=active_level)

        # On-the-fly reconstruction
        k = self.k_up(c_kv, active_level=active_level).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(c_kv, active_level=active_level).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        out = self.out_proj(attn_out, active_level=active_level)
        return self.norm(x + out)


# Backwards-compatible alias for existing test suites
JambaSelfAttentionBlock = LatentAttentionBlock


class ConditioningGateway(nn.Module):
    """
    Conditioning Compression Gateway (CED-inspired, non-splitting).
    Compresses u in R^554 -> c_ctx in R^d_model (128) with inference caching.
    """
    def __init__(self, cond_dim: int = CONDITION_DIM, d_model: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.fc1 = ResidualLinear(cond_dim, 256)
        self.fc2 = ResidualLinear(256, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.register_buffer("cached_ctx", torch.zeros(1, d_model))
        self.register_buffer("cached_u", torch.zeros(1, cond_dim))
        self.register_buffer("cache_valid", torch.zeros(1, dtype=torch.int64))

    def forward(self, u: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        batch = u.shape[0]
        # Short-circuit JIT tracing, ONNX export, or training to avoid tensor boolean evaluation
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or self.training or u.shape[0] > 1:
            h1 = F.silu(self.fc1(u, active_level=active_level))
            return self.norm(self.fc2(h1, active_level=active_level))

        # Check if cache is initialized
        if self.cache_valid.item() > 0:
            cos_sim = F.cosine_similarity(u, self.cached_u, dim=-1)
            if bool((cos_sim > 0.99).all().item()):
                return self.cached_ctx.expand(batch, -1)

        h1 = F.silu(self.fc1(u, active_level=active_level))
        ctx = self.norm(self.fc2(h1, active_level=active_level))
        self.cached_ctx.copy_(ctx.detach())
        self.cached_u.copy_(u.detach())
        self.cache_valid.copy_(torch.ones(1, dtype=torch.int64, device=u.device))
        return ctx



class AuxFreeRouter(nn.Module):
    """
    Auxiliary-Loss-Free Router with detached external load balancing bias.
    Avoids competing auxiliary loss objectives by updating load bias directly outside autograd.
    """
    def __init__(self, d_model: int, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.num_experts = num_experts
        self.linear1 = ResidualLinear(d_model, 64)
        self.linear2 = ResidualLinear(64, num_experts)
        self.register_buffer("load_bias", torch.zeros(num_experts))
        self.bias_update_rate = 0.001

    def forward(self, h_last: torch.Tensor, active_level: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r_h = F.silu(self.linear1(h_last, active_level=active_level))
        raw_logits = self.linear2(r_h, active_level=active_level)
        routing_logits = raw_logits + self.load_bias.detach().unsqueeze(0)
        return routing_logits, raw_logits

    @torch.no_grad()
    def update_load_bias(self, routing_probs: torch.Tensor):
        mean_load = routing_probs.mean(dim=0)
        target_load = 1.0 / self.num_experts
        imbalance = mean_load - target_load
        self.load_bias -= self.bias_update_rate * imbalance


class Mamba2MoETrajectory(nn.Module):
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
        
        self.latent_proj = ResidualLinear(latent_dim, d_model)
        self.cond_proj = ResidualLinear(cond_dim, d_model)
        self.gateway = ConditioningGateway(cond_dim=cond_dim, d_model=d_model)
        
        self.router = AuxFreeRouter(d_model=d_model, num_experts=num_experts)
        # Compatibility aliases for legacy state dictionaries
        self.router_linear1 = self.router.linear1
        self.router_linear2 = self.router.linear2
        
        # DeepSeek-style invariant shared base expert
        self.shared_base_expert = MambaSSDBlock(d_model=d_model)
        # Residual specialty experts (displacement vectors Delta W_k)
        self.residual_experts = nn.ModuleList([MambaSSDBlock(d_model=d_model) for _ in range(num_experts)])
        self.experts = self.residual_experts  # Compatibility alias
        
        # O(1) hash-addressed Engram knowledge bank
        self.engram_bank = EngramBank(bank_size=32768, embed_dim=d_model)
        
        # Router -> dense soup coefficient projection
        self.soup_proj = nn.Sequential(
            nn.Linear(num_experts, 32),
            nn.SiLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.attention_fusion = LatentAttentionBlock(d_model=d_model, num_heads=4, d_compress=32)
        
        self.stationarity_gate = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.mu_head = ResidualLinear(d_model, latent_dim)
        self.log_var_head = ResidualLinear(d_model, latent_dim)
        
        self.quality_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def get_activation_sparsity_loss(self) -> torch.Tensor:
        l1_reg = torch.mean(torch.abs(self.shared_base_expert.D))
        for expert in self.residual_experts:
            l1_reg = l1_reg + torch.mean(torch.abs(expert.D))
        return 0.001 * l1_reg

    def compute_dense_soup_blend(
        self,
        h: torch.Tensor,
        soup_alpha: torch.Tensor,
        active_level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes the dense soup forward representation:
        W_dense = W_base + sum alpha_k * Delta W_k
        In PyTorch, we evaluate this as base_out + sum alpha_k * residual_out
        so gradients propagate cleanly into the soup projection and residual experts.
        """
        batch = h.shape[0]
        base_out = self.shared_base_expert(h, active_level=active_level)
        soup_residual = torch.zeros_like(base_out)
        for k in range(self.num_experts):
            alpha_k = soup_alpha[:, k].view(batch, 1, 1)
            exp_out = self.residual_experts[k](h, active_level=active_level)
            soup_residual = soup_residual + alpha_k * exp_out
        return base_out + soup_residual

    def compute_soup_deficit_loss(
        self,
        z_seq: torch.Tensor,
        u: torch.Tensor,
        expert_mask: torch.Tensor,
        tau_moe: torch.Tensor,
        active_level: Optional[int] = None
    ) -> torch.Tensor:
        """
        Calculates L_soup_deficit = || y_dense_soup - stop_gradient(y_sparse_moe) ||^2.
        Trains dense soup parameters to faithfully match sparse MoE output.
        """
        batch, seq_len, _ = z_seq.shape
        z_emb = self.latent_proj(z_seq, active_level=active_level)
        u_emb = self.cond_proj(u, active_level=active_level).unsqueeze(1)
        u_ctx = self.gateway(u, active_level=active_level).unsqueeze(1)
        h = z_emb + u_emb + u_ctx
        h, _ = self.engram_bank(h)

        routing_logits, _ = self.router(h[:, -1, :], active_level=active_level)
        raw_weights = F.softmax(routing_logits / torch.clamp(tau_moe, min=0.05), dim=-1)
        gated_weights = raw_weights * expert_mask
        sparse_weights = gated_weights / (torch.sum(gated_weights, dim=-1, keepdim=True) + 1e-8)

        # Sparse MoE teacher output
        base_out = self.shared_base_expert(h, active_level=active_level)
        sparse_residual = torch.zeros_like(base_out)
        for k in range(self.num_experts):
            w_k = sparse_weights[:, k].view(batch, 1, 1)
            sparse_residual = sparse_residual + w_k * self.residual_experts[k](h, active_level=active_level)
        y_sparse = base_out + sparse_residual

        # Dense soup student output
        soup_alpha = self.soup_proj(sparse_weights.detach())
        y_soup = self.compute_dense_soup_blend(h, soup_alpha, active_level=active_level)

        return F.mse_loss(y_soup, y_sparse.detach())

    def compute_flow_matching_loss(
        self,
        z_target: torch.Tensor,
        u: torch.Tensor,
        expert_mask: torch.Tensor,
        tau_moe: torch.Tensor,
        sigma_min: float = 1e-4
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len, dim = z_target.shape
        device = z_target.device
        
        t = torch.rand(batch, 1, 1, device=device)
        x_0 = torch.randn_like(z_target)
        x_1 = z_target
        
        x_t = (1.0 - (1.0 - sigma_min) * t) * x_0 + t * x_1
        target_v = x_1 - (1.0 - sigma_min) * x_0
        
        pred_v, _, _, _, _ = self.forward(x_t, u, expert_mask=expert_mask, tau_moe=tau_moe)
        flow_loss = F.mse_loss(pred_v, target_v)
        flow_loss = torch.nan_to_num(flow_loss, nan=1.0, posinf=50.0, neginf=0.0)
        
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
        return_stationarity: bool = False,
        return_soup_output: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        batch, seq_len, _ = z_seq.shape
        
        z_emb = self.latent_proj(z_seq, active_level=active_level)
        u_emb = self.cond_proj(u, active_level=active_level).unsqueeze(1)
        u_ctx = self.gateway(u, active_level=active_level).unsqueeze(1)
        h = z_emb + u_emb + u_ctx
        
        # Engram physical prior lookup
        h, _ = self.engram_bank(h)
        
        stat_scores = self.stationarity_gate(h)
        
        routing_logits, raw_logits = self.router(h[:, -1, :], active_level=active_level)
        moe_logits = raw_logits
        
        raw_weights = F.softmax(routing_logits / torch.clamp(tau_moe, min=0.05), dim=-1)
        gated_weights = raw_weights * expert_mask
        weights = gated_weights / (torch.sum(gated_weights, dim=-1, keepdim=True) + 1e-8)
        
        # Auxiliary-loss-free bias tracking during training
        if self.training:
            self.router.update_load_bias(weights.detach())
        
        # DeepSeek-style Shared Base Expert + Sparse Residual Blend
        base_out = self.shared_base_expert(h, active_level=active_level)
        residual_blend = torch.zeros_like(base_out)
        for k in range(self.num_experts):
            w_k = weights[:, k].view(batch, 1, 1)
            expert_out = self.residual_experts[k](h, active_level=active_level)
            residual_blend = residual_blend + w_k * expert_out

        blended_h = base_out + residual_blend
        blended_h = self.attention_fusion(blended_h, active_level=active_level)

        mu = self.mu_head(blended_h, active_level=active_level)
        log_var = torch.clamp(self.log_var_head(blended_h, active_level=active_level), min=-15.0, max=10.0)
        
        sigma = (torch.exp(0.5 * log_var.float()) * (1.0 + drift_scale * 2.0) + 1e-6).to(log_var.dtype)
        quality_score = self.quality_head(blended_h[:, -1, :])

        if bool(return_soup_output):
            soup_alpha = self.soup_proj(weights.detach())
            soup_h = self.compute_dense_soup_blend(h, soup_alpha, active_level=active_level)
            soup_h = self.attention_fusion(soup_h, active_level=active_level)
            mu_soup = self.mu_head(soup_h, active_level=active_level)
            if bool(return_stationarity):
                return mu, sigma, moe_logits, quality_score, log_var, stat_scores, mu_soup
            return mu, sigma, moe_logits, quality_score, log_var, mu_soup
        
        # Ensure the condition strictly evaluates as a primitive Python boolean
        if bool(return_stationarity):
            return mu, sigma, moe_logits, quality_score, log_var, stat_scores
            
        return mu, sigma, moe_logits, quality_score, log_var

