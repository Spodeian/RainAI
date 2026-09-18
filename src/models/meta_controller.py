"""
Invasive Global Meta-Controller: Dynamically supervises MoE routing,
ambisonic order, context pruning, diffusion bypass, and progressive quantization awareness
based on audio buffer health and active weight slice level.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvasiveMetaController(nn.Module):
    """
    Lightweight Invasive Meta-Controller network evaluating in O(1) time.
    Features:
        - LayerNorm stabilization across disparate metric domains (ms, logits, ratios)
        - 1D State Space (Mamba/SSD) Telemetry Tracker: maintains infinite-horizon memory
          of hardware drift, OS jitter, and thermal creep.
        - Bifurcated Telemetry: distinguishes CPU bottlenecks from GPU/VRAM starvation.
        - Progressive Slice Awareness: adjusts expert gating and routing temperature
          based on active weight slice level k in [0, 4] (Ternary -> FP32).
    Inputs:
        - moe_logits: (batch, num_experts)
        - telemetry: (batch, 4) [buffer_health_ms, cpu_headroom, gpu_headroom, delta_t_ms]
          OR buffer_health_ms: (batch, 1) (backward-compatible fallback)
        - user_weights: (batch, 3) [quality_pref, perf_pref, target_buffer_ms]
        - quality_scores: (batch, 2) [encoder_quality, mamba_quality] from shared-weight critics
        - active_slice_level: (batch, 1) or scalar k in [0, 4]
    Outputs:
        - expert_mask: (batch, num_experts) continuous mask in [0, 1]
        - tau_moe: (batch, 1) continuous temperature in [0.1, 2.0]
        - ambisonic_order: (batch, 1) discrete {0, 1} (Order 0 = Mono, Order 1 = FOA)
        - diffusion_bypass: (batch, 1) continuous factor in [0, 1]
        - synthesis_blend: (batch, 1) continuous [0, 1] blend (0 = neural, 1 = procedural/DDSP fallback)
        - panic_factor: (batch, 1)
        - quality_drop: (batch, 1)
        - jitter_factor: (batch, 1)
    """
    def __init__(self, num_experts: int = 8, d_state: int = 16):
        super().__init__()
        self.num_experts = num_experts
        self.d_state = d_state
        
        # Total raw input features: num_experts (8) + telemetry (4) + user_weights (3) + quality_scores (2) + active_slice (1) = 18
        in_dim = num_experts + 4 + 3 + 2 + 1
        
        # Affine LayerNorms for numerical stability across heterogeneous metrics
        self.input_norm = nn.LayerNorm(in_dim)
        
        # Miniature 1D State Space (Mamba/SSD) block for temporal trend tracking
        self.telemetry_proj = nn.Linear(4, 32)
        self.A_log = nn.Parameter(torch.log(torch.rand(32, d_state) + 0.5))
        self.B_proj = nn.Linear(32, d_state)
        self.C_proj = nn.Linear(32, d_state)
        self.D_param = nn.Parameter(torch.ones(32))
        
        # Feature fusion and MLP policy heads
        self.fusion = nn.Sequential(
            nn.Linear(in_dim + 32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, num_experts + 5)
        )

    def forward(
        self, 
        moe_logits: torch.Tensor, 
        buffer_or_telemetry: torch.Tensor, 
        user_weights: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        telemetry_state: Optional[torch.Tensor] = None,
        active_slice_level: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        moe_logits: (batch, num_experts)
        buffer_or_telemetry: (batch, 4) [buffer_health_ms, cpu_headroom, gpu_headroom, delta_t_ms]
                             or (batch, 1) buffer_health_ms
        user_weights: (batch, 3) [quality_vs_perf, buffer_target, battery_saver]
        quality_scores: (batch, 2) optional self-reported quality scores from sub-networks
        telemetry_state: (batch, 32, d_state) optional recurrent SSM hidden state
        active_slice_level: (batch, 1) or scalar in [0, 4] indicating active quantization tier
        """
        batch_size = moe_logits.shape[0]
        device = moe_logits.device
        
        # Format telemetry tensor (batch, 4)
        if buffer_or_telemetry.shape[-1] == 1:
            buf = buffer_or_telemetry
            cpu_h = torch.ones_like(buf)
            gpu_h = torch.ones_like(buf)
            dt_ms = torch.full_like(buf, 10.0)
            telemetry = torch.cat([buf, cpu_h, gpu_h, dt_ms], dim=-1)
        else:
            telemetry = buffer_or_telemetry
            
        buffer_health_ms = telemetry[:, 0:1]
        cpu_headroom = telemetry[:, 1:2]
        gpu_headroom = telemetry[:, 2:3]
        delta_t_ms = telemetry[:, 3:4]
        
        if quality_scores is None:
            quality_scores = torch.ones(batch_size, 2, device=device)
            
        if active_slice_level is None:
            slice_norm = torch.ones(batch_size, 1, device=device)  # Default: full precision (level 4)
        elif isinstance(active_slice_level, (int, float)):
            slice_norm = torch.full((batch_size, 1), float(active_slice_level) / 4.0, device=device)
        else:
            slice_norm = (active_slice_level.float() / 4.0).clamp(0.0, 1.0)
            
            # Catch 0-D scalar tensors and broadcast to batch size
            if slice_norm.dim() == 0:
                slice_norm = slice_norm.view(1, 1).expand(batch_size, 1)
            # Catch 1-D tensors and append the feature dimension
            elif slice_norm.dim() == 1:
                slice_norm = slice_norm.unsqueeze(-1)
            
        # Normalize inputs into stable dynamic range
        normalized_telemetry = torch.cat([
            buffer_health_ms / 100.0,
            cpu_headroom,
            gpu_headroom,
            delta_t_ms / 50.0
        ], dim=-1)
        
        raw_features = torch.cat([moe_logits, normalized_telemetry, user_weights, quality_scores, slice_norm], dim=-1)
        normed_features = self.input_norm(raw_features)
        
        # Miniature 1D Mamba / State Space recurrence over hardware telemetry
        u_telem = F.silu(self.telemetry_proj(normalized_telemetry))
        b_telem = self.B_proj(u_telem).unsqueeze(1)
        c_telem = self.C_proj(u_telem).unsqueeze(1)
        
        A_log_safe = torch.clamp(self.A_log, max=8.0)
        decay = torch.exp(-torch.exp(A_log_safe)).unsqueeze(0)
        if telemetry_state is None:
            telemetry_state = torch.zeros(batch_size, 32, self.d_state, device=device)
            
        new_state = decay * telemetry_state + (u_telem.unsqueeze(-1) * b_telem)
        mamba_out = torch.sum(new_state * c_telem, dim=-1) + self.D_param * u_telem
        
        # Joint fusion
        fused = torch.cat([normed_features, mamba_out], dim=-1)
        raw_out = self.fusion(fused)
        
        # 1. Stress & Panic Mechanics
        buffer_panic = torch.clamp((25.0 - buffer_health_ms) / 25.0, min=0.0, max=1.0)
        jitter = torch.clamp((delta_t_ms - 15.0) / 20.0, min=0.0, max=1.0)
        hw_starvation = torch.clamp(2.0 - (cpu_headroom + gpu_headroom), min=0.0, max=1.0)
        mean_quality = quality_scores.mean(dim=-1, keepdim=True)
        quality_drop = 1.0 - mean_quality
        
        stress = torch.clamp(buffer_panic * 0.4 + jitter * 0.25 + hw_starvation * 0.2 + quality_drop * 0.15, min=0.0, max=1.0)
        
        # 2. Expert Activation Gating modulated by active slice level
        # At lower precision tiers (e.g. Ternary slice 0), FLOP cost per expert is lower,
        # but capacity is tighter, so meta-controller allows more active experts under low buffer stress.
        expert_gates = torch.sigmoid(raw_out[:, :self.num_experts])
        dynamic_threshold = 0.4 + 0.45 * torch.max(stress, 1.0 - gpu_headroom) * (0.5 + 0.5 * slice_norm)
        tau_gate = 0.15
        
        raw_mask = torch.sigmoid((expert_gates - dynamic_threshold) / tau_gate)
        mamba_damping = torch.sigmoid(mamba_out[:, :self.num_experts] * 0.2)
        expert_mask = 0.85 * raw_mask + 0.15 * mamba_damping
        
        # Primary expert always active
        expert_mask = torch.cat([torch.clamp(expert_mask[:, :1], min=0.85), expert_mask[:, 1:]], dim=-1)

        # 3. Routing Temperature tau_moe in [0.1, 2.0]
        raw_tau = torch.sigmoid(raw_out[:, self.num_experts:self.num_experts+1])
        tau_moe = 0.1 + 1.9 * raw_tau * (1.0 - 0.7 * stress)

        # 4. Ambisonic Order Gate (FOA vs Mono)
        ambisonic_gate = torch.sigmoid(raw_out[:, self.num_experts+1:self.num_experts+2])
        ambisonic_order = (ambisonic_gate > (0.5 + 0.4 * stress)).long()

        # 5. Diffusion Crossfade / Bypass Factor
        raw_diff = torch.sigmoid(raw_out[:, self.num_experts+2:self.num_experts+3])
        diffusion_bypass = torch.sigmoid((raw_diff - (0.5 + 0.4 * stress)) / 0.1)
        
        # 6. Synthesis Blend Factor
        raw_blend = torch.sigmoid(raw_out[:, self.num_experts+3:self.num_experts+4])
        synthesis_blend = torch.clamp(raw_blend + stress, min=0.0, max=1.0)
        
        # 7. Recommended Pre-Generated / Thinking Steps in [1.0, 5.0]
        raw_steps = torch.sigmoid(raw_out[:, self.num_experts+4:self.num_experts+5])
        pre_generated_steps = 1.0 + 4.0 * raw_steps * (1.0 - 0.75 * stress)

        # 8. Recommended Dynamic Safety Buffer Reserve (in ms)
        safety_reserve_ms = 15.0 + 25.0 * jitter + 15.0 * hw_starvation

        return {
            "expert_mask": expert_mask,
            "tau_moe": tau_moe,
            "ambisonic_order": ambisonic_order,
            "diffusion_bypass": diffusion_bypass,
            "synthesis_blend": synthesis_blend,
            "pre_generated_steps": pre_generated_steps,
            "panic_factor": buffer_panic,
            "quality_drop": quality_drop,
            "jitter_factor": jitter,
            "safety_reserve_ms": safety_reserve_ms,
            "telemetry_state": new_state
        }
