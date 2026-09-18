"""
Residual Multi-Slice Quantization Modules for RainAI.
Implements weight parameterization as a sum of discrete residual slices:
    W = sum_{i=0}^{N-1} S_i

Allows dynamic evaluation at any slice truncation depth k in [0, N-1],
enabling simultaneous training aligned with Rust engine geometric midpoints:
    - S0: Ternary 1.58b ([0.5, 1.807) bits) - Base Structural Scaffold
    - S1: Coarse INT4/INT5 ([3.585, 5.585) bits) - Magnitude Envelope
    - S2: Standard Posit8 / INT8 ([2.585, 7.585) bits) - Tapered Unum / Linear Fine Texture
    - S3: High Dynamic Range BF16 / Posit16 ([7.585, 16.585) bits) - Wide Dynamic Exponent
    - S4: Master Studio FP32 (>= 16.585 bits) - IEEE-754 Full Gradient Continuum
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_DEFAULT_SLICES = 5  # S0: Ternary158, S1: INT4/INT5, S2: Posit8/INT8, S3: BF16/Posit16, S4: Studio FP32


def smooth_round(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    O(1) Continuous differentiable relaxation of torch.round().
    Replaces hard Straight-Through Estimators (STE) with a mathematically smooth staircase.
    f(x) = floor(x) + sigmoid((x - floor(x) - 0.5) / tau)
    """
    floor_x = torch.floor(x)
    rem = x - floor_x
    return floor_x + torch.sigmoid((rem - 0.5) / tau)


class ResidualWeight(nn.Module):
    """
    Decomposes a weight tensor into N residual slices: W = sum_{i=0}^{N-1} S_i.
    Uses continuous sigmoid relaxations for quantization on base slices
    and continuous residual learning on higher slices.
    """
    def __init__(self, shape: Tuple[int, ...], num_slices: int = NUM_DEFAULT_SLICES):
        super().__init__()
        self.shape = shape
        self.num_slices = num_slices
        
        # S0: Base structural scaffold (initialized with standard Kaiming/Xavier scale)
        # S1..S_{N-1}: Progressively smaller initial residuals
        slices = []
        fan_in = shape[1] if len(shape) >= 2 else shape[0]
        std = (2.0 / max(fan_in, 1)) ** 0.5
        
        # Base slice S0 holds bulk of energy
        s0 = nn.Parameter(torch.randn(*shape) * std * 0.7)
        slices.append(s0)
        
        # Subsequent residual slices hold progressively finer detail
        for i in range(1, num_slices):
            decay = 0.5 ** i
            si = nn.Parameter(torch.randn(*shape) * std * decay * 0.3)
            slices.append(si)
            
        self.slices = nn.ParameterList(slices)
        
        # Learnable scale for S0 ternary mapping
        self.gamma_scale = nn.Parameter(torch.tensor(1.0))
        
        # Learnable temperature for the smooth quantization staircase
        # Initialize at sig(-2.0) ≈ 0.11
        self.tau_raw = nn.Parameter(torch.tensor(-2.0))

    @property
    def tau(self) -> torch.Tensor:
        """Bounded temperature [0.01, 1.0]. Approaches hard staircase as tau -> 0.01."""
        return 0.01 + 0.99 * torch.sigmoid(self.tau_raw)

    def get_effective_weight(self, active_level: Optional[int] = None, quantize_base: bool = True) -> torch.Tensor:
        """
        Computes W_eff = sum_{i=0}^{k} S_i where k = active_level (clamped to [0, num_slices-1]).
        If active_level is None, evaluates the full unquantized sum over all slices.
        """
        if active_level is None:
            max_k = self.num_slices - 1
        else:
            max_k = min(max(int(active_level), 0), self.num_slices - 1)
            
        # S0 processing:
        s0 = self.slices[0]
        if quantize_base and max_k == 0:
            # Smooth Ternary 1.58b mapping
            gamma = (s0.abs().mean() * self.gamma_scale.abs()).clamp(min=1e-6)
            s0_scaled = s0 / gamma
            
            s0_ternary = smooth_round(s0_scaled, self.tau).clamp(-1.0, 1.0)
            w_acc = s0_ternary * gamma
        else:
            w_acc = s0
            
        # Accumulate residual slices up to max_k
        if not quantize_base and max_k > 0:
            # Fast vectorized accumulation for evaluation / continuous layers
            slice_stack = torch.stack(list(self.slices)[1:max_k + 1])
            w_acc = w_acc + slice_stack.sum(dim=0)
        else:
            # Iterative continuous accumulation for active quantization training
            for i in range(1, max_k + 1):
                si = self.slices[i]
                if quantize_base and i == 1 and max_k == 1:
                    # S1: Smooth Coarse INT4 simulation
                    scale_4 = (si.abs().max() / 7.0).clamp(min=1e-6)
                    si_q = smooth_round(si / scale_4, self.tau).clamp(-8.0, 7.0) * scale_4
                    w_acc = w_acc + si_q
                elif quantize_base and i == 2 and max_k == 2:
                    # S2: Smooth INT8 simulation
                    scale_8 = (si.abs().max() / 127.0).clamp(min=1e-6)
                    si_q = smooth_round(si / scale_8, self.tau).clamp(-128.0, 127.0) * scale_8
                    w_acc = w_acc + si_q
                else:
                    w_acc = w_acc + si
                    
        return w_acc

    def load_state_dict(self, state_dict, strict=True):
        """Custom state_dict loader to support legacy flat weight checkpoints."""
        if "weight" in state_dict and "slices.0" not in state_dict:
            # Fallback for old checkpoints: map legacy flat weight into base slice S0
            self.slices[0].data.copy_(state_dict["weight"])
            return
        super().load_state_dict(state_dict, strict=strict)


class ResidualLinear(nn.Module):
    """
    Linear layer with weight matrix parameterized as a sum of residual slices W = sum_i S_i.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_slices: int = NUM_DEFAULT_SLICES,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_slices = num_slices
        
        self.weight_res = ResidualWeight((out_features, in_features), num_slices=num_slices)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        w = self.weight_res.get_effective_weight(active_level=active_level, quantize_base=self.training)
        return F.linear(x, w, self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Full uncompressed weight property for inspection and compatibility."""
        return self.weight_res.get_effective_weight(active_level=None, quantize_base=False)


class ResidualConv1d(nn.Module):
    """
    1D Convolutional layer with weight kernel parameterized as a sum of residual slices.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        num_slices: int = NUM_DEFAULT_SLICES,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_slices = num_slices
        
        shape = (out_channels, in_channels // groups, kernel_size)
        self.weight_res = ResidualWeight(shape, num_slices=num_slices)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, active_level: Optional[int] = None) -> torch.Tensor:
        w = self.weight_res.get_effective_weight(active_level=active_level, quantize_base=self.training)
        return F.conv1d(
            x, w, self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )

    @property
    def weight(self) -> torch.Tensor:
        """Full uncompressed weight property for inspection and compatibility."""
        return self.weight_res.get_effective_weight(active_level=None, quantize_base=False)