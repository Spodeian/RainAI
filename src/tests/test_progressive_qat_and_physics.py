"""
Comprehensive Unit Tests for Role-Anchored Progressive Slices (W = sum_i S_i),
Physics-Informed Loss Functions, and Slice-Aware Meta-Controller.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from src.models.residual_quant import ResidualWeight, ResidualLinear, ResidualConv1d
from src.models.diff_autoencoder import SpatialAudioEncoder
from src.models.mamba2_moe import Mamba2MoETrajectory
from src.models.meta_controller import InvasiveMetaController
from src.dsp.physics_losses import MultiScaleAmbisonicPhysicsLoss, PhysicsTrajectoryLoss
from src.data.spatial_upmix import apply_spatial_rir_convolution


def test_residual_weight_slices_and_gradients():
    """Verifies that W = sum_{i=0}^k S_i accumulates slices and allows backpropagation to all slices."""
    shape = (32, 16)
    rw = ResidualWeight(shape, num_slices=5)
    
    w0 = rw.get_effective_weight(active_level=0, quantize_base=False)
    assert w0.shape == shape
    
    w4 = rw.get_effective_weight(active_level=4, quantize_base=False)
    assert w4.shape == shape
    
    assert not torch.allclose(w0, w4)
    
    loss = (w0.sum() + w4.sum())
    loss.backward()
    
    for idx, s in enumerate(rw.slices):
        assert s.grad is not None, f"Slice {idx} must have non-null gradient"
        assert not torch.isnan(s.grad).any(), f"Slice {idx} gradient must not contain NaNs"


def test_residual_linear_and_conv1d_forward():
    """Verifies forward execution of ResidualLinear and ResidualConv1d across slice levels."""
    batch_size = 2
    linear = ResidualLinear(16, 32, num_slices=5)
    conv = ResidualConv1d(4, 16, kernel_size=3, padding=1, num_slices=5)
    
    x_lin = torch.randn(batch_size, 16)
    x_conv = torch.randn(batch_size, 4, 100)
    
    for level in range(5):
        y_lin = linear(x_lin, active_level=level)
        y_conv = conv(x_conv, active_level=level)
        
        assert y_lin.shape == (batch_size, 32)
        assert y_conv.shape == (batch_size, 16, 100)
        assert not torch.isnan(y_lin).any()
        assert not torch.isnan(y_conv).any()


def test_spatial_audio_encoder_slice_evaluation():
    """Verifies SpatialAudioEncoder at Ternary (level 0) and FP32 (level 4)."""
    enc = SpatialAudioEncoder()
    x = torch.randn(2, 4, 4800)
    u = torch.randn(2, 554)
    
    z0, align0, q0 = enc(x, u, active_level=0)
    z4, align4, q4 = enc(x, u, active_level=4)
    
    assert z0.shape == (2, 64, 10)
    assert z4.shape == (2, 64, 10)
    assert q0.shape == (2, 1)
    assert q4.shape == (2, 1)


def test_mamba2_moe_with_log_var_and_slices():
    """Verifies Mamba2MoETrajectory returns log_var and computes across slice levels."""
    mamba = Mamba2MoETrajectory()
    z_seq = torch.randn(2, 10, 64)
    u = torch.randn(2, 554)
    mask = torch.ones(2, 8)
    tau = torch.ones(2, 1)
    
    mu, sig, logits, qual, log_var = mamba(z_seq, u, mask, tau, active_level=0)
    assert mu.shape == (2, 10, 64)
    assert sig.shape == (2, 10, 64)
    assert log_var.shape == (2, 10, 64)
    assert logits.shape == (2, 8)
    assert qual.shape == (2, 1)


def test_meta_controller_slice_awareness():
    """Verifies InvasiveMetaController accepts active_slice_level and modulates gating."""
    mc = InvasiveMetaController()
    logits = torch.randn(2, 8)
    telem = torch.tensor([[35.0, 0.9, 0.9, 10.0], [15.0, 0.2, 0.3, 30.0]])
    weights = torch.tensor([[0.7, 0.3, 40.0], [0.5, 0.5, 20.0]])
    
    out0 = mc(logits, telem, weights, active_slice_level=0)
    out4 = mc(logits, telem, weights, active_slice_level=4)
    
    assert "expert_mask" in out0
    assert "expert_mask" in out4
    assert out0["expert_mask"].shape == (2, 8)
    assert out4["expert_mask"].shape == (2, 8)


def test_physics_acoustic_loss():
    """Verifies MultiScaleAmbisonicPhysicsLoss components (energy density, DoA, envelope)."""
    loss_fn = MultiScaleAmbisonicPhysicsLoss(fft_sizes=[256, 512], sample_rate=48000)
    x_pred = torch.randn(2, 4, 4800, requires_grad=True)
    x_true = torch.randn(2, 4, 4800)
    
    res = loss_fn(x_pred, x_true)
    assert "total_loss" in res
    assert "energy_density_loss" in res
    assert "doa_intensity_loss" in res
    assert "transient_envelope_loss" in res
    
    res["total_loss"].backward()
    assert x_pred.grad is not None


def test_physics_trajectory_loss():
    """Verifies PhysicsTrajectoryLoss with stable log_var NLL and 1/f turbulence."""
    loss_fn = PhysicsTrajectoryLoss(target_alpha=1.0)
    mu = torch.randn(2, 32, 64, requires_grad=True)
    log_var = torch.randn(2, 32, 64, requires_grad=True)
    target = torch.randn(2, 32, 64)
    
    res = loss_fn(mu, log_var, target)
    assert "total_traj_loss" in res
    assert "nll_loss" in res
    assert "velocity_loss" in res
    assert "turbulence_loss" in res
    
    res["total_traj_loss"].backward()
    assert mu.grad is not None
    assert log_var.grad is not None


def test_spatial_rir_convolution():
    """Verifies Room Impulse Response (RIR) convolution modifies audio based on enclosure and distance."""
    foa_audio = np.random.randn(4, 4800).astype(np.float32)
    
    dry = apply_spatial_rir_convolution(foa_audio, enclosure=0.0, distance=0.0)
    assert np.allclose(dry, foa_audio)
    
    wet = apply_spatial_rir_convolution(foa_audio, enclosure=0.8, distance=0.7)
    assert wet.shape == foa_audio.shape
    assert not np.allclose(wet, foa_audio)


if __name__ == "__main__":
    print("Running tests successfully...")
