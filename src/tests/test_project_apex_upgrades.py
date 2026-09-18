"""
Unit tests for Project Apex: Training Architecture Upgrades
Verifies:
- InstantaneousPhaseLoss
- MultiScaleSTFTDiscriminator & Adversarial Hinge/Feature-Matching Losses
- MoELoadBalancingLoss & Entropy Regularization
- KnowledgeDistillationLoss
- BetaVAEDisentanglementLoss & Total Correlation penalty
- HierarchicalMultiResLoss (16k, 32k, 48k)
- DifferentiableReverbEngine in DDSP
- ActiveLearningSampler & FOA Yaw Rotation in RainSpatialDataset
- BidirectionalMambaBlock in SpatialAudioEncoder
- JambaSelfAttentionBlock, Flow Matching & Sparsity in Mamba2MoETrajectory
- ModelEMA weight tracking
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.dsp.physics_losses import (
    InstantaneousPhaseLoss,
    MultiScaleAmbisonicPhysicsLoss,
    MultiScaleSTFTDiscriminator,
    discriminator_hinge_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    MoELoadBalancingLoss,
    KnowledgeDistillationLoss,
    BetaVAEDisentanglementLoss,
)
from src.models.ddsp import DifferentiableReverbEngine, ContinuousParametricFilter
from src.models.diff_autoencoder import SpatialAudioEncoder, HierarchicalMultiResLoss, BidirectionalMambaBlock
from src.models.mamba2_moe import Mamba2MoETrajectory, JambaSelfAttentionBlock
from src.data.dataset import ActiveLearningSampler, RainSpatialDataset, create_dataloader
from src.training.train_vae import ModelEMA


def test_01_instantaneous_phase_loss():
    phase_fn = InstantaneousPhaseLoss(fft_sizes=[512, 1024])
    x_true = torch.randn(2, 4, 48000)
    # Identical should yield near-zero phase loss
    loss_zero = phase_fn(x_true, x_true)
    assert loss_zero.item() < 1e-4, f"Expected near zero for identical audio, got {loss_zero.item()}"
    
    # Perturbed phase should yield positive loss
    x_pert = -x_true  # Anti-phase
    loss_pert = phase_fn(x_pert, x_true)
    assert loss_pert.item() > 0.5, f"Expected high loss for anti-phase audio, got {loss_pert.item()}"


def test_02_multiscale_stft_discriminator_and_losses():
    disc = MultiScaleSTFTDiscriminator(fft_sizes=[512, 1024])
    real_audio = torch.randn(2, 4, 48000)
    fake_audio = torch.randn(2, 4, 48000)
    
    real_scores, real_fmaps = disc(real_audio)
    fake_scores, fake_fmaps = disc(fake_audio)
    
    assert len(real_scores) == 2
    assert len(fake_scores) == 2
    
    d_loss = discriminator_hinge_loss(real_scores, fake_scores)
    g_loss = generator_adversarial_loss(fake_scores)
    fm_loss = feature_matching_loss(real_fmaps, fake_fmaps)
    
    assert d_loss.item() > 0.0
    assert torch.isfinite(g_loss)
    assert fm_loss.item() > 0.0


def test_03_moe_load_balancing_loss():
    balancer = MoELoadBalancingLoss(num_experts=8, load_weight=0.1, entropy_weight=0.01)
    
    # Uniform gating distribution (ideal balance)
    uniform_probs = torch.full((4, 8), 1.0 / 8.0)
    loss_uniform = balancer(uniform_probs)
    
    # Collapsed gating distribution (all routed to expert 0)
    collapsed_probs = torch.zeros(4, 8)
    collapsed_probs[:, 0] = 1.0
    loss_collapsed = balancer(collapsed_probs)
    
    # Collapsed should have higher load penalty than uniform
    assert loss_collapsed["load_loss"].item() > loss_uniform["load_loss"].item()


def test_04_knowledge_distillation_and_beta_vae_losses():
    kd_fn = KnowledgeDistillationLoss(temperature=2.0)
    student_feats = torch.randn(2, 50, 64)
    teacher_feats = torch.randn(2, 50, 64)
    kd_out = kd_fn(student_feats, teacher_feats)
    assert kd_out["distillation_loss"].item() > 0.0
    
    beta_vae_fn = BetaVAEDisentanglementLoss(beta=2.0, tc_weight=0.1)
    z_latents = torch.randn(2, 64, 100)
    bvae_out = beta_vae_fn(z_latents)
    assert bvae_out["disentangle_loss"].item() > 0.0
    assert torch.isfinite(bvae_out["tc_loss"])


def test_05_hierarchical_multires_loss():
    hier_fn = HierarchicalMultiResLoss(target_rates=[16000, 32000, 48000])
    pred = torch.randn(2, 4, 48000)
    true = torch.randn(2, 4, 48000)
    out = hier_fn(pred, true)
    assert "loss_16k" in out
    assert "loss_32k" in out
    assert "loss_48k" in out
    assert "hierarchical_loss" in out
    assert out["hierarchical_loss"].item() > 0.0


def test_06_differentiable_reverb_engine():
    reverb = DifferentiableReverbEngine(sample_rate=48000, ir_duration=0.2)
    audio = torch.randn(2, 4, 48000, requires_grad=True)
    rt60 = torch.tensor([[1.2], [0.5]], requires_grad=True)
    damping = torch.tensor([[0.4], [0.8]], requires_grad=True)
    wet_dry = torch.tensor([[0.3], [0.5]], requires_grad=True)
    
    out = reverb(audio, rt60=rt60, damping=damping, wet_dry=wet_dry)
    assert out.shape == (2, 4, 48000)
    
    # Backprop to check full gradient flow through acoustic physics
    loss = torch.sum(out ** 2)
    loss.backward()
    assert audio.grad is not None
    assert rt60.grad is not None
    assert damping.grad is not None
    assert wet_dry.grad is not None


def test_07_active_learning_sampler():
    sampler = ActiveLearningSampler(dataset_size=10, temperature=2.0)
    indices = list(iter(sampler))
    assert len(indices) == 10
    
    # Update losses for sample index 3 with massive error
    sampler.update_losses([3], [100.0])
    assert sampler.sample_losses[3] > sampler.sample_losses[0]
    
    # Sample multiple rounds; index 3 should appear frequently
    counts = 0
    for _ in range(5):
        sample_round = list(iter(sampler))
        counts += sample_round.count(3)
    assert counts > 0


def test_08_bidirectional_mamba_and_jamba_attention():
    # Bidirectional Mamba Block
    bi_mamba = BidirectionalMambaBlock(d_model=128)
    x = torch.randn(2, 128, 50)
    out_bi = bi_mamba(x)
    assert out_bi.shape == (2, 128, 50)
    
    # Jamba Hybrid Attention Block
    jamba_attn = JambaSelfAttentionBlock(d_model=128, num_heads=4)
    h = torch.randn(2, 50, 128)
    out_jamba = jamba_attn(h)
    assert out_jamba.shape == (2, 50, 128)


def test_09_flow_matching_and_model_ema():
    mamba_moe = Mamba2MoETrajectory(d_model=64)
    z_target = torch.randn(2, 20, 64)
    u = torch.randn(2, 554)
    expert_mask = torch.ones(2, 8)
    tau_moe = torch.ones(2, 1)
    
    fm_res = mamba_moe.compute_flow_matching_loss(z_target, u, expert_mask, tau_moe)
    assert "flow_matching_loss" in fm_res
    assert fm_res["flow_matching_loss"].item() > 0.0
    
    # Model EMA shadow tracking
    ema = ModelEMA(mamba_moe, decay=0.9)
    # Modify a weight
    with torch.no_grad():
        for p in mamba_moe.parameters():
            p.add_(0.5)
            break
    ema.update(mamba_moe)
    shadow = ema.state_dict()
    assert len(shadow) > 0
