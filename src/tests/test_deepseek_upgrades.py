"""
Unit tests for DeepSeek-inspired efficiency upgrades:
1. EngramBank deterministic lookup, collision behavior, and gated residual fusion
2. LatentAttentionBlock (MLA) low-rank KV compression
3. ConditioningGateway caching behavior
4. AuxFreeRouter detached external load balancing
5. Mamba2MoETrajectory with shared base expert, dense soup blend, and deficit loss
"""

import torch
import torch.nn.functional as F
import pytest

from src.models.engram import EngramBank
from src.models.mamba2_moe import (
    LatentAttentionBlock,
    ConditioningGateway,
    AuxFreeRouter,
    Mamba2MoETrajectory,
)


def test_01_engram_bank_deterministic_lookup():
    """Verify EngramBank produces deterministic O(1) hash indices and fused output."""
    bank = EngramBank(bank_size=32768, embed_dim=64, num_hash_heads=4)
    x = torch.randn(2, 10, 64)
    
    indices1 = bank.compute_hash_indices(x)
    indices2 = bank.compute_hash_indices(x)
    assert torch.equal(indices1, indices2), "Engram hash indices must be strictly deterministic"
    assert indices1.shape == (2, 10, 4)
    assert torch.all(indices1 >= 0) and torch.all(indices1 < 32768)

    fused, gate = bank(x)
    assert fused.shape == (2, 10, 64)
    assert gate.shape == (2, 10, 64)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_02_mla_latent_attention_compression():
    """Verify Multi-Head Latent Attention compresses KV and maintains valid output."""
    mla = LatentAttentionBlock(d_model=128, num_heads=4, d_compress=32)
    x = torch.randn(2, 16, 128)
    
    # Check intermediate KV compression shape
    c_kv = mla.kv_down(x)
    assert c_kv.shape == (2, 16, 32), f"Expected compressed KV shape (2, 16, 32), got {c_kv.shape}"
    
    out = mla(x)
    assert out.shape == (2, 16, 128)
    assert torch.all(torch.isfinite(out))


def test_03_conditioning_gateway_caching():
    """Verify ConditioningGateway caches context tokens when conditioning is static."""
    gateway = ConditioningGateway(cond_dim=554, d_model=128)
    gateway.eval()  # Eval mode to test inference caching
    
    u1 = torch.randn(1, 554)
    out1 = gateway(u1)
    assert bool(gateway.cache_valid.item()) is True

    
    # Send identical conditioning - should return cached representation
    out1_cached = gateway(u1)
    assert torch.equal(out1, out1_cached)

    # Perturb slightly (< 1% difference) - cosine similarity > 0.99
    u1_perturbed = u1 + 0.001 * torch.randn_like(u1)
    out1_perturbed = gateway(u1_perturbed)
    assert torch.equal(out1, out1_perturbed)

    # Completely different conditioning - should update cache
    u2 = -u1
    out2 = gateway(u2)
    assert not torch.equal(out1, out2)


def test_04_aux_free_router_detached_bias():
    """Verify AuxFreeRouter updates load bias without leaking into the computational graph."""
    router = AuxFreeRouter(d_model=128, num_experts=8)
    h = torch.randn(4, 128, requires_grad=True)
    
    routing_logits, raw_logits = router(h)
    assert routing_logits.shape == (4, 8)
    assert raw_logits.shape == (4, 8)
    
    loss = torch.sum(raw_logits ** 2)
    loss.backward()
    
    # Verify load_bias has NO grad
    assert router.load_bias.grad is None
    
    # Update bias externally
    probs = F.softmax(routing_logits.detach(), dim=-1)
    prev_bias = router.load_bias.clone()
    router.update_load_bias(probs)
    assert not torch.equal(prev_bias, router.load_bias)


def test_05_mamba2_moe_shared_base_and_dense_soup():
    """Verify Mamba2MoETrajectory shared base expert, soup output, and deficit loss."""
    mamba = Mamba2MoETrajectory(d_model=64)
    z_seq = torch.randn(2, 8, 64)
    u = torch.randn(2, 554)
    expert_mask = torch.ones(2, 8)
    tau_moe = torch.ones(2, 1)

    # Check forward pass with soup output requested
    res = mamba(z_seq, u, expert_mask, tau_moe, return_soup_output=True)
    mu, sigma, moe_logits, qual, log_var, mu_soup = res
    
    assert mu.shape == (2, 8, 64)
    assert mu_soup.shape == (2, 8, 64)
    assert moe_logits.shape == (2, 8)
    assert torch.all(torch.isfinite(mu))
    assert torch.all(torch.isfinite(mu_soup))

    # Check soup deficit loss computation and backward pass
    deficit_loss = mamba.compute_soup_deficit_loss(z_seq, u, expert_mask, tau_moe)
    assert deficit_loss.dim() == 0
    assert deficit_loss.item() >= 0.0
    
    deficit_loss.backward()
    # Check that residual experts receive gradients from deficit loss
    assert mamba.residual_experts[0].in_proj.weight_res.slices[0].grad is not None

