"""
RainAI Core Neural Architecture & Physics Models
"""

from .residual_quant import ResidualWeight, ResidualLinear, ResidualConv1d
from .diff_autoencoder import SpatialAudioEncoder, HierarchicalMultiResLoss, AffineAlignment, LearnedMixedPrecisionQuantizer
from .ddsp import ContinuousParametricFilter, DifferentiableReverbEngine, spherical_harmonics_foa
from .mamba2_moe import Mamba2MoETrajectory, MambaSSDBlock, JambaSelfAttentionBlock
from .meta_controller import InvasiveMetaController
from .physics_losses import (
    MultiScaleAmbisonicPhysicsLoss,
    PhysicsTrajectoryLoss,
    MoELoadBalancingLoss,
    BetaVAEDisentanglementLoss,
    MultiScaleSTFTDiscriminator,
    compute_slice_aware_hwil_penalty,
    discriminator_hinge_loss,
    generator_adversarial_loss,
    feature_matching_loss
)

__all__ = [
    "ResidualWeight", "ResidualLinear", "ResidualConv1d",
    "SpatialAudioEncoder", "HierarchicalMultiResLoss", "AffineAlignment", "LearnedMixedPrecisionQuantizer",
    "ContinuousParametricFilter", "DifferentiableReverbEngine", "spherical_harmonics_foa",
    "Mamba2MoETrajectory", "MambaSSDBlock", "JambaSelfAttentionBlock",
    "InvasiveMetaController",
    "MultiScaleAmbisonicPhysicsLoss", "PhysicsTrajectoryLoss", "MoELoadBalancingLoss",
    "BetaVAEDisentanglementLoss", "MultiScaleSTFTDiscriminator",
    "compute_slice_aware_hwil_penalty", "discriminator_hinge_loss", 
    "generator_adversarial_loss", "feature_matching_loss"
]