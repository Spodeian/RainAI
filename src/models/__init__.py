"""RainAI Neural Models, Quantization, and Physics Modules."""

from .diff_autoencoder import (
    SpatialAudioEncoder,
    AffineAlignment,
    LearnedMixedPrecisionQuantizer,
    BidirectionalMambaBlock,
    HierarchicalMultiResLoss,
    LATENT_DIM,
    CONDITION_DIM,
)
from .mamba2_moe import (
    Mamba2MoETrajectory,
    MambaSSDBlock,
    JambaSelfAttentionBlock,
    NUM_EXPERTS,
)
from .meta_controller import InvasiveMetaController
from .ddsp import (
    ContinuousParametricFilter,
    DifferentiableReverbEngine,
    spherical_harmonics_foa,
    SAMPLE_RATE,
)
from .physics_losses import (
    MultiScaleAmbisonicPhysicsLoss,
    PhysicsTrajectoryLoss,
    InstantaneousPhaseLoss,
    MultiScaleSTFTDiscriminator,
    MoELoadBalancingLoss,
    KnowledgeDistillationLoss,
    BetaVAEDisentanglementLoss,
    compute_slice_aware_hwil_penalty,
)
from .residual_quant import (
    ResidualWeight,
    ResidualLinear,
    ResidualConv1d,
    NUM_DEFAULT_SLICES,
)

__all__ = [
    "SpatialAudioEncoder",
    "AffineAlignment",
    "LearnedMixedPrecisionQuantizer",
    "BidirectionalMambaBlock",
    "HierarchicalMultiResLoss",
    "LATENT_DIM",
    "CONDITION_DIM",
    "Mamba2MoETrajectory",
    "MambaSSDBlock",
    "JambaSelfAttentionBlock",
    "NUM_EXPERTS",
    "InvasiveMetaController",
    "ContinuousParametricFilter",
    "DifferentiableReverbEngine",
    "spherical_harmonics_foa",
    "SAMPLE_RATE",
    "MultiScaleAmbisonicPhysicsLoss",
    "PhysicsTrajectoryLoss",
    "InstantaneousPhaseLoss",
    "MultiScaleSTFTDiscriminator",
    "MoELoadBalancingLoss",
    "KnowledgeDistillationLoss",
    "BetaVAEDisentanglementLoss",
    "compute_slice_aware_hwil_penalty",
    "ResidualWeight",
    "ResidualLinear",
    "ResidualConv1d",
    "NUM_DEFAULT_SLICES",
]