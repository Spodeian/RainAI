"""
Engram Conditional Memory Architecture (DeepSeek-style O(1) Knowledge Bank).
Decouples static acoustic physics / material priors from dynamic neural computation.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BANK_SIZE = 32768  # Option B: Rich configuration (32K entries, ~2MB)
DEFAULT_EMBED_DIM = 64
DEFAULT_HASH_HEADS = 4


class EngramBank(nn.Module):
    """
    O(1) hash-addressed static knowledge bank for physical audio priors.
    
    Attributes:
        bank_size: Number of stored discrete prior representations (32,768).
        embed_dim: Dimensionality of each stored embedding (64).
        num_hash_heads: Number of independent hash projections for collision resistance.
    """
    def __init__(
        self,
        bank_size: int = DEFAULT_BANK_SIZE,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_hash_heads: int = DEFAULT_HASH_HEADS,
    ):
        super().__init__()
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        self.num_hash_heads = num_hash_heads

        # Static knowledge bank embedding table
        self.bank = nn.Embedding(bank_size, embed_dim)
        nn.init.normal_(self.bank.weight, mean=0.0, std=0.02)

        # Multi-head hash projections (fixed random orthogonal projections for deterministic hashing)
        hash_weights = []
        for i in range(num_hash_heads):
            gen = torch.Generator().manual_seed(42 + i * 1337)
            w = torch.randn(embed_dim, generator=gen)
            w = w / (torch.norm(w) + 1e-8)
            hash_weights.append(w)
        self.register_buffer("hash_projections", torch.stack(hash_weights, dim=0))  # (H, embed_dim)

        self.register_buffer(
            "hash_primes",
            torch.tensor([2654435761, 2246822519, 3266489917, 668265263], dtype=torch.int64)
        )

        # Gated fusion layer: combines retrieved engram knowledge with neural feature
        self.fuse_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)

    def compute_hash_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes deterministic multi-head hash indices from continuous feature tensor.
        Input x: (*shape, embed_dim)
        Output: (*shape, num_hash_heads) of integer indices in [0, bank_size - 1].
        """
        # Linear projection per hash head: (*shape, num_hash_heads)
        projections = torch.matmul(x, self.hash_projections.t())

        # Quantize projection into integer bins
        scaled = (projections * 1000.0).to(torch.int64)

        primes = self.hash_primes[:self.num_hash_heads]
        indices = torch.abs((scaled * primes) % self.bank_size)
        return indices

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves and aggregates embeddings across hash heads.
        indices: (*shape, num_hash_heads)
        Returns: (*shape, embed_dim)
        """
        retrieved = self.bank(indices)
        return torch.mean(retrieved, dim=-2)

    def forward(
        self,
        neural_features: torch.Tensor,
        query_key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs O(1) engram retrieval and gated residual fusion.
        """
        query = neural_features if query_key is None else query_key
        indices = self.compute_hash_indices(query)
        engram_prior = self.lookup(indices)

        concat = torch.cat([neural_features, engram_prior], dim=-1)
        gate = torch.sigmoid(self.fuse_gate(concat))
        fused = gate * engram_prior + (1.0 - gate) * neural_features
        return self.out_norm(fused), gate
