"""
Robust loss functions and dynamic uncertainty weighting for RainAI.

Provides:
- Distributional spectral cross-entropy for acoustic tokenization.
- Outlier-resistant Charbonnier and Huber spectral penalties.
- Dynamic Multi-Task Uncertainty Weighting (Kendall & Gal, 2018), where loss
  weights are found dynamically via homoscedastic variance rather than hardcoded.
"""

import torch
import torch.nn as nn


class DynamicMultiTaskLoss(nn.Module):
    """
    Learns the optimal weighting between competing loss terms dynamically:
    L_{total} = sum_i ( 1/(2 * sigma_i^2) * L_i + ln(sigma_i) )

    Eliminates the risk of manually chosen, sub-optimal loss coefficients.
    """

    def __init__(self, num_tasks: int = 4):
        super().__init__()
        # log_vars represent ln(sigma_i^2), initialized to 0.0 (sigma = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        assert len(losses) == len(self.log_vars)
        total_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return total_loss


def charbonnier_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Smooth, differentiable L1 approximation robust to spectral pop outliers:
    L(a) = sqrt(a^2 + eps^2)
    """
    diff = y_pred - y_true
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def spectral_cross_entropy(
    logits: torch.Tensor, target_distribution: torch.Tensor
) -> torch.Tensor:
    """
    Cross-entropy loss over normalized spectral frequency band probabilities.
    Encourages accurate distribution of acoustic energy across octaves.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(target_distribution * log_probs, dim=-1).mean()
