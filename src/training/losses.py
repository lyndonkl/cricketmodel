"""
Loss Functions for Cricket Ball Prediction

Implements Focal Loss and other specialized loss functions
for handling class imbalance in cricket outcome prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard examples.
    This is particularly useful for cricket prediction where:
    - Dot balls (~35-40%) are often "easy" predictions
    - Wickets (~5%) are rare but critical
    - Boundaries require capturing specific patterns

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        gamma: Focusing parameter. Higher gamma increases focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
               gamma=2 is recommended starting point. (default: 2.0)
        alpha: Class weights (optional). Can be a scalar or per-class tensor.
               If provided, applies class-specific weighting.
        reduction: Specifies the reduction to apply to the output:
                   'none' | 'mean' | 'sum'. (default: 'mean')
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value (scalar if reduction='mean' or 'sum')
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        # p_t = probability assigned to the correct class
        batch_size = logits.size(0)
        p_t = probs[torch.arange(batch_size, device=logits.device), targets]

        # Compute focal weight: (1 - p_t)^gamma
        # This down-weights easy examples (high p_t) and up-weights hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Combine: focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Prevents overconfident predictions by distributing some probability
    mass to non-target classes.

    Args:
        smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform). (default: 0.1)
        reduction: 'mean' | 'sum' | 'none'. (default: 'mean')
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_probs)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        loss = (-smooth_labels * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss with Label Smoothing.

    This provides the benefits of both:
    - Focal loss: Focus on hard examples, handle class imbalance
    - Label smoothing: Prevent overconfidence, improve calibration

    Args:
        gamma: Focal loss focusing parameter. (default: 2.0)
        alpha: Class weights for focal loss. (default: None)
        smoothing: Label smoothing factor. (default: 0.1)
        focal_weight: Weight for focal loss component. (default: 0.7)
        smooth_weight: Weight for label smoothing component. (default: 0.3)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        smoothing: float = 0.1,
        focal_weight: float = 0.7,
        smooth_weight: float = 0.3
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(logits, targets)
        smooth = self.smooth_loss(logits, targets)
        return self.focal_weight * focal + self.smooth_weight * smooth


def compute_class_weights_inverse_freq(
    class_counts: torch.Tensor,
    smoothing: float = 1.0
) -> torch.Tensor:
    """
    Compute inverse frequency class weights.

    Args:
        class_counts: Number of samples per class [num_classes]
        smoothing: Smoothing factor to prevent extreme weights. (default: 1.0)

    Returns:
        Class weights [num_classes]
    """
    total = class_counts.sum()
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts + smoothing)
    # Normalize to sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights


def compute_class_weights_effective_samples(
    class_counts: torch.Tensor,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    Compute class weights using effective number of samples.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (2019)

    Args:
        class_counts: Number of samples per class [num_classes]
        beta: Hyperparameter controlling the weighting. (default: 0.9999)
              Higher beta = more weight on rare classes.

    Returns:
        Class weights [num_classes]
    """
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    # Normalize
    weights = weights / weights.sum() * len(class_counts)
    return weights
