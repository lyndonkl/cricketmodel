"""
Calibration utilities for Cricket Ball Prediction Model.

Implements temperature scaling and calibration metrics for
improving probability estimation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Optional, Tuple
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling for model calibration.

    Temperature scaling is a simple but effective technique to calibrate
    neural network predictions. It divides logits by a learned temperature
    parameter T, which is optimized on a validation set to minimize NLL.

    Benefits:
    - Preserves accuracy (doesn't change argmax predictions)
    - Dramatically improves calibration (ECE can drop from 0.08 to 0.02)
    - Single parameter - fast to optimize

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)

    Usage:
        # After training
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(model, val_loader, device)

        # During inference
        logits = model(batch)
        calibrated_logits = temp_scaler(logits)
        probs = F.softmax(calibrated_logits, dim=-1)
    """

    def __init__(self, initial_temperature: float = 1.0):
        """
        Args:
            initial_temperature: Starting temperature value. Default 1.0 means
                                no initial scaling.
        """
        super().__init__()
        # Temperature must be positive, so we use a parameter that can be any value
        # but we'll clamp it during optimization
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Model output logits [batch_size, num_classes]

        Returns:
            Temperature-scaled logits [batch_size, num_classes]
        """
        return logits / self.temperature.clamp(min=0.01)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Optimize temperature on validation set.

        Uses L-BFGS optimizer to find optimal temperature that minimizes
        negative log-likelihood on the validation set.

        Args:
            model: Trained model (will be set to eval mode)
            val_loader: Validation DataLoader
            device: Device to run on
            max_iter: Maximum L-BFGS iterations
            lr: L-BFGS learning rate

        Returns:
            Optimized temperature value
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)

                # Handle multi-head output (dict) vs single-head (tensor)
                if isinstance(outputs, dict):
                    logits = outputs['main']
                else:
                    logits = outputs

                all_logits.append(logits)
                all_labels.append(batch.y)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Optimize temperature using L-BFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature.clamp(min=0.01)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get calibrated probabilities from logits.

        Args:
            logits: Model output logits [batch_size, num_classes]

        Returns:
            Calibrated probabilities [batch_size, num_classes]
        """
        scaled_logits = self.forward(logits)
        return F.softmax(scaled_logits, dim=-1)


def compute_ece_from_tensors(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error from PyTorch tensors.

    This is a tensor-based version for use during training.

    Args:
        probs: Softmax probabilities [N, C]
        labels: True class labels [N]
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].float().mean()
            ece += prop_in_bin * torch.abs(avg_accuracy - avg_confidence)

    return ece.item()


def compute_reliability_data(
    labels: list,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.

    Args:
        labels: Ground truth labels
        probs: Predicted probabilities [N, C]
        n_bins: Number of bins

    Returns:
        Tuple of (bin_confidences, bin_accuracies, bin_counts)
    """
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies = (predictions == np.array(labels)).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = np.sum(in_bin)
        bin_counts.append(count)

        if count > 0:
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_accuracies.append(np.mean(accuracies[in_bin]))
        else:
            # Use bin center for empty bins
            bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(0)

    return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)


def create_reliability_diagram_figure(
    labels: list,
    probs: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
):
    """
    Create reliability diagram figure for WandB logging.

    A reliability diagram shows calibration quality by comparing
    average predicted confidence to actual accuracy in each bin.
    A perfectly calibrated model has all points on the diagonal.

    Args:
        labels: Ground truth labels
        probs: Predicted probabilities [N, C]
        n_bins: Number of bins
        title: Figure title

    Returns:
        matplotlib figure object (or None if matplotlib not available)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    bin_confidences, bin_accuracies, bin_counts = compute_reliability_data(
        labels, probs, n_bins
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Bar chart showing accuracy vs confidence per bin
    bar_width = 1.0 / n_bins
    bin_edges = np.linspace(0, 1 - bar_width, n_bins)

    # Gap bars (difference from perfect calibration)
    gaps = bin_accuracies - bin_confidences
    colors = ['tab:red' if g < 0 else 'tab:blue' for g in gaps]

    ax1.bar(bin_edges + bar_width/2, bin_accuracies, width=bar_width * 0.8,
            alpha=0.7, edgecolor='black', label='Outputs')
    ax1.scatter(bin_confidences, bin_accuracies, c='red', s=50, zorder=5,
                label='Accuracy vs Confidence')

    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Fraction of Positives (Accuracy)')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Histogram of confidences
    ax2.bar(bin_edges + bar_width/2, bin_counts, width=bar_width * 0.8,
            alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
