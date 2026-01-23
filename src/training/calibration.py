"""
Calibration utilities for Cricket Ball Prediction Model.

Implements calibration metrics for evaluating probability estimation quality.
"""

import torch
from typing import Tuple
import numpy as np


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
