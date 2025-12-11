"""Evaluation metrics for cricket prediction."""

from dataclasses import dataclass

import torch


OUTCOME_NAMES = ["dot", "single", "two", "three", "four", "six", "wicket"]


@dataclass
class OutcomeMetrics:
    """Per-outcome metrics."""

    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    confusion_matrix: torch.Tensor
    calibration_error: float


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probs: torch.Tensor | None = None,
    num_classes: int = 7,
) -> OutcomeMetrics:
    """
    Compute comprehensive metrics.

    Args:
        predictions: Predicted class indices [N]
        labels: True class indices [N]
        probs: Predicted probabilities [N, num_classes] for calibration
        num_classes: Number of outcome classes

    Returns:
        OutcomeMetrics with all computed metrics
    """
    # Overall accuracy
    correct = (predictions == labels).float()
    accuracy = correct.mean().item()

    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for pred, true in zip(predictions, labels):
        confusion[true.item(), pred.item()] += 1

    # Per-class metrics
    precision = {}
    recall = {}
    f1 = {}

    for i, name in enumerate(OUTCOME_NAMES[:num_classes]):
        # True positives, false positives, false negatives
        tp = confusion[i, i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp

        # Precision
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision[name] = prec

        # Recall
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall[name] = rec

        # F1
        f1[name] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Calibration error (Expected Calibration Error)
    calibration_error = 0.0
    if probs is not None:
        calibration_error = _compute_ece(probs, labels, num_bins=10)

    return OutcomeMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=confusion,
        calibration_error=calibration_error,
    )


def _compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    ece = 0.0
    for bin_idx in range(num_bins):
        bin_lower = bin_idx / num_bins
        bin_upper = (bin_idx + 1) / num_bins

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = accuracies[in_bin].mean().item()
            ece += abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def print_metrics(metrics: OutcomeMetrics) -> None:
    """Print metrics in readable format."""
    print(f"\nOverall Accuracy: {metrics.accuracy:.4f}")
    print(f"Calibration Error: {metrics.calibration_error:.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 40)

    for name in OUTCOME_NAMES[:len(metrics.precision)]:
        print(
            f"{name:<10} "
            f"{metrics.precision[name]:<10.4f} "
            f"{metrics.recall[name]:<10.4f} "
            f"{metrics.f1[name]:<10.4f}"
        )

    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix)
