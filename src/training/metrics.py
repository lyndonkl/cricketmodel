"""
Evaluation Metrics for Cricket Ball Prediction

Provides comprehensive metrics for multi-class classification
with focus on imbalanced cricket outcome prediction.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import Counter


# Class names for cricket outcomes
CLASS_NAMES = ['Dot', 'Single', 'Two', 'Three', 'Four', 'Six', 'Wicket']


def compute_accuracy(labels: List[int], predictions: List[int]) -> float:
    """Compute accuracy."""
    correct = sum(l == p for l, p in zip(labels, predictions))
    return correct / len(labels) if len(labels) > 0 else 0.0


def compute_class_accuracies(
    labels: List[int],
    predictions: List[int],
    num_classes: int = 7
) -> Dict[str, float]:
    """Compute per-class accuracy."""
    class_correct = Counter()
    class_total = Counter()

    for l, p in zip(labels, predictions):
        class_total[l] += 1
        if l == p:
            class_correct[l] += 1

    accuracies = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            accuracies[CLASS_NAMES[c]] = class_correct[c] / class_total[c]
        else:
            accuracies[CLASS_NAMES[c]] = 0.0

    return accuracies


def compute_f1_scores(
    labels: List[int],
    predictions: List[int],
    num_classes: int = 7
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per class and macro/weighted averages."""

    # Per-class counts
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()

    for l, p in zip(labels, predictions):
        if l == p:
            true_positives[l] += 1
        else:
            false_positives[p] += 1
            false_negatives[l] += 1

    # Per-class metrics
    per_class = {}
    for c in range(num_classes):
        tp = true_positives[c]
        fp = false_positives[c]
        fn = false_negatives[c]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[CLASS_NAMES[c]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn,
        }

    # Macro average (unweighted mean)
    f1_values = [per_class[name]['f1'] for name in CLASS_NAMES]
    precision_values = [per_class[name]['precision'] for name in CLASS_NAMES]
    recall_values = [per_class[name]['recall'] for name in CLASS_NAMES]

    macro_f1 = np.mean(f1_values)
    macro_precision = np.mean(precision_values)
    macro_recall = np.mean(recall_values)

    # Weighted average
    total_support = sum(per_class[name]['support'] for name in CLASS_NAMES)
    if total_support > 0:
        weighted_f1 = sum(
            per_class[name]['f1'] * per_class[name]['support']
            for name in CLASS_NAMES
        ) / total_support
        weighted_precision = sum(
            per_class[name]['precision'] * per_class[name]['support']
            for name in CLASS_NAMES
        ) / total_support
        weighted_recall = sum(
            per_class[name]['recall'] * per_class[name]['support']
            for name in CLASS_NAMES
        ) / total_support
    else:
        weighted_f1 = weighted_precision = weighted_recall = 0.0

    return {
        'per_class': per_class,
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
        },
    }


def compute_top_k_accuracy(
    labels: List[int],
    probs: np.ndarray,
    k: int = 2
) -> float:
    """Compute top-k accuracy."""
    if len(labels) == 0:
        return 0.0

    correct = 0
    for i, label in enumerate(labels):
        top_k_preds = np.argsort(probs[i])[-k:]
        if label in top_k_preds:
            correct += 1

    return correct / len(labels)


def compute_log_loss(
    labels: List[int],
    probs: np.ndarray,
    eps: float = 1e-15
) -> float:
    """Compute cross-entropy loss."""
    probs = np.clip(probs, eps, 1 - eps)
    n_samples = len(labels)
    if n_samples == 0:
        return 0.0

    loss = -sum(np.log(probs[i, labels[i]]) for i in range(n_samples))
    return loss / n_samples


def compute_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    num_classes: int = 7
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for l, p in zip(labels, predictions):
        cm[l, p] += 1
    return cm


def compute_metrics(
    labels: List[int],
    predictions: List[int],
    probs: Optional[np.ndarray] = None,
    num_classes: int = 7
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        probs: Predicted probabilities [num_samples, num_classes]
        num_classes: Number of classes

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Basic accuracy
    metrics['accuracy'] = compute_accuracy(labels, predictions)

    # Per-class accuracy
    metrics['class_accuracies'] = compute_class_accuracies(labels, predictions, num_classes)

    # F1 scores
    f1_metrics = compute_f1_scores(labels, predictions, num_classes)
    metrics['f1_macro'] = f1_metrics['macro']['f1']
    metrics['f1_weighted'] = f1_metrics['weighted']['f1']
    metrics['precision_macro'] = f1_metrics['macro']['precision']
    metrics['recall_macro'] = f1_metrics['macro']['recall']
    metrics['per_class_f1'] = f1_metrics['per_class']

    # Confusion matrix
    metrics['confusion_matrix'] = compute_confusion_matrix(labels, predictions, num_classes)

    # Probability-based metrics (if available)
    if probs is not None:
        metrics['log_loss'] = compute_log_loss(labels, probs)
        metrics['top_2_accuracy'] = compute_top_k_accuracy(labels, probs, k=2)
        metrics['top_3_accuracy'] = compute_top_k_accuracy(labels, probs, k=3)

    return metrics


def print_classification_report(
    labels: List[int],
    predictions: List[int],
    probs: Optional[np.ndarray] = None
) -> str:
    """
    Print a formatted classification report.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        probs: Predicted probabilities

    Returns:
        Formatted report string
    """
    metrics = compute_metrics(labels, predictions, probs)

    lines = []
    lines.append("=" * 70)
    lines.append("Classification Report")
    lines.append("=" * 70)

    # Overall metrics
    lines.append(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"Macro F1: {metrics['f1_macro']:.4f}")
    lines.append(f"Weighted F1: {metrics['f1_weighted']:.4f}")

    if 'log_loss' in metrics:
        lines.append(f"Log Loss: {metrics['log_loss']:.4f}")
        lines.append(f"Top-2 Accuracy: {metrics['top_2_accuracy']:.4f}")
        lines.append(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")

    # Per-class metrics
    lines.append("\n" + "-" * 70)
    lines.append(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 70)

    for class_name in CLASS_NAMES:
        class_metrics = metrics['per_class_f1'][class_name]
        lines.append(
            f"{class_name:<10} "
            f"{class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} "
            f"{class_metrics['f1']:>10.4f} "
            f"{class_metrics['support']:>10}"
        )

    # Confusion matrix
    lines.append("\n" + "-" * 70)
    lines.append("Confusion Matrix")
    lines.append("-" * 70)

    cm = metrics['confusion_matrix']
    header = "Pred:  " + "  ".join(f"{name[:3]:>5}" for name in CLASS_NAMES)
    lines.append(header)
    lines.append("True:")
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i][:3]:>5}  " + "  ".join(f"{v:>5}" for v in row)
        lines.append(row_str)

    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    return report


def plot_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix using matplotlib.

    Args:
        labels: Ground truth labels
        predictions: Predicted labels
        save_path: Path to save figure (optional)
        normalize: Whether to normalize by row
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required for plotting")
        return

    cm = compute_confusion_matrix(labels, predictions)

    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        cm_normalized = cm

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()
