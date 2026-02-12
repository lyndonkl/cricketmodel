"""
Evaluation Metrics for Cricket Score-Ahead Regression

Provides regression metrics for evaluating score-ahead predictions.
"""

import numpy as np
from typing import Dict


def compute_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics for score-ahead predictions.

    Args:
        targets: Ground truth score-ahead values
        predictions: Predicted score-ahead values

    Returns:
        Dict with mae, rmse, r_squared, median_ae
    """
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)

    errors = predictions - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    median_ae = np.median(abs_errors)

    # R² (coefficient of determination)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r_squared': float(r_squared),
        'median_ae': float(median_ae),
    }


def print_regression_report(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> str:
    """
    Print a formatted regression report.

    Args:
        targets: Ground truth score-ahead values
        predictions: Predicted score-ahead values

    Returns:
        Formatted report string
    """
    metrics = compute_regression_metrics(targets, predictions)

    lines = []
    lines.append("=" * 50)
    lines.append("Score-Ahead Regression Report")
    lines.append("=" * 50)
    lines.append(f"  MAE:       {metrics['mae']:.2f}")
    lines.append(f"  RMSE:      {metrics['rmse']:.2f}")
    lines.append(f"  Median AE: {metrics['median_ae']:.2f}")
    lines.append(f"  R²:        {metrics['r_squared']:.4f}")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)
    return report
