"""Training utilities for cricket ball prediction."""

from .trainer import Trainer, TrainingConfig
from .metrics import (
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix,
)
from .calibration import (
    compute_ece_from_tensors,
    compute_reliability_data,
    create_reliability_diagram_figure,
)
from .losses import (
    FocalLoss,
    BinaryFocalLoss,
    BinaryHeadLoss,
    compute_binary_head_metrics,
)
from .distributed import (
    is_distributed,
    is_main_process,
    get_rank,
    get_local_rank,
    get_world_size,
    get_backend,
    is_mps_available,
    setup_distributed,
    cleanup_distributed,
    barrier,
    reduce_tensor,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    # Metrics
    "compute_metrics",
    "print_classification_report",
    "plot_confusion_matrix",
    # Calibration
    "compute_ece_from_tensors",
    "compute_reliability_data",
    "create_reliability_diagram_figure",
    # Losses
    "FocalLoss",
    "BinaryFocalLoss",
    "BinaryHeadLoss",
    "compute_binary_head_metrics",
    # DDP utilities
    "is_distributed",
    "is_main_process",
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "get_backend",
    "is_mps_available",
    "setup_distributed",
    "cleanup_distributed",
    "barrier",
    "reduce_tensor",
]
