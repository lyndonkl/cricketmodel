"""
Configuration for Cricket Ball Prediction Model

Central configuration for data processing, model architecture, and training.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Paths
    raw_data_dir: str = "data/t20s_male_json"
    processed_data_dir: str = "data/processed"

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio is implicit: 1 - train_ratio - val_ratio

    # Processing
    min_history: int = 1  # Minimum balls of history for prediction
    seed: int = 42

    # DataLoader
    batch_size: int = 64
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Architecture
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4

    # Embedding dimensions
    venue_embed_dim: int = 32
    team_embed_dim: int = 32
    player_embed_dim: int = 64

    # Regularization
    dropout: float = 0.1

    # Task
    num_classes: int = 7


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Schedule
    epochs: int = 100
    warmup_epochs: int = 5
    scheduler: str = "cosine"  # 'cosine' or 'plateau'

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Class weighting
    use_class_weights: bool = True


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Runtime
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'mps'
    seed: int = 42

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.data.processed_data_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)


def get_device(device_config: str = "auto") -> "torch.device":
    """Get appropriate device based on config."""
    import torch

    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
