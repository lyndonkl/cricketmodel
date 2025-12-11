"""Configuration for cricket prediction model."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Embedding dimensions
    player_embed_dim: int = 64
    venue_embed_dim: int = 32
    team_embed_dim: int = 32

    # Graph structure
    node_hidden_dim: int = 64
    num_gat_heads: int = 4
    gat_dropout: float = 0.1

    # Temporal transformer
    temporal_seq_len: int = 24
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dim: int = 128
    transformer_dropout: float = 0.1

    # Output
    num_outcomes: int = 7  # dot, 1, 2, 3, 4, 6, wicket


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup_epochs: int = 5

    # Device
    device: Literal["mps", "cuda", "cpu"] = "mps"
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5


@dataclass
class DataConfig:
    """Data loading configuration."""

    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split is implicit: 1 - train - val

    # Filtering
    min_balls_per_match: int = 60  # Exclude incomplete matches
    formats: list[str] = field(default_factory=lambda: ["T20"])


@dataclass
class Config:
    """Root configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
