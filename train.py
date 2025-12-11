#!/usr/bin/env python3
"""Train cricket prediction model."""

import argparse

from src.config import Config, ModelConfig, TrainingConfig, DataConfig
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train cricket prediction model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint dir")
    args = parser.parse_args()

    # Create config
    config = Config(
        model=ModelConfig(
            temporal_seq_len=24,
            transformer_dim=128,
            num_gat_heads=4,
            transformer_heads=4,
            transformer_layers=2,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
        ),
        data=DataConfig(
            data_dir=args.data_dir,
        ),
    )

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
