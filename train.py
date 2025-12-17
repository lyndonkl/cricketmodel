#!/usr/bin/env python3
"""
Main Training Script for Cricket Ball Prediction Model

Usage:
    python train.py                    # Train with default config
    python train.py --epochs 50        # Override specific settings
    python train.py --test-only        # Evaluate existing model
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from src.config import Config, get_device, set_seed
from src.data import (
    CricketDataset,
    create_dataloaders,
    compute_class_weights,
    get_class_distribution,
    EntityMapper,
)
from src.model import (
    CricketHeteroGNN,
    ModelConfig,
    count_parameters,
    get_model_summary,
)
from src.training import Trainer, TrainingConfig


def setup_data_directory(raw_dir: str, processed_dir: str):
    """
    Setup the data directory structure expected by CricketDataset.

    CricketDataset expects:
    - {root}/raw/matches/*.json

    Args:
        raw_dir: Directory containing raw JSON match files
        processed_dir: Directory for processed dataset
    """
    root = os.path.dirname(processed_dir)
    raw_matches_dir = os.path.join(root, "raw", "matches")

    # Create directory structure
    os.makedirs(raw_matches_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Check if matches already linked/copied
    existing = list(Path(raw_matches_dir).glob("*.json"))
    if len(existing) > 0:
        print(f"Found {len(existing)} match files in {raw_matches_dir}")
        return root

    # Link or copy match files from raw_dir
    source_files = list(Path(raw_dir).glob("*.json"))
    if len(source_files) == 0:
        raise RuntimeError(f"No JSON files found in {raw_dir}")

    print(f"Setting up data: linking {len(source_files)} match files...")

    for src_file in source_files:
        dst_file = os.path.join(raw_matches_dir, src_file.name)
        if not os.path.exists(dst_file):
            # Create symlink (or copy on Windows)
            try:
                os.symlink(src_file.absolute(), dst_file)
            except OSError:
                shutil.copy(str(src_file), dst_file)

    return root


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Cricket Ball Prediction Model"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/t20s_male_json",
        help="Directory containing raw JSON match files"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory for processed dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Model
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of message passing layers"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run test evaluation (requires trained model)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for imbalanced data"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Setup data directory
    print("\n" + "=" * 60)
    print("Setting up data...")
    print("=" * 60)

    data_root = setup_data_directory(args.data_dir, args.processed_dir)
    print(f"Data root: {data_root}")

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating datasets...")
    print("=" * 60)

    train_dataset = CricketDataset(
        root=data_root,
        split="train",
        min_history=1,
        seed=args.seed,
    )
    val_dataset = CricketDataset(
        root=data_root,
        split="val",
    )
    test_dataset = CricketDataset(
        root=data_root,
        split="test",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Class distribution
    print("\nClass distribution (train):")
    dist = get_class_distribution(train_dataset)
    for name, info in dist.items():
        print(f"  {name}: {info['count']} ({info['percentage']:.1f}%)")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Get entity counts from metadata
    metadata = train_dataset.get_metadata()
    print(f"\nEntity counts:")
    print(f"  Venues: {metadata['num_venues']}")
    print(f"  Teams: {metadata['num_teams']}")
    print(f"  Players: {metadata['num_players']}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)

    model_config = ModelConfig(
        num_venues=metadata["num_venues"],
        num_teams=metadata["num_teams"],
        num_players=metadata["num_players"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    model = CricketHeteroGNN(model_config)
    print(get_model_summary(model))

    # Compute class weights
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_dataset)
        print(f"Class weights: {class_weights.tolist()}")

    # Test only mode
    if args.test_only:
        print("\n" + "=" * 60)
        print("Test evaluation only")
        print("=" * 60)

        checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"No checkpoint found at {checkpoint_path}")

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Create trainer just for evaluation
        training_config = TrainingConfig(checkpoint_dir=args.checkpoint_dir)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            class_weights=class_weights,
            device=device,
        )

        # Test
        test_metrics = trainer.test(test_loader)

        # Save test results
        results_path = os.path.join(args.checkpoint_dir, "test_results.json")
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for k, v in test_metrics.items():
            if hasattr(v, "tolist"):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
        with open(results_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
        print(f"Test results saved to {results_path}")

        return

    # Create trainer
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    training_config = TrainingConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        class_weights=class_weights,
        device=device,
    )

    # Train
    history = trainer.train()

    # Test
    print("\n" + "=" * 60)
    print("Final evaluation on test set...")
    print("=" * 60)

    test_metrics = trainer.test(test_loader)

    # Save test results
    results_path = os.path.join(args.checkpoint_dir, "test_results.json")
    serializable_metrics = {}
    for k, v in test_metrics.items():
        if hasattr(v, "tolist"):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v
    with open(results_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2, default=str)
    print(f"Test results saved to {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
