#!/usr/bin/env python3
"""
Main Training Script for Cricket Ball Prediction Model

Supports both single-GPU and Distributed Data Parallel (DDP) training.

Usage:
    # Single GPU (unchanged)
    python train.py                    # Train with default config
    python train.py --epochs 50        # Override specific settings
    python train.py --test-only        # Evaluate existing model

    # Multi-GPU with DDP
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=4 train.py --batch-size 32

    # All available GPUs
    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
"""

import argparse
import json
import os

# Enable MPS fallback for PyTorch Geometric ops not yet implemented on MPS
# (e.g., scatter_reduce used in GATv2Conv attention)
# This must be set BEFORE importing torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Fix OpenMP duplicate library error on macOS with conda + pip
# This occurs when multiple copies of libomp are linked (conda's and pip's)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config import set_seed
from src.data import (
    create_dataloaders,
    compute_class_weights,
)
from src.data.dataset import create_dataloaders_distributed
from src.model import (
    CricketHeteroGNN,
    ModelConfig,
    get_model_summary,
)
from src.training import Trainer, TrainingConfig
from src.training.distributed import (
    is_distributed,
    is_main_process,
    get_rank,
    get_local_rank,
    get_world_size,
    setup_distributed,
    cleanup_distributed,
    barrier,
)


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
        help="Batch size per GPU (effective batch = batch_size * num_gpus)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers per GPU"
    )
    parser.add_argument(
        "--ollama-workers",
        type=int,
        default=4,
        help="Number of parallel workers for Ollama player classification"
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
        "--test-only",
        action="store_true",
        help="Only run test evaluation (requires trained model)"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for imbalanced data"
    )

    # WandB
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cricket-gnn",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (auto-generated if not set)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage (required for DDP with torchrun on non-CUDA systems)"
    )

    return parser.parse_args()


def main():
    """Main training function with DDP support."""
    args = parse_args()

    # Resolve paths to absolute paths
    args.data_dir = os.path.abspath(args.data_dir)
    args.processed_dir = os.path.abspath(args.processed_dir)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # === DDP Setup ===
    # Auto-detect DDP from environment variables set by torchrun
    rank, local_rank, world_size = setup_distributed()
    ddp_enabled = world_size > 1
    is_main = is_main_process()

    # Set device based on --force-cpu flag, DDP status, and available hardware
    # Platform behavior:
    #   - --force-cpu: Always use CPU (required for DDP with Gloo on non-CUDA)
    #   - Linux/RunPod with CUDA: Use cuda:{local_rank} for DDP
    #   - Mac with MPS (no DDP): Use MPS for single-device training
    #   - CPU fallback: Use CPU
    if args.force_cpu:
        device = torch.device('cpu')
    elif ddp_enabled:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
        else:
            # Gloo backend requires CPU tensors - MPS not compatible
            device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Set seed (with rank offset for DDP to ensure different data order)
    set_seed(args.seed + rank)

    if is_main:
        print(f"Using device: {device}")
        if ddp_enabled:
            print(f"DDP enabled: {world_size} GPUs")
            print(f"Effective batch size: {args.batch_size * world_size}")

    # === WandB Setup ===
    if args.wandb and is_main:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "patience": args.patience,
                "world_size": world_size,
                "device": str(device),
            }
        )
        print("WandB logging enabled")

    # === Create Dataloaders ===
    if is_main:
        print("\n" + "=" * 60)
        print("Creating datasets and dataloaders...")
        print("=" * 60)
        print(f"Raw data: {args.data_dir}")
        print(f"Processed cache: {args.processed_dir}")

    if ddp_enabled:
        train_loader, val_loader, test_loader, train_sampler = create_dataloaders_distributed(
            root=args.processed_dir,
            raw_data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=rank,
            world_size=world_size,
            min_history=1,
            seed=args.seed,
            ollama_workers=args.ollama_workers,
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            root=args.processed_dir,
            raw_data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_history=1,
            seed=args.seed,
            ollama_workers=args.ollama_workers,
        )
        train_sampler = None

    # Access datasets for metadata and stats
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    if is_main:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    # Get entity counts from metadata
    metadata = train_dataset.get_metadata()
    if is_main:
        print(f"\nEntity counts:")
        print(f"  Venues: {metadata['num_venues']}")
        print(f"  Teams: {metadata['num_teams']}")
        print(f"  Players: {metadata['num_players']}")

    # === Create Model ===
    if is_main:
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
    if is_main:
        print(get_model_summary(model))

    # Compute class weights (only rank 0 computes, others load from cache)
    class_weights = None
    class_distribution = None
    if not args.no_class_weights:
        cache_path = os.path.join(train_dataset.processed_dir, 'class_weights.pt')

        if is_main:
            # Rank 0 computes and saves to cache
            class_weights, class_distribution = compute_class_weights(train_dataset)

            # Print class distribution
            print("\nClass distribution (train):")
            for name, info in class_distribution.items():
                print(f"  {name}: {info['count']} ({info['percentage']:.1f}%)")
            print(f"\nClass weights: {class_weights.tolist()}")

        # Synchronize: rank 0 finishes saving before others try to load
        barrier()

        if not is_main:
            # Other ranks load directly from cache file
            cached = torch.load(cache_path, weights_only=False)
            class_weights = cached['weights']

    # === Test Only Mode ===
    if args.test_only:
        if is_main:
            print("\n" + "=" * 60)
            print("Test evaluation only")
            print("=" * 60)

            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"No checkpoint found at {checkpoint_path}")

            # Load model
            # weights_only=False needed for PyTorch 2.6+ compatibility
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
                force_cpu=args.force_cpu,
            )

            # Test
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

        cleanup_distributed()
        return

    # === Training ===
    if is_main:
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
        force_cpu=args.force_cpu,
        train_sampler=train_sampler,
        rank=rank,
        world_size=world_size,
        use_wandb=args.wandb,
    )

    # Train
    _history = trainer.train()

    # Test (main process only)
    if is_main:
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

    # === Cleanup ===
    if args.wandb and is_main:
        import wandb
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
