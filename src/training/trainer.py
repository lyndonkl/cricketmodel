"""Training loop with MPS/CUDA support and optional DDP.

Best practices implemented from PyTorch documentation:
- model.train()/model.eval() mode switching
- torch.no_grad() for inference
- drop_last=True for consistent batch sizes
- Proper device handling for MPS/CUDA/CPU
- Gradient clipping
- Learning rate scheduling
- DDP support for multi-GPU training
"""

import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from ..config import Config
from ..data import CricketDataset
from ..model import CricketPredictor
from .metrics import compute_metrics, print_metrics


def get_device(preferred: str = "mps") -> torch.device:
    """Get best available device."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


class Trainer:
    """Training loop for cricket prediction model."""

    def __init__(
        self,
        config: Config,
        model: CricketPredictor | None = None,
        dataset: CricketDataset | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0

        # Device setup
        if self.is_distributed:
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = get_device(config.training.device)

        if self.is_main_process:
            print(f"Using device: {self.device}")
            if self.is_distributed:
                print(f"DDP: rank {rank}/{world_size}")

        # Load dataset if not provided
        if dataset is None:
            if self.is_main_process:
                print("Loading dataset...")
            dataset = CricketDataset(
                data_dir=Path(config.data.data_dir) / "t20s_male_json",
                history_len=config.model.temporal_seq_len,
                min_balls=config.data.min_balls_per_match,
            )
            if self.is_main_process:
                print(f"Loaded {len(dataset)} samples")
                print(f"Players: {dataset.num_players}, Venues: {dataset.num_venues}")

        self.dataset = dataset

        # Split dataset
        train_size = int(len(dataset) * config.data.train_split)
        val_size = int(len(dataset) * config.data.val_split)
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        # Create model if not provided
        if model is None:
            model = CricketPredictor(
                num_players=dataset.num_players,
                num_venues=dataset.num_venues,
                num_teams=dataset.num_teams,
                num_outcomes=config.model.num_outcomes,
                hidden_dim=config.model.transformer_dim,
                gat_heads=config.model.num_gat_heads,
                transformer_heads=config.model.transformer_heads,
                transformer_layers=config.model.transformer_layers,
                dropout=config.model.transformer_dropout,
                seq_len=config.model.temporal_seq_len,
            )

        self.model = model.to(self.device)

        # Wrap model in DDP if distributed
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(exist_ok=True)

    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """Create dataloader with appropriate settings."""
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.training.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=drop_last,
        )

    def _move_batch(self, batch: dict) -> dict:
        """Move batch to device."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc="Training", disable=not self.is_main_process)
        for batch in pbar:
            batch = self._move_batch(batch)

            self.optimizer.zero_grad()

            output = self.model(batch)
            loss = self.criterion(output["logits"], batch["label"])

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple[float, dict]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Evaluating", disable=not self.is_main_process):
            batch = self._move_batch(batch)

            output = self.model(batch)
            loss = self.criterion(output["logits"], batch["label"])

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(output["probs"].argmax(dim=-1).cpu())
            all_labels.append(batch["label"].cpu())
            all_probs.append(output["probs"].cpu())

        avg_loss = total_loss / num_batches

        # Compute metrics
        predictions = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        probs = torch.cat(all_probs)

        metrics = compute_metrics(predictions, labels, probs)

        return avg_loss, metrics

    def train(
        self,
        callback: Callable[[int, float, float], None] | None = None,
    ) -> None:
        """Full training loop."""
        train_loader = self._create_dataloader(
            self.train_dataset, shuffle=True, drop_last=True
        )
        val_loader = self._create_dataloader(
            self.val_dataset, shuffle=False, drop_last=False
        )

        best_val_loss = float("inf")

        for epoch in range(self.config.training.epochs):
            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics = self.evaluate(val_loader)

            # Update scheduler (after warmup)
            if epoch >= self.config.training.warmup_epochs:
                self.scheduler.step()

            if self.is_main_process:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}")
                print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Callback
            if callback:
                callback(epoch, train_loss, val_loss)

            # Save best model (main process only)
            if self.is_main_process:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best.pt")

                # Periodic checkpoint
                if (epoch + 1) % self.config.training.save_every == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final evaluation on test set
        if self.is_main_process:
            print("\nFinal evaluation on test set:")
            test_loader = self._create_dataloader(
                self.test_dataset, shuffle=False, drop_last=False
            )
            test_loss, test_metrics = self.evaluate(test_loader)
            print_metrics(test_metrics)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename

        # Get model state dict (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()

        torch.save({
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model state (handle DDP wrapper)
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint: {path}")


def train_ddp(rank: int, world_size: int, config: Config) -> None:
    """Training function for DDP."""
    setup_ddp(rank, world_size)

    try:
        trainer = Trainer(config, rank=rank, world_size=world_size)
        trainer.train()
    finally:
        cleanup_ddp()
