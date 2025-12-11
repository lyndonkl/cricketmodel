"""Training loop with MPS support."""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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


class Trainer:
    """Training loop for cricket prediction model."""

    def __init__(
        self,
        config: Config,
        model: CricketPredictor | None = None,
        dataset: CricketDataset | None = None,
    ):
        self.config = config
        self.device = get_device(config.training.device)
        print(f"Using device: {self.device}")

        # Load dataset if not provided
        if dataset is None:
            print("Loading dataset...")
            dataset = CricketDataset(
                data_dir=Path(config.data.data_dir) / "t20s_male_json",
                history_len=config.model.temporal_seq_len,
                min_balls=config.data.min_balls_per_match,
            )
            print(f"Loaded {len(dataset)} samples")
            print(f"Players: {dataset.num_players}, Venues: {dataset.num_venues}")

        self.dataset = dataset

        # Split dataset
        train_size = int(len(dataset) * config.data.train_split)
        val_size = int(len(dataset) * config.data.val_split)
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
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
                gat_layers=3,
                transformer_heads=config.model.transformer_heads,
                transformer_layers=config.model.transformer_layers,
                dropout=config.model.transformer_dropout,
                seq_len=config.model.temporal_seq_len,
            )

        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
        )

        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _create_dataloader(
        self,
        dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create dataloader with appropriate settings."""
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True if self.device.type != "mps" else False,
        )

    def _move_batch(self, batch: dict) -> dict:
        """Move batch to device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
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

        for batch in tqdm(dataloader, desc="Evaluating"):
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
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        val_loader = self._create_dataloader(self.val_dataset, shuffle=False)

        best_val_loss = float("inf")

        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics.accuracy:.4f}")

            # Callback
            if callback:
                callback(epoch, train_loss, val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best.pt")

            # Periodic checkpoint
            if (epoch + 1) % self.config.training.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final evaluation on test set
        print("\nFinal evaluation on test set:")
        test_loader = self._create_dataloader(self.test_dataset, shuffle=False)
        test_loss, test_metrics = self.evaluate(test_loader)
        print_metrics(test_metrics)

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint: {path}")
