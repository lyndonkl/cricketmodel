"""
Trainer for Cricket Ball Prediction Model

Handles training loop, validation, early stopping, and checkpointing.
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, print_classification_report


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
    scheduler: str = 'cosine'  # 'cosine' or 'plateau'

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True

    # Logging
    log_interval: int = 100  # batches
    eval_interval: int = 1  # epochs


class Trainer:
    """
    Training manager for CricketHeteroGNN.

    Handles:
    - Training loop with progress tracking
    - Validation with early stopping
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: CricketHeteroGNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            class_weights: Optional class weights for imbalanced data
            device: Device to train on
        """
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Model
        self.model = model.to(self.device)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        if config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs - config.warmup_epochs,
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
            )

        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.epoch = 0
        self.global_step = 0

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

        # Create checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} [Train]')

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            self.global_step += 1

            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / total:.4f}',
                })

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        desc: str = 'Eval'
    ) -> Tuple[float, float, List[int], List[int], np.ndarray]:
        """
        Evaluate on a data loader.

        Returns:
            (average_loss, accuracy, all_labels, all_preds, all_probs)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(loader, desc=desc)

        for batch in pbar:
            batch = batch.to(self.device)

            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)

            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=1)

            total_loss += loss.item() * batch.y.size(0)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            all_labels.extend(batch.y.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
            all_probs.append(probs.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{total_loss / total:.4f}',
                'acc': f'{correct / total:.4f}',
            })

        avg_loss = total_loss / total
        accuracy = correct / total
        all_probs = np.concatenate(all_probs, axis=0)

        return avg_loss, accuracy, all_labels, all_preds, all_probs

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history dict
        """
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_labels, val_preds, val_probs = self.evaluate(
                self.val_loader, f'Epoch {epoch + 1} [Val]'
            )

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            if epoch >= self.config.warmup_epochs:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Check for improvement
            improved = val_loss < self.best_val_loss - self.config.min_delta

            if improved:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0

                # Save best model
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")

            # Save periodic checkpoint
            if not self.config.save_best_only:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                break

        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        self._save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': asdict(self.config),
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']

        print(f"Loaded checkpoint from epoch {self.epoch + 1}")

    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.config.checkpoint_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def test(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate on test set with detailed metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dict with test metrics
        """
        print("\nEvaluating on test set...")

        # Load best model
        self.load_checkpoint('best_model.pt')

        # Evaluate
        test_loss, test_acc, labels, preds, probs = self.evaluate(
            test_loader, 'Test'
        )

        # Compute detailed metrics
        metrics = compute_metrics(labels, preds, probs)
        metrics['loss'] = test_loss

        # Print report
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print_classification_report(labels, preds, probs)

        return metrics


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    **config_kwargs
) -> Trainer:
    """
    Convenience function to create a trainer.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Optional class weights
        **config_kwargs: Override default config values

    Returns:
        Configured Trainer
    """
    config = TrainingConfig(**config_kwargs)
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
    )
