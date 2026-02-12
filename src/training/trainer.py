"""
Trainer for Cricket Ball Prediction Model

Handles training loop, validation, early stopping, and checkpointing.
Supports both single-GPU and Distributed Data Parallel (DDP) training.

DDP Usage:
    torchrun --nproc_per_node=2 train.py
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .losses import ScoreRegressionLoss

if TYPE_CHECKING:
    from torch.utils.data.distributed import DistributedSampler


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

    # Loss function
    huber_delta: float = 10.0  # Huber loss transition point (MSE for errors < delta, MAE for larger)

    # Mixed precision
    use_amp: bool = False  # Enable Automatic Mixed Precision (CUDA only)


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
        device: Optional[torch.device] = None,
        force_cpu: bool = False,
        # DDP parameters
        train_sampler: Optional["DistributedSampler"] = None,
        rank: int = 0,
        world_size: int = 1,
        # WandB
        use_wandb: bool = False,
    ):
        """
        Args:
            model: CricketHeteroGNN model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            force_cpu: Force CPU usage (for DDP with Gloo backend)
            train_sampler: DistributedSampler for DDP (to call set_epoch)
            rank: Process rank (0 for main process)
            world_size: Total number of processes
            use_wandb: Enable WandB logging (only on rank 0)
        """
        self.config = config
        self.use_wandb = use_wandb

        # DDP configuration
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main = rank == 0
        self.train_sampler = train_sampler

        # Device setup - handles CUDA, MPS (Apple Silicon), and CPU
        if device is not None:
            self.device = device
        elif force_cpu:
            self.device = torch.device('cpu')
        elif self.is_distributed:
            # In DDP, assign device based on platform
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{rank}')
            else:
                # Gloo backend (non-CUDA DDP) requires CPU tensors
                self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Model - wrap with DDP if distributed
        self.model = model.to(self.device)

        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # device_ids and output_device are CUDA-specific
            # For CPU/MPS, omit these parameters
            if torch.cuda.is_available():
                self.model = DDP(
                    self.model,
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True,
                )
            else:
                # CPU or MPS: don't specify device_ids
                self.model = DDP(
                    self.model,
                    find_unused_parameters=True,
                )

        # Data - store batch_size before wrapping (PrefetchLoader doesn't expose it)
        self.batch_size = train_loader.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader

        # GPU prefetching - async transfer of next batch while current processes
        if str(self.device).startswith('cuda'):
            from torch_geometric.loader import PrefetchLoader
            self.train_loader = PrefetchLoader(self.train_loader, device=self.device)
            self.val_loader = PrefetchLoader(self.val_loader, device=self.device)
            if self.is_main:
                print("GPU prefetching enabled")

        # Loss - Huber (Smooth L1) loss for score regression
        self.criterion = ScoreRegressionLoss(delta=config.huber_delta)
        if self.is_main:
            print(f"Using Huber (SmoothL1) loss for score regression (delta={config.huber_delta})")

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
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
        self.patience_counter = 0
        self.epoch = 0
        self.global_step = 0

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }

        # AMP (Automatic Mixed Precision) setup
        self.use_amp = config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            if self.is_main:
                print("AMP enabled: using mixed precision training")
        else:
            self.scaler = None
            if config.use_amp and not torch.cuda.is_available() and self.is_main:
                print("Warning: AMP requested but CUDA not available, running in FP32")

        # Create checkpoint dir (only on main process)
        if self.is_main:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            average_loss
        """
        self.model.train()

        total_loss = 0.0
        total = 0

        # Only show progress bar on main process
        loader = self.train_loader
        if self.is_main:
            loader = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} [Train]')

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)

            # Forward pass with optional AMP autocast
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch.y)

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct norm)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)

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
            total += batch.y.size(0)

            self.global_step += 1

            # Update progress bar (main process only)
            if self.is_main and batch_idx % 10 == 0:
                loader.set_postfix({
                    'loss': f'{loss.item():.4f}',
                })

        avg_loss = total_loss / total

        # In DDP, aggregate metrics across all processes
        if self.is_distributed:
            from .distributed import reduce_tensor
            avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
            avg_loss = reduce_tensor(avg_loss_tensor).item()

        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        desc: Optional[str] = 'Eval'
    ) -> Tuple[float, Dict]:
        """
        Evaluate on a data loader with regression metrics.

        Args:
            loader: DataLoader to evaluate on
            desc: Description for progress bar (None to suppress progress bar)

        Returns:
            (average_loss, metrics)
            metrics dict contains: mae, rmse, r_squared
        """
        self.model.eval()

        total_loss = 0.0
        total = 0

        # Regression accumulators
        total_se = 0.0     # sum of squared errors
        total_ae = 0.0     # sum of absolute errors
        sum_targets = 0.0
        sum_sq_targets = 0.0

        # Only show progress bar on main process and if desc is provided
        eval_loader = loader
        if self.is_main and desc is not None:
            eval_loader = tqdm(loader, desc=desc)

        for batch in eval_loader:
            batch = batch.to(self.device)

            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)

            # Accumulate regression metrics
            score_pred = outputs['score'].squeeze(-1)
            targets = batch.y.float()
            total_se += ((score_pred - targets) ** 2).sum().item()
            total_ae += (score_pred - targets).abs().sum().item()
            sum_targets += targets.sum().item()
            sum_sq_targets += (targets ** 2).sum().item()

            total_loss += loss.item() * batch.y.size(0)
            total += batch.y.size(0)

            if self.is_main and desc is not None:
                eval_loader.set_postfix({
                    'loss': f'{total_loss / total:.4f}',
                })

        avg_loss = total_loss / total

        # Compute regression metrics
        mae = total_ae / total
        rmse = (total_se / total) ** 0.5
        mean_target = sum_targets / total
        ss_tot = sum_sq_targets - total * mean_target ** 2
        r_squared = 1 - (total_se / ss_tot) if ss_tot > 0 else 0.0

        head_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
        }

        # In DDP, aggregate metrics across all processes
        if self.is_distributed:
            from .distributed import reduce_tensor
            avg_loss_tensor = torch.tensor([avg_loss], device=self.device)
            avg_loss = reduce_tensor(avg_loss_tensor).item()

            mae_tensor = torch.tensor([mae], device=self.device)
            rmse_tensor = torch.tensor([rmse], device=self.device)
            r2_tensor = torch.tensor([r_squared], device=self.device)

            head_metrics['mae'] = reduce_tensor(mae_tensor).item()
            head_metrics['rmse'] = reduce_tensor(rmse_tensor).item()
            head_metrics['r_squared'] = reduce_tensor(r2_tensor).item()

        return avg_loss, head_metrics

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Training history dict
        """
        if self.is_main:
            print(f"Training on {self.device}")
            print(f"Train batches: {len(self.train_loader)}")
            print(f"Val batches: {len(self.val_loader)}")
            if self.is_distributed:
                print(f"World size: {self.world_size}")
                print(f"Effective batch size: {self.batch_size * self.world_size}")
            print()

        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # CRITICAL: Set epoch on sampler for proper shuffling in DDP
            # This ensures each epoch uses a different random ordering
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # Train
            train_loss = self.train_epoch()

            # Validate (all processes, but only main logs)
            val_desc = f'Epoch {epoch + 1} [Val]' if self.is_main else None
            val_loss, head_metrics = self.evaluate(self.val_loader, val_desc)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            if epoch >= self.config.warmup_epochs:
                if self.config.scheduler == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history (all processes for consistency)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)

            # Print epoch summary (main process only)
            if self.is_main:
                print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  MAE: {head_metrics['mae']:.2f}  RMSE: {head_metrics['rmse']:.2f}  R²: {head_metrics['r_squared']:.4f}")
                print(f"  LR: {current_lr:.6f}")

            # Log to WandB (main process only)
            if self.is_main and self.use_wandb:
                import wandb

                wandb_log = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr,
                    "metrics/mae": head_metrics["mae"],
                    "metrics/rmse": head_metrics["rmse"],
                    "metrics/r_squared": head_metrics["r_squared"],
                }

                wandb.log(wandb_log)

            # Check for improvement
            improved = val_loss < self.best_val_loss - self.config.min_delta

            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model (main process only)
                if self.is_main:
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.is_main:
                    print(f"  No improvement for {self.patience_counter} epochs")

            # Synchronize before next epoch (ensures checkpoint is saved)
            if self.is_distributed:
                from .distributed import barrier
                barrier()

            # Save periodic checkpoint (main process only)
            if not self.config.save_best_only and self.is_main:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

            # Early stopping
            if self.patience_counter >= self.config.patience:
                if self.is_main:
                    print(f"\n[warning] Early stopping triggered after {epoch + 1} epochs")
                break

        elapsed_time = time.time() - start_time
        if self.is_main:
            print(f"\nTraining completed in {elapsed_time / 60:.1f} minutes")
            print(f"Best validation loss: {self.best_val_loss:.4f}")

            # Save training history
            self._save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Handles DDP by extracting the underlying model via model.module
        if wrapped in DistributedDataParallel.
        """
        path = os.path.join(self.config.checkpoint_dir, filename)

        # Get underlying model if wrapped in DDP
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module

        checkpoint_dict = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }

        # Save scaler state if using AMP
        if self.scaler is not None:
            checkpoint_dict['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint_dict, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        # weights_only=False needed for PyTorch 2.6+ compatibility
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Load scaler state if available and using AMP
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.epoch + 1}")

    def _save_history(self):
        """Save training history to JSON."""
        path = os.path.join(self.config.checkpoint_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def test(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate on test set with detailed regression metrics.

        Args:
            test_loader: Test data loader

        Returns:
            Dict with test metrics (mae, rmse, r_squared)
        """
        # Wrap with prefetch if on GPU
        if str(self.device).startswith('cuda'):
            from torch_geometric.loader import PrefetchLoader
            test_loader = PrefetchLoader(test_loader, device=self.device)

        print("\nEvaluating on test set...")

        # Load best model
        self.load_checkpoint('best_model.pt')

        # Evaluate
        test_loss, head_metrics = self.evaluate(test_loader, 'Test')

        # Build metrics dict
        metrics = {
            'loss': test_loss,
            'mae': head_metrics['mae'],
            'rmse': head_metrics['rmse'],
            'r_squared': head_metrics['r_squared'],
        }

        # Print report
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"MAE: {head_metrics['mae']:.2f}")
        print(f"RMSE: {head_metrics['rmse']:.2f}")
        print(f"R²: {head_metrics['r_squared']:.4f}")

        # Log to WandB
        if self.use_wandb:
            import wandb

            wandb_log = {
                "test/loss": test_loss,
                "test/mae": head_metrics["mae"],
                "test/rmse": head_metrics["rmse"],
                "test/r_squared": head_metrics["r_squared"],
            }

            wandb.log(wandb_log)

        return metrics


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    **config_kwargs
) -> Trainer:
    """
    Convenience function to create a trainer.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
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
    )
