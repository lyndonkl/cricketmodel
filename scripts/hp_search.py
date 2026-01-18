#!/usr/bin/env python3
"""
Hyperparameter Search Script using Optuna + WandB

Implements Bayesian optimization with early pruning for efficient
hyperparameter tuning of the Cricket GNN model.

Usage:
    # Run specific phase
    python scripts/hp_search.py --phase phase1_coarse --n-trials 10

    # Run full search
    python scripts/hp_search.py --phase full --n-trials 50

    # Resume previous study
    python scripts/hp_search.py --phase full --n-trials 50 --study-name my_study

    # With reduced epochs for faster iteration
    python scripts/hp_search.py --phase phase1_coarse --epochs 20 --patience 5

    # Sequential tuning with best params from previous phase
    python scripts/hp_search.py --phase phase2_architecture --best-params results/phase1_best.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Optional

# Set environment variables before importing torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import set_seed
from src.data import create_dataloaders, compute_class_weights
from src.model import CricketHeteroGNN, ModelConfig, get_model_summary
from src.model.hetero_gnn import (
    CricketHeteroGNNWithPooling,
    CricketHeteroGNNHybrid,
    CricketHeteroGNNPhaseModulated,
    CricketHeteroGNNInningsConditional,
    CricketHeteroGNNFull,
)
from src.training import Trainer, TrainingConfig


# =============================================================================
# Model Class Registry
# =============================================================================

MODEL_CLASSES = {
    "CricketHeteroGNN": CricketHeteroGNN,
    "CricketHeteroGNNWithPooling": CricketHeteroGNNWithPooling,
    "CricketHeteroGNNHybrid": CricketHeteroGNNHybrid,
    "CricketHeteroGNNPhaseModulated": CricketHeteroGNNPhaseModulated,
    "CricketHeteroGNNInningsConditional": CricketHeteroGNNInningsConditional,
    "CricketHeteroGNNFull": CricketHeteroGNNFull,
}


# =============================================================================
# Search Space Definitions
# =============================================================================

SEARCH_SPACES = {
    "phase1_coarse": {
        "hidden_dim": {"type": "categorical", "values": [64, 128, 256]},
        "lr": {"type": "categorical", "values": [1e-4, 1e-3]},
    },
    "phase2_architecture": {
        "num_layers": {"type": "int", "low": 2, "high": 5},
        "num_heads": {"type": "categorical", "values": [2, 4, 8]},
    },
    "phase3_training": {
        "lr": {"type": "float", "low": 5e-4, "high": 2e-3, "log": True},
        "dropout": {"type": "categorical", "values": [0.0, 0.1, 0.2, 0.3]},
        "weight_decay": {"type": "categorical", "values": [0.0, 0.01, 0.05]},
    },
    "phase4_loss": {
        "focal_gamma": {"type": "categorical", "values": [0.0, 1.0, 2.0, 3.0]},
        "use_class_weights": {"type": "categorical", "values": [True, False]},
    },
    "full": {
        # All hyperparameters for comprehensive search
        "hidden_dim": {"type": "int", "low": 64, "high": 256, "step": 32},
        "num_layers": {"type": "int", "low": 2, "high": 5},
        "num_heads": {"type": "categorical", "values": [2, 4, 8]},
        "lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 0.1, "log": True},
        "focal_gamma": {"type": "float", "low": 0.0, "high": 3.0},
    },
    # === Model Variant Search Phases ===
    "model_variants": {
        # Compare different model architectures with fixed hyperparameters
        "model_class": {"type": "categorical", "values": [
            "CricketHeteroGNN",
            "CricketHeteroGNNHybrid",
            "CricketHeteroGNNInningsConditional",
            "CricketHeteroGNNFull",
        ]},
    },
    "model_variants_all": {
        # All model variants including pooling and phase-modulated
        "model_class": {"type": "categorical", "values": list(MODEL_CLASSES.keys())},
    },
    "model_with_hyperparams": {
        # Search model variants AND key hyperparameters together
        "model_class": {"type": "categorical", "values": [
            "CricketHeteroGNN",
            "CricketHeteroGNNHybrid",
            "CricketHeteroGNNFull",
        ]},
        "hidden_dim": {"type": "categorical", "values": [64, 128, 256]},
        "lr": {"type": "categorical", "values": [5e-4, 1e-3]},
    },
    "full_with_model": {
        # Comprehensive search including model variants
        "model_class": {"type": "categorical", "values": [
            "CricketHeteroGNN",
            "CricketHeteroGNNHybrid",
            "CricketHeteroGNNInningsConditional",
            "CricketHeteroGNNFull",
        ]},
        "hidden_dim": {"type": "int", "low": 64, "high": 256, "step": 32},
        "num_layers": {"type": "int", "low": 2, "high": 5},
        "num_heads": {"type": "categorical", "values": [2, 4, 8]},
        "lr": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 0.1, "log": True},
        "focal_gamma": {"type": "float", "low": 0.0, "high": 3.0},
    },
}

# Default hyperparameter values (used when not being searched)
DEFAULT_PARAMS = {
    "model_class": "CricketHeteroGNN",  # Default to base model
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "lr": 1e-3,
    "dropout": 0.1,
    "weight_decay": 0.01,
    "focal_gamma": 2.0,
    "use_class_weights": True,
}


def suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a hyperparameter value based on its specification."""
    param_type = spec["type"]

    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["values"])
    elif param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    elif param_type == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


# =============================================================================
# Objective Function Factory
# =============================================================================


def create_objective(
    train_loader,
    val_loader,
    test_loader,
    metadata: Dict,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
    search_space: Dict[str, Dict],
    base_params: Dict[str, Any],
    base_config: Dict[str, Any],
    checkpoint_base_dir: str,
    use_wandb: bool = False,
) -> Callable[[optuna.Trial], float]:
    """
    Factory that creates an objective function with data closures.

    Args:
        train_loader: Training data loader (pre-loaded)
        val_loader: Validation data loader (pre-loaded)
        test_loader: Test data loader (pre-loaded)
        metadata: Dataset metadata with entity counts
        class_weights: Pre-computed class weights tensor
        device: Device to train on
        search_space: Hyperparameter search space for this phase
        base_params: Base hyperparameters (from previous phases or defaults)
        base_config: Base training config (epochs, patience, etc.)
        checkpoint_base_dir: Base directory for trial checkpoints
        use_wandb: Whether to log individual trials to WandB

    Returns:
        Objective function for Optuna optimization
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            F1 macro score (to maximize)
        """
        # 1. Start with base params and override with suggested values
        params = base_params.copy()
        for param_name, param_spec in search_space.items():
            params[param_name] = suggest_param(trial, param_name, param_spec)

        # Extract params for model vs training config
        model_class_name = params.get("model_class", DEFAULT_PARAMS["model_class"])
        hidden_dim = params.get("hidden_dim", DEFAULT_PARAMS["hidden_dim"])
        num_layers = params.get("num_layers", DEFAULT_PARAMS["num_layers"])
        num_heads = params.get("num_heads", DEFAULT_PARAMS["num_heads"])
        dropout = params.get("dropout", DEFAULT_PARAMS["dropout"])
        lr = params.get("lr", DEFAULT_PARAMS["lr"])
        weight_decay = params.get("weight_decay", DEFAULT_PARAMS["weight_decay"])
        focal_gamma = params.get("focal_gamma", DEFAULT_PARAMS["focal_gamma"])
        use_class_weights_flag = params.get("use_class_weights", DEFAULT_PARAMS["use_class_weights"])

        # 2. Create trial-specific checkpoint directory
        trial_checkpoint_dir = os.path.join(checkpoint_base_dir, f"trial_{trial.number}")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)

        # 3. Create model using the selected model class
        model_class = MODEL_CLASSES[model_class_name]
        model_config = ModelConfig(
            num_venues=metadata["num_venues"],
            num_teams=metadata["num_teams"],
            num_players=metadata["num_players"],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        model = model_class(model_config)

        # 4. Create training config
        training_config = TrainingConfig(
            lr=lr,
            weight_decay=weight_decay,
            epochs=base_config["epochs"],
            patience=base_config["patience"],
            checkpoint_dir=trial_checkpoint_dir,
            use_focal_loss=True,
            focal_gamma=focal_gamma,
        )

        # 5. Determine class weights for this trial
        trial_class_weights = class_weights if use_class_weights_flag else None

        # 6. Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            class_weights=trial_class_weights,
            device=device,
            use_wandb=False,  # WandB handled at study level
        )

        # 7. Train with pruning callback
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"Model: {model_class_name}")
        print(f"Params: {params}")
        print(f"{'='*60}")

        try:
            # Custom training loop with pruning
            for epoch in range(base_config["epochs"]):
                trainer.epoch = epoch

                # Train one epoch
                train_loss, train_acc = trainer.train_epoch()

                # Validate
                val_loss, val_acc, val_labels, val_preds, val_probs = trainer.evaluate(
                    val_loader, desc=None  # Suppress progress bar for cleaner output
                )

                # Update scheduler
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                if epoch >= training_config.warmup_epochs:
                    if training_config.scheduler == "plateau":
                        trainer.scheduler.step(val_loss)
                    else:
                        trainer.scheduler.step()

                # Record history
                trainer.history["train_loss"].append(train_loss)
                trainer.history["train_acc"].append(train_acc)
                trainer.history["val_loss"].append(val_loss)
                trainer.history["val_acc"].append(val_acc)
                trainer.history["lr"].append(current_lr)

                # Report to Optuna for pruning (using validation loss as intermediate value)
                trial.report(val_loss, epoch)

                # Check for pruning
                if trial.should_prune():
                    print(f"  Trial {trial.number} pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()

                # Check for improvement (early stopping logic)
                improved = val_loss < trainer.best_val_loss - training_config.min_delta
                if improved:
                    trainer.best_val_loss = val_loss
                    trainer.best_val_acc = val_acc
                    trainer.patience_counter = 0
                    trainer.save_checkpoint("best_model.pt")
                else:
                    trainer.patience_counter += 1

                # Print progress every 5 epochs
                if (epoch + 1) % 5 == 0:
                    print(
                        f"  Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    )

                # Early stopping
                if trainer.patience_counter >= training_config.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            # 8. Evaluate on test set with best model
            from src.training.metrics import compute_metrics

            trainer.load_checkpoint("best_model.pt")
            test_loss, test_acc, test_labels, test_preds, test_probs = trainer.evaluate(
                test_loader, desc=None
            )
            test_metrics = compute_metrics(test_labels, test_preds, test_probs)

            f1_macro = test_metrics["f1_macro"]

            print(f"\n  Trial {trial.number} completed:")
            print(f"    Test F1 Macro: {f1_macro:.4f}")
            print(f"    Test F1 Weighted: {test_metrics['f1_weighted']:.4f}")
            print(f"    Test Accuracy: {test_acc:.4f}")

            # 9. Store additional metrics as trial attributes
            trial.set_user_attr("model_class", model_class_name)
            trial.set_user_attr("test_f1_weighted", test_metrics["f1_weighted"])
            trial.set_user_attr("test_accuracy", test_acc)
            trial.set_user_attr("test_loss", test_loss)
            trial.set_user_attr("best_val_loss", trainer.best_val_loss)
            trial.set_user_attr("epochs_trained", epoch + 1)

            return f1_macro

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed with error: {e}")
            raise

    return objective


# =============================================================================
# Study Management
# =============================================================================


def run_phase(
    phase_name: str,
    n_trials: int,
    train_loader,
    val_loader,
    test_loader,
    metadata: Dict,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
    base_config: Dict[str, Any],
    base_params: Dict[str, Any],
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "cricket-gnn-optuna",
    checkpoint_base_dir: str = "checkpoints/optuna",
    seed: int = 42,
    n_jobs: int = 1,
) -> optuna.Study:
    """
    Run Optuna study for a specific phase.

    Args:
        phase_name: Name of the phase (key in SEARCH_SPACES)
        n_trials: Number of trials to run
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        metadata: Dataset metadata
        class_weights: Pre-computed class weights
        device: Device to train on
        base_config: Base training config
        base_params: Base hyperparameters
        storage: Optuna storage URL (for persistence)
        study_name: Study name (for resuming)
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        checkpoint_base_dir: Base directory for checkpoints
        seed: Random seed

    Returns:
        Completed Optuna study
    """
    if phase_name not in SEARCH_SPACES:
        raise ValueError(f"Unknown phase: {phase_name}. Available: {list(SEARCH_SPACES.keys())}")

    search_space = SEARCH_SPACES[phase_name]

    # Generate study name if not provided
    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"cricket_gnn_{phase_name}_{timestamp}"

    # Create phase-specific checkpoint directory
    phase_checkpoint_dir = os.path.join(checkpoint_base_dir, study_name)
    os.makedirs(phase_checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Optuna Study: {study_name}")
    print(f"Phase: {phase_name}")
    print(f"Search space: {list(search_space.keys())}")
    print(f"N trials: {n_trials}")
    print(f"Checkpoint dir: {phase_checkpoint_dir}")
    print(f"{'='*60}\n")

    # Create sampler and pruner
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=5,  # Don't prune before epoch 5
        interval_steps=1,
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximize F1 macro
        load_if_exists=True,
    )

    # Create objective function
    objective = create_objective(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata=metadata,
        class_weights=class_weights,
        device=device,
        search_space=search_space,
        base_params=base_params,
        base_config=base_config,
        checkpoint_base_dir=phase_checkpoint_dir,
        use_wandb=use_wandb,
    )

    # Initialize WandB if enabled
    wandb_callback = None
    if use_wandb:
        try:
            import wandb
            from optuna.integration.wandb import WeightsAndBiasesCallback

            # Initialize WandB run for the study
            wandb.init(
                project=wandb_project,
                name=study_name,
                config={
                    "phase": phase_name,
                    "n_trials": n_trials,
                    "search_space": search_space,
                    "base_params": base_params,
                    "base_config": base_config,
                    "sampler": "TPESampler",
                    "pruner": "MedianPruner",
                },
            )

            wandb_callback = WeightsAndBiasesCallback(
                metric_name="f1_macro",
                wandb_kwargs={"project": wandb_project},
                as_multirun=True,
            )
            callbacks = [wandb_callback]
        except ImportError:
            print("Warning: wandb or optuna.integration.wandb not available")
            callbacks = []
    else:
        callbacks = []

    # Run optimization
    if n_jobs > 1:
        print(f"Running {n_jobs} trials in parallel (WandB logging disabled for parallel runs)")
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=callbacks if n_jobs == 1 else [],  # WandB doesn't work well with parallel
            show_progress_bar=True,
            n_jobs=n_jobs,
        )
    finally:
        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    # Print results
    print(f"\n{'='*60}")
    print("Study Complete!")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best F1 Macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best params to file
    results = {
        "study_name": study_name,
        "phase": phase_name,
        "best_trial": study.best_trial.number,
        "best_f1_macro": study.best_value,
        "best_params": study.best_params,
        "best_trial_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    results_path = os.path.join(phase_checkpoint_dir, "best_params.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return study


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Phase selection
    parser.add_argument(
        "--phase",
        type=str,
        default="full",
        choices=list(SEARCH_SPACES.keys()),
        help="Phase to run",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials",
    )

    # Training config
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epochs per trial",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/t20s_male_json",
        help="Raw data directory",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )

    # Optuna config
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (for resuming)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_studies.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel trials (default: 1). Note: each job loads full dataset into memory",
    )

    # Sequential tuning
    parser.add_argument(
        "--best-params",
        type=str,
        default=None,
        help="JSON file with best params from previous phase",
    )

    # WandB
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cricket-gnn-optuna",
        help="WandB project name",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/optuna",
        help="Base checkpoint directory",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    args.data_dir = os.path.abspath(args.data_dir)
    args.processed_dir = os.path.abspath(args.processed_dir)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Set seed
    set_seed(args.seed)

    # Detect device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load base params from previous phase if provided
    base_params = DEFAULT_PARAMS.copy()
    if args.best_params:
        with open(args.best_params, "r") as f:
            prev_results = json.load(f)
        print(f"Loading best params from: {args.best_params}")
        print(f"Previous best params: {prev_results['best_params']}")
        base_params.update(prev_results["best_params"])

    # Create base config
    base_config = {
        "epochs": args.epochs,
        "patience": args.patience,
    }

    # ==========================================================================
    # Load data ONCE (expensive operation)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    # Reduce DataLoader workers when running parallel trials to avoid "too many open files"
    num_workers = 0 if args.n_jobs > 1 else 4

    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.processed_dir,
        raw_data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=num_workers,
        min_history=1,
        seed=args.seed,
    )

    train_dataset = train_loader.dataset
    metadata = train_dataset.get_metadata()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Venues: {metadata['num_venues']}")
    print(f"Teams: {metadata['num_teams']}")
    print(f"Players: {metadata['num_players']}")

    # Compute class weights
    class_weights, class_distribution = compute_class_weights(train_dataset)
    print("\nClass distribution:")
    for name, info in class_distribution.items():
        print(f"  {name}: {info['count']} ({info['percentage']:.1f}%)")

    # ==========================================================================
    # Run Optuna study
    # ==========================================================================
    study = run_phase(
        phase_name=args.phase,
        n_trials=args.n_trials,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metadata=metadata,
        class_weights=class_weights,
        device=device,
        base_config=base_config,
        base_params=base_params,
        storage=args.storage,
        study_name=args.study_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        checkpoint_base_dir=args.checkpoint_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1 Macro: {study.best_value:.4f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Optuna visualization (if available)
    try:
        import optuna.visualization as vis

        # Create visualization directory
        viz_dir = os.path.join(args.checkpoint_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(viz_dir, f"{args.phase}_param_importance.html"))

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(viz_dir, f"{args.phase}_history.html"))

        # Parallel coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(viz_dir, f"{args.phase}_parallel.html"))

        print(f"\nVisualizations saved to: {viz_dir}")
    except ImportError:
        print("\nNote: Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")


if __name__ == "__main__":
    main()
