# Training Progression Guide

This document provides a stage-by-stage guide to training the CricketHeteroGNN model variants.

---

## Progression Overview

```
Stage 1: Base Model
    └── Establish baseline, tune core hyperparameters
         │
Stage 2: Parallel Comparison
    ├── WithPooling (ball pooling hypothesis)
    └── Hybrid (matchup hypothesis)
         │
Stage 3: Advanced Features (from Hybrid)
    ├── PhaseModulated (FiLM hypothesis)
    └── InningsConditional (innings heads hypothesis)
         │
Stage 4: Full Model
    └── Combine all + ablation studies
```

---

## Stage 1: Base Model

### Goal
Establish a working baseline and tune core hyperparameters.

### Steps

1. **Verify data pipeline**
   ```bash
   python -c "from src.data import HeteroDataBuilder; print('OK')"
   ```

2. **Train base model with defaults**
   ```bash
   python train.py \
       --model base \
       --hidden_dim 128 \
       --num_layers 3 \
       --num_heads 4 \
       --lr 1e-3 \
       --dropout 0.1 \
       --epochs 100 \
       --run_name stage1_base_default
   ```

3. **Tune hidden_dim**
   ```bash
   for dim in 64 128 256; do
       python train.py --model base --hidden_dim $dim --run_name stage1_dim$dim
   done
   ```

4. **Tune num_layers (with best dim)**
   ```bash
   for layers in 2 3 4 5; do
       python train.py --model base --hidden_dim 128 --num_layers $layers \
           --run_name stage1_layers$layers
   done
   ```

5. **Tune learning rate**
   ```bash
   for lr in 1e-4 5e-4 1e-3 2e-3; do
       python train.py --model base --hidden_dim 128 --num_layers 3 \
           --lr $lr --run_name stage1_lr$lr
   done
   ```

### Success Criteria
- Training completes without errors
- Validation loss decreases
- Macro F1 > 0.15 (baseline, not great yet)

### Checkpoint
Save the best hyperparameters:
```python
best_config_stage1 = {
    "hidden_dim": 128,  # Update based on results
    "num_layers": 3,
    "num_heads": 4,
    "lr": 1e-3,
    "dropout": 0.1,
}
```

---

## Stage 2: Parallel Comparison

### Goal
Test two competing hypotheses:
- **Pooling**: Explicit ball sequence attention helps
- **Hybrid**: Explicit matchup modeling helps

### Steps

Run both variants in parallel with Stage 1 best config:

```bash
# Pooling variant
python train.py \
    --model pooling \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --lr 1e-3 \
    --dropout 0.1 \
    --run_name stage2_pooling

# Hybrid variant
python train.py \
    --model hybrid \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --lr 1e-3 \
    --dropout 0.1 \
    --run_name stage2_hybrid
```

### Compare Results

| Variant | Macro F1 | Weighted F1 | Notes |
|---------|----------|-------------|-------|
| Base (Stage 1) | | | Baseline |
| WithPooling | | | Ball attention |
| Hybrid | | | Matchup MLP |

### Decision
- If **Pooling > Base** and **Pooling > Hybrid**: Use Pooling as base for Stage 3
- If **Hybrid > Base** and **Hybrid > Pooling**: Use Hybrid as base for Stage 3 (expected)
- If **Both ≈ Base**: Investigate why features aren't helping

### Expected Outcome
Hybrid should outperform because cricket outcomes are dominated by batter-bowler matchups.

---

## Stage 3: Advanced Features

### Goal
Test FiLM modulation and innings heads on top of Hybrid.

### Prerequisites
- Stage 2 complete
- Hybrid chosen as best variant

### Steps

Run both advanced variants:

```bash
# PhaseModulated (FiLM)
python train.py \
    --model phase_modulated \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --lr 1e-3 \
    --dropout 0.1 \
    --run_name stage3_film

# InningsConditional
python train.py \
    --model innings_conditional \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --lr 1e-3 \
    --dropout 0.1 \
    --run_name stage3_innings
```

### Additional Analysis

**For PhaseModulated**: Check per-phase performance
```python
# After training, analyze:
phases = ['powerplay', 'middle_1', 'middle_2', 'death_1', 'death_2']
for phase in phases:
    mask = test_df['phase'] == phase
    phase_f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
    print(f"{phase}: {phase_f1:.3f}")
```

**For InningsConditional**: Check per-innings performance
```python
for innings in [1, 2]:
    mask = test_df['innings'] == innings
    innings_f1 = f1_score(y_true[mask], y_pred[mask], average='macro')
    print(f"Innings {innings}: {innings_f1:.3f}")
```

### Compare Results

| Variant | Macro F1 | Powerplay F1 | Death F1 | Inn 1 F1 | Inn 2 F1 |
|---------|----------|--------------|----------|----------|----------|
| Hybrid (Stage 2) | | | | | |
| PhaseModulated | | | | | |
| InningsConditional | | | | | |

### Decision
- If **FiLM helps in phases**: Include in Full model
- If **Innings helps in chases**: Include in Full model
- Document which component helps more

---

## Stage 4: Full Model

### Goal
Combine all components and run ablation studies.

### Steps

1. **Train Full model with all features**
   ```bash
   python train.py \
       --model full \
       --use_film \
       --use_hierarchical \
       --use_innings_heads \
       --hidden_dim 128 \
       --num_layers 3 \
       --num_heads 4 \
       --lr 1e-3 \
       --dropout 0.1 \
       --run_name stage4_full_all
   ```

2. **Run ablations** (see `02-ablation-studies.md` for full details)
   ```bash
   # Without FiLM
   python train.py --model full --no_film --use_hierarchical --use_innings_heads \
       --run_name stage4_full_nofilm

   # Without innings heads
   python train.py --model full --use_film --use_hierarchical --no_innings_heads \
       --run_name stage4_full_noinnings

   # Without hierarchical
   python train.py --model full --use_film --no_hierarchical --use_innings_heads \
       --run_name stage4_full_nohierarchical
   ```

3. **Statistical significance testing** (3 seeds each)
   ```bash
   for seed in 42 123 456; do
       python train.py --model full --use_film --use_hierarchical --use_innings_heads \
           --seed $seed --run_name stage4_full_s$seed
   done
   ```

### Final Model Selection

Based on ablation results, determine the final production configuration:

```python
final_config = {
    "model": "full",
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.1,
    "use_film": True,  # If ablation shows it helps
    "use_hierarchical": True,  # If ablation shows it helps
    "use_innings_heads": True,  # If ablation shows it helps
}
```

---

## Training Commands Reference

### Full Training Script Template

```python
# train.py (conceptual structure)
import argparse
import torch
import wandb
from src.model.hetero_gnn import (
    CricketHeteroGNN,
    CricketHeteroGNNWithPooling,
    CricketHeteroGNNHybrid,
    CricketHeteroGNNPhaseModulated,
    CricketHeteroGNNInningsConditional,
    CricketHeteroGNNFull,
)

MODEL_CLASSES = {
    "base": CricketHeteroGNN,
    "pooling": CricketHeteroGNNWithPooling,
    "hybrid": CricketHeteroGNNHybrid,
    "phase_modulated": CricketHeteroGNNPhaseModulated,
    "innings_conditional": CricketHeteroGNNInningsConditional,
    "full": CricketHeteroGNNFull,
}

def main(args):
    # Set seed
    set_seed(args.seed)

    # Init WandB
    wandb.init(project="cricket-gnn", name=args.run_name, config=vars(args))

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)

    # Create model
    ModelClass = MODEL_CLASSES[args.model]
    model = ModelClass(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(optimizer, args)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            **{f"val/{k}": v for k, v in val_metrics.items()}
        })

        scheduler.step()

        if early_stopping(val_metrics):
            break

    # Final test
    test_metrics = evaluate(model, test_loader)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    # Save model
    torch.save(model.state_dict(), f"checkpoints/{args.run_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    # Full model flags
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument("--no_film", action="store_true")
    parser.add_argument("--use_hierarchical", action="store_true")
    parser.add_argument("--no_hierarchical", action="store_true")
    parser.add_argument("--use_innings_heads", action="store_true")
    parser.add_argument("--no_innings_heads", action="store_true")

    args = parser.parse_args()
    main(args)
```

---

## Multi-GPU Training with DDP

For faster training on multiple GPUs, use `torchrun`:

### Basic DDP Commands

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train.py \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 100 \
    --batch-size 64

# 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 100 \
    --batch-size 32

# All available GPUs
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py
```

### DDP with Hyperparameter Sweeps

```bash
# Stage 1: Tune hidden_dim with 2 GPUs
for dim in 64 128 256; do
    torchrun --nproc_per_node=2 train.py --hidden-dim $dim
done

# Stage 4: Full model ablations with 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --model full \
    --use_film \
    --use_hierarchical \
    --use_innings_heads \
    --batch-size 32
```

### Effective Batch Size

With DDP, effective batch = batch_size * num_gpus:

| GPUs | --batch-size | Effective Batch |
|------|--------------|-----------------|
| 1    | 64           | 64              |
| 2    | 64           | 128             |
| 4    | 32           | 128             |
| 8    | 16           | 128             |

### WandB with DDP

When using WandB with DDP, only rank 0 should log to avoid duplicates:

```python
from src.training import is_main_process

if is_main_process():
    wandb.init(project="cricket-gnn", name=run_name)
    # ... logging ...
    wandb.finish()
```

See `notes/training-plan/06-distributed-training.md` for comprehensive DDP documentation.

---

## Checkpoints and Saving

### When to Save

- Best validation Macro F1
- Best validation loss
- Final epoch

```python
# Save best model
if val_macro_f1 > best_macro_f1:
    best_macro_f1 = val_macro_f1
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_macro_f1': val_macro_f1,
        'config': config,
    }, f"checkpoints/best_{run_name}.pt")
```

### Loading for Inference

```python
# Load model
checkpoint = torch.load("checkpoints/best_stage4_full.pt")
config = checkpoint['config']

model = CricketHeteroGNNFull(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Resource Requirements

### Memory Estimates

| Variant | hidden_dim=128 | hidden_dim=256 |
|---------|----------------|----------------|
| Base | ~2 GB | ~4 GB |
| WithPooling | ~2.2 GB | ~4.5 GB |
| Hybrid | ~2.5 GB | ~5 GB |
| PhaseModulated | ~3 GB | ~6 GB |
| Full | ~3.5 GB | ~7 GB |

### Training Time Estimates (per epoch, batch_size=64)

| Dataset Size | Base | Full |
|--------------|------|------|
| 10K samples | ~30 sec | ~60 sec |
| 100K samples | ~5 min | ~10 min |
| 1M samples | ~50 min | ~100 min |

---

## Progress Tracking Template

### Stage 1 Checklist

- [ ] Data pipeline working
- [ ] Base model trains without errors
- [ ] Validation loss decreases
- [ ] Hyperparameter tuning complete
- [ ] Best config saved

### Stage 2 Checklist

- [ ] Pooling variant trained
- [ ] Hybrid variant trained
- [ ] Results compared
- [ ] Winner selected

### Stage 3 Checklist

- [ ] PhaseModulated trained
- [ ] InningsConditional trained
- [ ] Per-phase analysis done
- [ ] Per-innings analysis done

### Stage 4 Checklist

- [ ] Full model trained
- [ ] Ablations complete (9 experiments)
- [ ] 3 seeds per experiment
- [ ] Statistical significance tested
- [ ] Final config determined
- [ ] Production model saved
