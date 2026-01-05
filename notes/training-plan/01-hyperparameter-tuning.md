# Hyperparameter Tuning Guide

This document outlines the hyperparameter search strategy for CricketHeteroGNN models.

---

## Hyperparameter Priority

Tune hyperparameters in this order (by impact on performance):

| Priority | Hyperparameter | Why First? |
|----------|----------------|------------|
| 1 | `hidden_dim` | Determines model capacity |
| 2 | `num_layers` | Depth vs over-smoothing trade-off |
| 3 | `num_heads` | Attention expressiveness |
| 4 | `lr` | Most impactful training dynamic |
| 5 | `dropout` | Primary regularization |
| 6 | `weight_decay` | Secondary regularization |

---

## Search Spaces

### Model Architecture

| HP | Search Space | Default | Notes |
|----|--------------|---------|-------|
| `hidden_dim` | [64, 128, 256] | 128 | Start middle, adjust based on overfit/underfit |
| `num_layers` | [2, 3, 4, 5] | 3 | More layers risk over-smoothing |
| `num_heads` | [2, 4, 8] | 4 | Must divide hidden_dim evenly |

### Training Dynamics

| HP | Search Space | Default | Notes |
|----|--------------|---------|-------|
| `lr` | [1e-4, 5e-4, 1e-3, 2e-3] | 1e-3 | Log scale search |
| `dropout` | [0.0, 0.1, 0.2, 0.3] | 0.1 | Start low |
| `weight_decay` | [0.0, 0.01, 0.05] | 0.01 | L2 regularization |
| `batch_size` | [32, 64, 128] | 64 | Memory dependent |

### Loss Function

| HP | Search Space | Default | Notes |
|----|--------------|---------|-------|
| `focal_gamma` | [0.0, 1.0, 2.0, 3.0] | 2.0 | 0 = CE, higher = focus on hard examples |
| `class_weights` | [None, balanced, custom] | balanced | Address class imbalance |

---

## Default Starting Configuration

```python
config = {
    # Model
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.1,

    # Training
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "epochs": 100,

    # Loss
    "focal_gamma": 2.0,
    "class_weights": "balanced",

    # Scheduler
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 1e-6,

    # Early stopping
    "patience": 15,
    "min_delta": 0.001,
}
```

---

## Tuning Strategy

### Phase 1: Coarse Grid Search

Start with coarse search on most impactful HPs:

```python
# Grid for Phase 1
grid_phase_1 = {
    "hidden_dim": [64, 128, 256],
    "lr": [1e-4, 1e-3],
}
# 6 experiments total
```

### Phase 2: Refine Architecture

Based on Phase 1 winner, tune depth:

```python
# Grid for Phase 2
grid_phase_2 = {
    "hidden_dim": [best_from_phase_1],
    "num_layers": [2, 3, 4, 5],
    "num_heads": [2, 4, 8],
}
# 12 experiments total
```

### Phase 3: Refine Training

With architecture fixed, tune training dynamics:

```python
# Grid for Phase 3
grid_phase_3 = {
    "lr": [5e-4, 1e-3, 2e-3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "weight_decay": [0.0, 0.01, 0.05],
}
# 36 experiments total
```

### Phase 4: Fine-tune Loss

Finally, tune loss function:

```python
# Grid for Phase 4
grid_phase_4 = {
    "focal_gamma": [0.0, 1.0, 2.0, 3.0],
    "class_weights": [None, "balanced"],
}
# 8 experiments total
```

---

## Implementation with WandB Sweeps

### Create Sweep Config

```yaml
# sweep_config.yaml
program: train.py
method: bayes
metric:
  name: val/macro_f1
  goal: maximize
parameters:
  hidden_dim:
    values: [64, 128, 256]
  num_layers:
    values: [2, 3, 4, 5]
  num_heads:
    values: [2, 4, 8]
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 2e-3
  dropout:
    values: [0.0, 0.1, 0.2, 0.3]
  weight_decay:
    values: [0.0, 0.01, 0.05]
early_terminate:
  type: hyperband
  min_iter: 10
  eta: 3
```

### Run Sweep

```bash
# Initialize sweep
wandb sweep sweep_config.yaml

# Launch agents (run on multiple machines if available)
wandb agent <sweep_id>
```

---

## Hyperparameter Interactions

### hidden_dim × num_heads

The number of heads must divide hidden_dim evenly:

| hidden_dim | Valid num_heads |
|------------|-----------------|
| 64 | 2, 4, 8 |
| 128 | 2, 4, 8 |
| 256 | 2, 4, 8 |

### num_layers × dropout

Deeper networks need more regularization:

| num_layers | Recommended dropout |
|------------|---------------------|
| 2 | 0.0 - 0.1 |
| 3 | 0.1 - 0.2 |
| 4+ | 0.2 - 0.3 |

### lr × batch_size

Scale learning rate with batch size:

| batch_size | Recommended lr |
|------------|----------------|
| 32 | 5e-4 - 1e-3 |
| 64 | 1e-3 - 2e-3 |
| 128 | 2e-3 - 5e-3 |

---

## Over-Smoothing Prevention

GNNs suffer from over-smoothing with too many layers. Monitor:

```python
# Add to training loop
def compute_smoothness(node_embeddings):
    """Lower = more similar (bad)"""
    norms = torch.norm(node_embeddings, dim=-1, keepdim=True)
    normalized = node_embeddings / (norms + 1e-8)
    similarity = torch.mm(normalized, normalized.t())
    return 1 - similarity.mean()

# Log per epoch
wandb.log({"smoothness": compute_smoothness(x_dict["player"])})
```

**Warning signs**:
- Smoothness dropping below 0.3
- All node embeddings converging to same vector
- Performance degrading after layer 3-4

---

## Learning Rate Scheduling

### Cosine Annealing with Warmup (Recommended)

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# With warmup
def adjust_learning_rate(optimizer, epoch, warmup_epochs=5, base_lr=1e-3):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        lr = scheduler.get_last_lr()[0]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

### ReduceLROnPlateau (Alternative)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize F1
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# After validation
scheduler.step(val_macro_f1)
```

---

## Practical Tips

### 1. Start Simple

```python
# Week 1: Basic config
config = {
    "hidden_dim": 128,
    "num_layers": 3,
    "num_heads": 4,
    "lr": 1e-3,
    "dropout": 0.1,
}
```

### 2. One Variable at a Time

Don't change multiple HPs simultaneously. If performance changes, you won't know which HP caused it.

### 3. Track Everything

```python
wandb.config.update(config)
wandb.log({
    "epoch": epoch,
    "train/loss": train_loss,
    "train/acc": train_acc,
    "val/loss": val_loss,
    "val/macro_f1": val_macro_f1,
    "val/weighted_f1": val_weighted_f1,
    "lr": scheduler.get_last_lr()[0],
})
```

### 4. Use Seeds for Reproducibility

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Run each HP config with 3 seeds
for seed in [42, 123, 456]:
    set_seed(seed)
    train(config)
```

### 5. Budget Your Experiments

| Phase | Experiments | Time per Exp | Total |
|-------|-------------|--------------|-------|
| Coarse grid | 6 | 30 min | 3 hours |
| Architecture | 12 | 30 min | 6 hours |
| Training | 36 | 30 min | 18 hours |
| Loss | 8 | 30 min | 4 hours |
| **Total** | **62** | | **31 hours** |

With 3 seeds each: ~93 hours of training.

---

## Quick Reference

```bash
# Tune hidden_dim first
python train.py --hidden_dim 64 --run_name dim64
python train.py --hidden_dim 128 --run_name dim128
python train.py --hidden_dim 256 --run_name dim256

# Then tune layers with best dim
python train.py --hidden_dim 128 --num_layers 2 --run_name layers2
python train.py --hidden_dim 128 --num_layers 3 --run_name layers3
python train.py --hidden_dim 128 --num_layers 4 --run_name layers4

# Then tune lr with best architecture
python train.py --hidden_dim 128 --num_layers 3 --lr 5e-4 --run_name lr5e4
python train.py --hidden_dim 128 --num_layers 3 --lr 1e-3 --run_name lr1e3
python train.py --hidden_dim 128 --num_layers 3 --lr 2e-3 --run_name lr2e3
```
