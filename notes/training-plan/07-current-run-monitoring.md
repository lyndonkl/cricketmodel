# Current Training Run Monitoring

## Run Details

| Field | Value |
|-------|-------|
| **Run ID** | `y28sy7aa` |
| **Start Time** | 2026-01-17 10:03:17 |
| **Configuration** | Default (hidden_dim=128, num_layers=3, num_heads=4) |
| **Epochs** | 100 |
| **Commit** | `3bbb8fb` (DDP unused parameters fix) |

---

## Current Status (Live)

Training is actively running with 14 DDP workers on Apple M3 Max (CPU mode).

### Latest Completed Epoch: 1

| Metric | Value | Target |
|--------|-------|--------|
| Train Loss | 1.3802 | decreasing |
| Val Loss | 1.3731 | decreasing |
| Train Acc | 0.2135 | - |
| Val Acc | 0.1455 | - |
| LR | 0.001 | - |

### Epoch 2 (In Progress)

- Train phase: Complete (loss=1.2898, acc=0.2027)
- Val phase: ~33% complete (loss=1.3458, acc=0.2035)

**Positive signs:**
- Train loss decreasing: 1.3802 → 1.2898 (improvement)
- Val accuracy improving: 0.1455 → 0.2035 (significant jump)
- Model saved best checkpoint at epoch 1

---

## Configuration Summary

```yaml
# Model
hidden_dim: 128
num_layers: 3
num_heads: 4
dropout: 0.1

# Training
lr: 0.001
weight_decay: 0.01
batch_size: 64
effective_batch_size: 896  # 64 × 14 workers
epochs: 100
patience: 10

# Loss
focal_gamma: 2.0
class_weights: balanced

# Data
train_samples: 544,685
val_samples: 67,264
test_samples: 69,281
```

---

## Decision Criteria

### At Epoch 15-20, evaluate f1_macro:

| Condition | Action |
|-----------|--------|
| f1_macro >= 0.15 | Proceed to Phase 1 hyperparameter tuning |
| 0.10 <= f1_macro < 0.15 | Continue training to epoch 50 |
| f1_macro < 0.10 | Investigate data/loss issues |

### Key Metrics to Watch

1. **val/f1_macro** - Primary decision metric (target: > 0.15 baseline)
2. **val/loss** - Should decrease consistently
3. **Per-class F1** - Especially for rare classes (three, wicket)

### Warning Signs

- val_loss increasing while train_loss decreasing → overfitting
- f1_macro stuck below 0.10 after epoch 10 → model not learning
- Any NaN/Inf losses → numerical instability

---

## Monitoring Commands

### Check current progress
```bash
tail -50 /Users/kushaldsouza/Documents/Projects/cricketmodel/wandb/run-20260117_100317-y28sy7aa/files/output.log
```

### Check if training is running
```bash
ps aux | grep "train.py" | grep -v grep
```

### View WandB dashboard
Visit: https://wandb.ai/cricket-gnn

---

## Next Steps After 15-20 Epochs

If proceeding to hyperparameter tuning (f1_macro >= 0.15):

### Phase 1: Coarse Grid Search (6 experiments)
```bash
for dim in 64 128 256; do
    for lr in 1e-4 1e-3; do
        python train.py --hidden-dim $dim --lr $lr --wandb --wandb-run-name phase1_dim${dim}_lr${lr}
    done
done
```

Reference: `notes/training-plan/01-hyperparameter-tuning.md`

---

## Log Updates

| Timestamp | Epoch | Val Loss | Val Acc | Notes |
|-----------|-------|----------|---------|-------|
| 2026-01-17 10:35 | 1 | 1.3731 | 0.1455 | Best model saved |
| 2026-01-17 11:40 | 2 (33%) | 1.3458 | 0.2035 | Val in progress |
