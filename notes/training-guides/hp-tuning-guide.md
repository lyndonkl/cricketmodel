# Hyperparameter Tuning Guide

This document covers Optuna + WandB integration for hyperparameter optimization and how to run training for the Cricket Ball Prediction GNN model.

---

## 1. What Optuna Does

Optuna performs **Bayesian hyperparameter optimization**:

1. **TPE Sampler:** Uses past trial results to intelligently suggest next hyperparameters (not random search)
2. **MedianPruner:** Stops underperforming trials early to save compute
3. **Study Persistence:** Saves results to SQLite for resumption

### Pruning Logic

```
- First 5 trials: Always complete (establish baseline)
- After warmup (5 epochs): Compare val_loss to median of completed trials
- If below median: Continue
- If above median: Prune (stop early)
```

---

## 2. Search Phases

| Phase | Parameters Searched | Purpose |
|-------|---------------------|---------|
| `phase1_coarse` | hidden_dim, lr | Find ballpark model size and learning rate |
| `phase2_architecture` | num_layers, num_heads | Tune depth and attention |
| `phase3_training` | lr, dropout, weight_decay | Regularization tuning |
| `phase4_loss` | focal_gamma, use_class_weights | Loss function tuning |
| `model_variants` | model_class | Compare architectures |
| `full` | All parameters | Joint optimization |
| `full_with_model` | All + model_class | Comprehensive search |

---

## 3. What You See in WandB During HP Search

### Study-Level Dashboard
- **Optimization History:** Trial number vs F1 macro (shows if Optuna is finding better configs)
- **Parameter Importance:** Which hyperparameters matter most
- **Parallel Coordinates:** Trace good trials to see parameter patterns

### Trial-Level Metrics
Each trial logs:
- Trial number and hyperparameters
- Final test F1 macro
- Whether it was pruned
- Epochs trained

---

## 4. Generated Visualizations

After study completion, HTML files in `checkpoints/optuna/visualizations/`:

| File | What It Shows |
|------|---------------|
| `{phase}_param_importance.html` | Bar chart of hyperparameter importance |
| `{phase}_history.html` | F1 vs trial number (should trend upward) |
| `{phase}_parallel.html` | Interactive parallel coordinates plot |

---

## 5. How to Run Training

### Regular Training

```bash
# Basic training with WandB
python train.py --wandb --wandb-project cricket-gnn

# With custom hyperparameters
python train.py --hidden-dim 128 --num-layers 3 --lr 1e-3 --dropout 0.2 --epochs 50 --wandb

# Test only (evaluate existing checkpoint)
python train.py --test-only --wandb
```

### Hyperparameter Search

```bash
# Install dependencies first
conda env update -f environment.yml

# Quick test (2 trials, 5 epochs)
python scripts/hp_search.py --phase phase1_coarse --n-trials 2 --epochs 5

# Phase 1: Find good hidden_dim and lr
python scripts/hp_search.py --phase phase1_coarse --n-trials 10 --epochs 25 --wandb

# Phase 2: Use Phase 1 results, tune architecture
python scripts/hp_search.py --phase phase2_architecture --n-trials 12 --epochs 25 \
    --best-params checkpoints/optuna/*/best_params.json --wandb

# Full search (comprehensive)
python scripts/hp_search.py --phase full_with_model --n-trials 50 --epochs 30 --wandb
```

---

## 6. CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | full | Search phase to run |
| `--n-trials` | 20 | Number of Optuna trials |
| `--epochs` | 30 | Epochs per trial |
| `--patience` | 5 | Early stopping patience |
| `--batch-size` | 64 | Batch size |
| `--device` | auto | Device: `cpu`, `cuda`, or `mps` (auto-detects if not set) |
| `--wandb` | False | Enable WandB logging |
| `--wandb-project` | cricket-gnn-optuna | WandB project name |
| `--best-params` | None | JSON from previous phase |
| `--storage` | sqlite:///optuna_studies.db | Optuna storage |

---

## 7. Phased HP Search Strategy

### Recommended Workflow

**Phase 1: Coarse Search**
```bash
python scripts/hp_search.py --phase phase1_coarse --n-trials 10 --epochs 25 --wandb
```
- Goal: Find ballpark hidden_dim and learning rate
- Output: `checkpoints/optuna/phase1_coarse_*/best_params.json`

**Phase 2: Architecture Tuning**
```bash
python scripts/hp_search.py --phase phase2_architecture --n-trials 12 --epochs 25 \
    --best-params checkpoints/optuna/phase1_coarse_*/best_params.json --wandb
```
- Goal: Tune num_layers and num_heads using Phase 1 results
- Output: Updated best_params.json

**Phase 3: Training Dynamics**
```bash
python scripts/hp_search.py --phase phase3_training --n-trials 15 --epochs 30 \
    --best-params checkpoints/optuna/phase2_architecture_*/best_params.json --wandb
```
- Goal: Fine-tune dropout, weight_decay, and learning rate

**Phase 4: Loss Function**
```bash
python scripts/hp_search.py --phase phase4_loss --n-trials 10 --epochs 30 \
    --best-params checkpoints/optuna/phase3_training_*/best_params.json --wandb
```
- Goal: Optimize focal_gamma and class weighting

### Alternative: Full Joint Search
```bash
python scripts/hp_search.py --phase full_with_model --n-trials 50 --epochs 30 --wandb
```
- Searches all parameters simultaneously
- More trials needed but can find unexpected combinations

---

## 8. Understanding Optuna Output

### Console Output
```
[I 2024-01-15 10:23:45,678] Trial 5 finished with value: 0.2341 and parameters: {...}
[I 2024-01-15 10:25:12,345] Trial 6 pruned.
```

- **finished with value:** Trial completed, shows F1 macro score
- **pruned:** Trial stopped early due to poor performance

### Study Summary
After completion:
```
Best trial:
  Value: 0.2567
  Params:
    hidden_dim: 128
    lr: 0.00087
    num_layers: 3
    ...
```

### Files Generated

| File | Purpose |
|------|---------|
| `best_params.json` | Best hyperparameters for next phase |
| `study_results.json` | Complete study history |
| `best_model.pt` | Checkpoint from best trial |
| `visualizations/*.html` | Interactive Plotly charts |

---

## 9. Resuming Studies

Optuna studies are saved to SQLite and can be resumed:

```bash
# Resume a study (will continue from where it left off)
python scripts/hp_search.py --phase phase1_coarse --n-trials 20 --storage sqlite:///optuna_studies.db
```

The study name includes a timestamp, so each run creates a new study unless you specify the same storage and study name.

---

## 10. Troubleshooting

### Common Issues

**MPS (Apple Silicon) Errors**
- If you see `MPSNDArray` or buffer allocation errors on Mac, use CPU instead:
  ```bash
  python scripts/hp_search.py --phase phase1_coarse --n-trials 10 --epochs 25 --wandb --device cpu
  ```
- MPS support in PyTorch is still maturing; CPU is more stable for this workload

**Out of Memory**
- Reduce `--batch-size` (try 32 instead of 64)
- Reduce hidden_dim search range

**All Trials Pruned**
- Increase warmup epochs in pruner
- Check if data loading is correct

**No Improvement Across Trials**
- Expand search ranges
- Try different phase
- Check for data issues

**WandB Not Logging**
- Ensure `--wandb` flag is set
- Check `wandb login` status

### Debugging Tips

```bash
# Run single trial with verbose output
python scripts/hp_search.py --phase phase1_coarse --n-trials 1 --epochs 5 --wandb

# Check study database
sqlite3 optuna_studies.db "SELECT * FROM trials LIMIT 10;"
```
