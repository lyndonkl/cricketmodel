# Metrics and Monitoring Guide

This document defines what metrics to track during training and how to interpret them.

---

## Metric Categories

1. **Primary Metrics** - Decision metrics for model selection
2. **Cricket-Specific Metrics** - Domain-relevant evaluation
3. **Training Diagnostics** - Health monitoring
4. **Calibration Metrics** - Probability quality

---

## Primary Metrics

### Macro F1 Score (Main Decision Metric)

**What it measures**: Average F1 across all classes (treats each class equally).

**Target**: > 0.25

**Why this metric**:
- Handles class imbalance (wickets are rare)
- Single number for model comparison
- Industry standard for multi-class classification

```python
from sklearn.metrics import f1_score

macro_f1 = f1_score(y_true, y_pred, average='macro')
```

### Weighted F1 Score

**What it measures**: F1 weighted by class frequency.

**Target**: > 0.35

**Why this metric**:
- Reflects real-world distribution
- Higher than Macro F1 (majority class easier)

```python
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
```

### Log Loss (Cross-Entropy)

**What it measures**: Quality of probability estimates.

**Target**: < 1.5

**Why this metric**:
- Penalizes confident wrong predictions
- Measures calibration quality
- Used in training loss

```python
from sklearn.metrics import log_loss

logloss = log_loss(y_true, y_prob)
```

---

## Cricket-Specific Metrics

### Wicket Recall

**What it measures**: What fraction of actual wickets did we predict?

**Target**: > 0.20

**Why it matters**: Wickets are game-changing events. Missing them is costly.

```python
from sklearn.metrics import recall_score

# Assuming class 5 = wicket
wicket_recall = recall_score(y_true, y_pred, labels=[5], average='macro')
```

### Boundary Precision

**What it measures**: When we predict a boundary, how often are we right?

**Target**: > 0.30

**Why it matters**: Betting on boundaries requires precision.

```python
from sklearn.metrics import precision_score

# Assuming class 4 = four, class 6 = six
boundary_precision = precision_score(
    y_true, y_pred,
    labels=[4, 6],
    average='macro'
)
```

### Expected Runs Error

**What it measures**: Difference between predicted and actual expected runs.

**Why it matters**: Economic value of predictions.

```python
# Outcome to runs mapping
runs_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 6}  # 5 = wicket

def expected_runs(probs):
    """Calculate expected runs from probability distribution."""
    return sum(probs[i] * runs_map[i] for i in range(7))

# Per-ball error
pred_expected = expected_runs(y_prob)
actual_runs = runs_map[y_true]
error = abs(pred_expected - actual_runs)
```

### Phase-Specific Accuracy

**What it measures**: Accuracy within each game phase.

**Why it matters**: Model may excel in some phases, fail in others.

```python
phases = ['powerplay', 'middle_1', 'middle_2', 'death_1', 'death_2', 'super_over']

for phase in phases:
    mask = (data['phase'] == phase)
    phase_acc = accuracy_score(y_true[mask], y_pred[mask])
    print(f"{phase}: {phase_acc:.3f}")
```

---

## Training Diagnostics

### Loss Curves

**What to plot**: Train loss and Val loss per epoch.

```python
wandb.log({
    "train/loss": train_loss,
    "val/loss": val_loss,
    "epoch": epoch
})
```

**Healthy pattern**:
```
Loss
 ^
 |  \
 |   \_____ train
 |    \____ val
 +-----------------> Epochs
```

**Unhealthy patterns**:
- Val loss increases while train decreases → Overfitting
- Both losses plateau early → Underfitting
- Loss oscillates wildly → LR too high

### Accuracy Curves

**What to plot**: Train and Val accuracy per epoch.

```python
wandb.log({
    "train/accuracy": train_acc,
    "val/accuracy": val_acc,
})
```

### Per-Class F1 Over Time

**What to plot**: F1 for each class across epochs.

```python
for cls in range(7):
    cls_f1 = f1_score(y_true, y_pred, labels=[cls], average='macro')
    wandb.log({f"val/f1_class_{cls}": cls_f1})
```

**Why it matters**: Reveals if model is ignoring minority classes.

### Gradient Norms

**What to plot**: L2 norm of gradients per layer.

```python
def log_gradient_norms(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            wandb.log({f"gradients/{name}": grad_norm})
```

**Warning signs**:
- Gradient norm → 0: Vanishing gradients
- Gradient norm → ∞: Exploding gradients
- Gradient norm oscillates: LR instability

### FiLM Parameter Distributions

**What to plot**: Histogram of gamma and beta values (for PhaseModulated/Full).

```python
# In model forward or after training
for layer_idx, conv_block in enumerate(model.conv_blocks):
    gamma = conv_block.film_gamma.weight.data
    beta = conv_block.film_beta.weight.data

    wandb.log({
        f"film/layer_{layer_idx}_gamma_mean": gamma.mean(),
        f"film/layer_{layer_idx}_gamma_std": gamma.std(),
        f"film/layer_{layer_idx}_beta_mean": beta.mean(),
        f"film/layer_{layer_idx}_beta_std": beta.std(),
    })
```

**Expected**:
- Gamma centered around 1 (identity scaling)
- Beta centered around 0 (no shift)
- Some variance across phases

---

## Calibration Metrics

### Reliability Diagram

**What it shows**: Are probability estimates calibrated?

```python
from sklearn.calibration import calibration_curve

for cls in range(7):
    prob_true, prob_pred = calibration_curve(
        y_true == cls,
        y_prob[:, cls],
        n_bins=10
    )
    plt.plot(prob_pred, prob_true, label=f"Class {cls}")

plt.plot([0, 1], [0, 1], '--', label='Perfect')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.legend()
```

**Interpretation**:
- Points above diagonal → Underconfident
- Points below diagonal → Overconfident

### Expected Calibration Error (ECE)

**What it measures**: Average calibration error across bins.

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate ECE for multi-class predictions."""
    y_pred = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece
```

**Target**: ECE < 0.10

---

## Confusion Matrix Analysis

### What to Log

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Log as image
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Normalized Confusion Matrix')
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close()
```

### What to Look For

| Pattern | Meaning | Action |
|---------|---------|--------|
| Diagonal dominates | Good classification | Keep training |
| One row all same | Predicts majority class | Add class weights |
| Off-diagonal clusters | Confuses similar classes | Feature engineering |
| Wicket row low | Missing rare events | Focal loss / oversample |

---

## WandB Dashboard Setup

### Recommended Panels

1. **Overview**
   - Macro F1 (line)
   - Weighted F1 (line)
   - Log Loss (line)

2. **Training Health**
   - Train/Val Loss (line, overlaid)
   - Train/Val Accuracy (line, overlaid)
   - Learning Rate (line)

3. **Per-Class Performance**
   - Class F1 scores (bar, per class)
   - Confusion Matrix (image)

4. **Cricket Metrics**
   - Wicket Recall (line)
   - Boundary Precision (line)
   - Expected Runs Error (line)

5. **Calibration**
   - Reliability Diagram (image)
   - ECE (line)

6. **Gradients** (if debugging)
   - Gradient norms per layer (line)

### Example Dashboard Config

```yaml
# wandb_panels.yaml
panels:
  - name: "Primary Metrics"
    charts:
      - type: line
        metrics: ["val/macro_f1", "val/weighted_f1"]
      - type: line
        metrics: ["val/log_loss"]

  - name: "Training Health"
    charts:
      - type: line
        metrics: ["train/loss", "val/loss"]
      - type: line
        metrics: ["train/accuracy", "val/accuracy"]

  - name: "Cricket Specific"
    charts:
      - type: line
        metrics: ["val/wicket_recall", "val/boundary_precision"]
```

---

## Quick Reference: Target Metrics

| Metric | Poor | OK | Good | Excellent |
|--------|------|-----|------|-----------|
| Macro F1 | <0.20 | 0.20-0.25 | 0.25-0.30 | >0.30 |
| Weighted F1 | <0.30 | 0.30-0.35 | 0.35-0.40 | >0.40 |
| Log Loss | >2.0 | 1.5-2.0 | 1.2-1.5 | <1.2 |
| Accuracy | <35% | 35-40% | 40-45% | >45% |
| Wicket Recall | <0.10 | 0.10-0.20 | 0.20-0.30 | >0.30 |
| Boundary Precision | <0.20 | 0.20-0.30 | 0.30-0.40 | >0.40 |
| ECE | >0.15 | 0.10-0.15 | 0.05-0.10 | <0.05 |

---

## Logging Code Template

```python
import wandb
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss,
    confusion_matrix, precision_score, recall_score
)

def log_metrics(y_true, y_pred, y_prob, phase_labels, prefix="val"):
    """Log all metrics to WandB."""

    metrics = {
        f"{prefix}/macro_f1": f1_score(y_true, y_pred, average='macro'),
        f"{prefix}/weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        f"{prefix}/accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}/log_loss": log_loss(y_true, y_prob),
        f"{prefix}/wicket_recall": recall_score(y_true, y_pred, labels=[5], average='macro'),
        f"{prefix}/boundary_precision": precision_score(y_true, y_pred, labels=[4, 6], average='macro'),
    }

    # Per-class F1
    for cls in range(7):
        metrics[f"{prefix}/f1_class_{cls}"] = f1_score(
            y_true, y_pred, labels=[cls], average='macro'
        )

    # Phase-specific accuracy
    for phase in ['powerplay', 'middle_1', 'middle_2', 'death_1', 'death_2']:
        mask = (phase_labels == phase)
        if mask.sum() > 0:
            metrics[f"{prefix}/acc_{phase}"] = accuracy_score(
                y_true[mask], y_pred[mask]
            )

    wandb.log(metrics)

    # Confusion matrix (less frequently)
    if wandb.run.step % 10 == 0:
        cm = confusion_matrix(y_true, y_pred)
        fig = plot_confusion_matrix(cm)
        wandb.log({f"{prefix}/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
```
