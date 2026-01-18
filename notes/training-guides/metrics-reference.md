# WandB Metrics Reference

This document provides comprehensive documentation for understanding all metrics logged during Cricket Ball Prediction GNN model training.

---

## 1. Understanding Classification Metrics

### Accuracy

**General Definition:** Proportion of correct predictions out of all predictions.

```
Accuracy = Correct Predictions / Total Predictions
```

**In Our Context:** With 7 classes (Dot, Single, Two, Three, Four, Six, Wicket), random guessing gives ~14.3% accuracy. The class imbalance (Dots=39%, Singles=34%) means a model predicting only "Dot" would get 39% accuracy without learning anything useful.

**Why It's Limited:** Accuracy alone is misleading for imbalanced datasets. A model ignoring rare events (wickets, sixes) can still achieve decent accuracy.

---

### Precision

**General Definition:** Of all predictions for class X, how many were actually X?

```
Precision = True Positives / (True Positives + False Positives)
```

**In Our Context:**
- `val/precision_wicket`: If the model predicts "wicket" 100 times and 30 are actual wickets → 30% precision
- High precision = few false alarms
- Low precision = many false positives (crying wolf)

**Cricket Interpretation:** High boundary precision means when the model says "Four" or "Six", it's usually right. Important for risk assessment.

---

### Recall (Sensitivity)

**General Definition:** Of all actual class X samples, how many did we correctly identify?

```
Recall = True Positives / (True Positives + False Negatives)
```

**In Our Context:**
- `val/recall_wicket`: If there are 100 actual wickets and we correctly predict 25 → 25% recall
- High recall = catching most events of that type
- Low recall = missing many actual events

**Cricket Interpretation:** Wicket recall is critical - if we miss 80% of wickets, the model can't help with bowling strategy or predicting collapses.

---

### F1 Score

**General Definition:** Harmonic mean of precision and recall, balancing both.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why Harmonic Mean?** Penalizes extreme imbalances. If precision=100% but recall=1%, F1=1.98% (not 50.5% like arithmetic mean).

**Variants:**

| Metric | Formula | Use Case |
|--------|---------|----------|
| `f1_macro` | Average F1 across all classes (unweighted) | Treats rare classes equally - **primary metric** |
| `f1_weighted` | Average F1 weighted by class frequency | Favors common classes |

**In Our Context:** `f1_macro` is our optimization target because we care about predicting wickets (5.7%) as much as dots (39%).

---

### Log Loss (Cross-Entropy)

**General Definition:** Measures confidence of probability predictions, not just correctness.

```
Log Loss = -Σ log(predicted_probability_of_true_class) / n_samples
```

**In Our Context:**
- Penalizes confident wrong predictions heavily
- A model predicting 90% confidence on wrong class loses more than 51% confidence
- Perfect predictions → log_loss approaches 0
- Random guessing (1/7 = 14.3%) → log_loss ≈ 1.95

**Why Track It:** Ensures model probabilities are meaningful, not just argmax predictions.

---

### Top-K Accuracy

**General Definition:** Was the true class in the top K predictions?

**In Our Context:**
- `top_2_accuracy`: True label in top-2 predictions
- `top_3_accuracy`: True label in top-3 predictions

**Cricket Interpretation:** Even if the model doesn't nail the exact outcome, predicting "could be a dot or single" (top-2) is useful for strategic decisions.

---

### Expected Calibration Error (ECE)

**General Definition:** Measures if predicted probabilities match actual frequencies.

```
ECE = Σ (samples_in_bin / total) × |accuracy_in_bin - confidence_in_bin|
```

**Calculation:**
1. Group predictions into 10 bins by confidence (0-10%, 10-20%, etc.)
2. For each bin, compare average confidence vs actual accuracy
3. Weighted average of gaps

**In Our Context:**
- ECE < 0.05: Excellent calibration
- ECE < 0.10: Good calibration
- ECE > 0.15: Probabilities are unreliable

**Why It Matters:** For simulations, we need probability outputs to be trustworthy. If model says 20% chance of wicket, it should actually happen ~20% of the time.

---

## 2. Cricket-Specific Metrics

### Wicket Recall

**Formula:** `TP_wicket / (TP_wicket + FN_wicket)`

**Why Tracked Separately:** Wickets are rare (5.7%) but game-changing. Missing them means the model can't capture crucial match moments.

**Target:** > 25% is reasonable given inherent unpredictability.

---

### Boundary Precision

**Formula:** `(TP_four + TP_six) / (predictions_four + predictions_six)`

**Why Tracked Separately:** Boundaries (Four + Six) represent high-value outcomes. False boundary predictions would overestimate scoring in simulations.

**Target:** > 35% indicates the model has learned boundary-prone situations.

---

### Expected Runs Error

**Formula:** Mean Absolute Error between predicted and actual expected runs.

```
Expected_Runs = Σ (probability_class × runs_for_class)
# Where runs = [0, 1, 2, 3, 4, 6, 0] for [Dot, Single, Two, Three, Four, Six, Wicket]
```

**Why Tracked:** Directly measures economic value of predictions for simulation purposes.

---

## 3. Complete Metrics List

### Validation Metrics (Logged Every Epoch)

| Metric | Type | Description |
|--------|------|-------------|
| `val/loss` | Float | Focal/cross-entropy loss |
| `val/accuracy` | Float | Overall accuracy |
| `val/f1_macro` | Float | **Primary metric** - unweighted F1 |
| `val/f1_weighted` | Float | Class-weighted F1 |
| `val/precision_macro` | Float | Unweighted precision |
| `val/recall_macro` | Float | Unweighted recall |
| `val/log_loss` | Float | Cross-entropy from probabilities |
| `val/top_2_accuracy` | Float | Top-2 accuracy |
| `val/top_3_accuracy` | Float | Top-3 accuracy |
| `val/ece` | Float | Expected Calibration Error |
| `val/wicket_recall` | Float | Wicket class recall |
| `val/boundary_precision` | Float | Four+Six precision |
| `val/expected_runs_error` | Float | MAE for expected runs |
| `val/f1_{class}` | Float | Per-class F1 (7 classes) |
| `val/precision_{class}` | Float | Per-class precision (7 classes) |
| `val/recall_{class}` | Float | Per-class recall (7 classes) |
| `val/confusion_matrix` | Image | 7x7 heatmap (every 10 epochs) |

### Training Metrics (Logged Every Epoch)

| Metric | Type | Description |
|--------|------|-------------|
| `train/loss` | Float | Training loss |
| `train/accuracy` | Float | Training accuracy |
| `learning_rate` | Float | Current learning rate |
| `epoch` | Int | Current epoch number |

### Test Metrics (Logged Once After Training)

Same as validation metrics but prefixed with `test/`.

---

## 4. Interpreting WandB Charts

### Loss Curves
- **Good:** Train and val loss both decrease, gap stays small
- **Overfitting:** Train loss decreases, val loss increases or plateaus
- **Underfitting:** Both losses plateau high early

### F1 Macro Over Time
- **Good:** Steady increase, eventual plateau
- **Bad:** Stuck near random baseline (~0.10)

### Per-Class F1 Breakdown
- Check if rare classes (Three, Wicket, Six) are being learned
- If `f1_three = 0`, model ignores this class

### Confusion Matrix
- **Diagonal dominance:** Good predictions
- **Row with no diagonal:** Class being ignored
- **Off-diagonal clusters:** Systematic confusion between classes

---

## 5. Realistic Performance Targets

### Why Perfect Prediction is Impossible

Cricket ball outcomes are **inherently stochastic**:
- Same bowler, same batsman, same situation → different outcomes
- Millisecond timing differences, wind, pitch variation
- Mental state, fatigue, match pressure

Even expert commentators can't predict individual balls.

### Reasonable Performance Targets

| Metric | Random Baseline | Acceptable | Good | Excellent |
|--------|-----------------|------------|------|-----------|
| Accuracy | 14.3% | 25-28% | 30-35% | 38%+ |
| F1 Macro | 0.10 | 0.18-0.22 | 0.25-0.30 | 0.32+ |
| Wicket Recall | 0% | 15-20% | 25-35% | 40%+ |
| Boundary Precision | 14% | 25-30% | 35-45% | 50%+ |
| ECE | - | < 0.15 | < 0.10 | < 0.05 |
| Log Loss | 1.95 | 1.5-1.7 | 1.3-1.5 | < 1.3 |

### What "Good Enough" Means for Simulations

For Monte Carlo simulations, you need:
1. **Calibrated probabilities** (ECE < 0.10): So 20% wicket prediction = 20% actual rate
2. **Reasonable class coverage**: All classes predicted, not just majorities
3. **Contextual sensitivity**: Different predictions for powerplay vs death overs

**Accuracy matters less than probability quality** for simulations.
