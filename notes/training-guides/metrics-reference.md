# WandB Metrics Reference

This document provides comprehensive documentation for understanding all metrics logged during Cricket Score-Ahead Regression GNN model training.

---

## 1. Understanding Regression Metrics

The model predicts **score-ahead**: how many more runs the batting team will score from the current ball to the end of the innings. All evaluation metrics measure how close these continuous predictions are to the actual values.

---

### MAE (Mean Absolute Error)

**General Definition:** The average magnitude of prediction errors.

```
MAE = Σ |predicted - actual| / n_samples
```

**In Our Context:** If MAE = 8.5, predictions are off by ~8.5 runs on average. Every run of error counts equally — overpredicting by 20 is twice as bad as overpredicting by 10.

**Why It Helps:** The most intuitive metric. Directly interpretable in cricketing terms: "on average, the model is wrong by X runs." Good for communicating prediction quality to non-technical stakeholders.

**Limitation:** Treats all errors equally. A few wildly wrong predictions (collapse innings, record-breaking partnerships) can inflate MAE even if most predictions are good.

---

### RMSE (Root Mean Squared Error)

**General Definition:** Square root of the average squared errors. Penalizes large errors more than small ones.

```
RMSE = √(Σ (predicted - actual)² / n_samples)
```

**In Our Context:**
- RMSE is always ≥ MAE
- If RMSE ≈ MAE: errors are consistent in size across predictions
- If RMSE >> MAE: some predictions are way off (outlier innings the model misses)

**Why It Helps:** Reveals whether the model has a few catastrophic failures. In cricket, these would be collapse innings (predicting 80 when team collapses for 30) or massive hitting sprees the model doesn't anticipate.

**Relationship to MAE:** The ratio RMSE/MAE indicates error distribution shape. A ratio near 1.0 means uniform errors; a ratio above 1.2 signals heavy-tailed errors worth investigating.

---

### R² (Coefficient of Determination)

**General Definition:** Proportion of variance in the target that the model explains, compared to always predicting the mean.

```
R² = 1 - (Σ (actual - predicted)²) / (Σ (actual - mean(actual))²)
```

**In Our Context:**
- R² = 1.0: Perfect prediction
- R² = 0.5: Model explains 50% of score-ahead variance
- R² = 0.0: No better than always predicting the average score-ahead
- R² < 0.0: Actively worse than the mean (model is harmful)

**Why It Helps:** The primary signal for whether the model is learning meaningful patterns. Unlike MAE/RMSE, R² is scale-independent — you can compare across different datasets or target ranges without worrying about units.

**Cricket Interpretation:** Score-ahead variance comes from match context (overs remaining, wickets in hand, pitch conditions, batting quality). A positive R² means the model captures some of this context. Higher R² means better contextual understanding.

---

### Median AE (Median Absolute Error)

**General Definition:** The middle value when all absolute errors are sorted.

```
Median AE = median(|predicted - actual|)
```

**In Our Context:** If median AE = 6.0 but MAE = 9.0, half of all predictions are within 6 runs, but outlier innings pull the average error up.

**Why It Helps:** Completely robust to outliers. Tells you the "typical" prediction quality. In cricket, outlier innings (collapses, rain interruptions, extraordinary individual performances) are inherently unpredictable — median AE shows how good the model is on normal innings.

**When to Use:** Compare median AE vs MAE to diagnose whether poor MAE is a systemic issue (median AE also high) or an outlier problem (median AE much lower than MAE).

---

### Val Loss (Huber / Smooth L1)

**General Definition:** The actual loss function optimized during training. Huber loss transitions between MSE and MAE behavior based on a delta threshold.

```
Huber(error) = 0.5 × error²           if |error| < delta
             = delta × (|error| - 0.5 × delta)  if |error| >= delta
```

**In Our Context:** With `huber_delta=10.0`:
- Errors < 10 runs: MSE-like behavior (smooth gradients, precise learning)
- Errors ≥ 10 runs: MAE-like behavior (robust to outlier innings)

**Why It Helps:** This is the metric that early stopping, checkpointing, and HP search optimize against. It balances precision on normal predictions with robustness to extreme innings (collapses, massive hitting). Lower is better.

**Tuning Delta:** The `huber_delta` hyperparameter controls the transition:
- Lower delta (e.g., 1.0): More robust to outliers, but slower convergence on small errors
- Higher delta (e.g., 20.0): More emphasis on getting close predictions exactly right, but outliers can destabilize training

---

## 2. Complete Metrics List

### Validation Metrics (Logged Every Epoch)

| Metric | Type | Description |
|--------|------|-------------|
| `val/loss` | Float | Huber (SmoothL1) loss — **optimization target** |
| `metrics/mae` | Float | Mean Absolute Error in runs |
| `metrics/rmse` | Float | Root Mean Squared Error in runs |
| `metrics/r_squared` | Float | Coefficient of determination |

### Training Metrics (Logged Every Epoch)

| Metric | Type | Description |
|--------|------|-------------|
| `train/loss` | Float | Training Huber loss |
| `learning_rate` | Float | Current learning rate |
| `epoch` | Int | Current epoch number |

### Test Metrics (Logged Once After Training)

| Metric | Type | Description |
|--------|------|-------------|
| `test/loss` | Float | Test Huber loss |
| `test/mae` | Float | Test MAE |
| `test/rmse` | Float | Test RMSE |
| `test/r_squared` | Float | Test R² |

---

## 3. Interpreting WandB Charts

### Loss Curves
- **Good:** Train and val loss both decrease, gap stays small
- **Overfitting:** Train loss decreases, val loss increases or plateaus
- **Underfitting:** Both losses plateau high early

### MAE / RMSE Over Time
- **Good:** Steady decrease, eventual plateau
- **RMSE >> MAE divergence:** Model struggles with outlier innings — consider lowering `huber_delta`

### R² Over Time
- **Good:** Starts near 0, increases toward 0.3-0.5+
- **Stuck at 0:** Model isn't learning beyond mean prediction
- **Negative:** Model is worse than the mean — check for bugs or data issues

---

## 4. Realistic Performance Targets

### Why Perfect Prediction is Impossible

Score-ahead depends on **future events that haven't happened yet**:
- A single dropped catch can change the score by 50+ runs
- Weather interruptions, pitch deterioration, tactical declarations
- Individual form fluctuations within a match
- Mental state, fatigue, match pressure

Even expert analysts can't predict exact remaining scores.

### Reasonable Performance Targets

| Metric | Baseline (Mean Predictor) | Acceptable | Good | Excellent |
|--------|---------------------------|------------|------|-----------|
| MAE | ~25-30 runs | 15-20 | 10-15 | < 10 |
| RMSE | ~35-40 runs | 20-28 | 15-20 | < 15 |
| R² | 0.0 | 0.15-0.25 | 0.30-0.50 | 0.50+ |
| Median AE | ~20-25 runs | 12-18 | 8-12 | < 8 |

*Note: Exact baselines depend on dataset composition (T20 only vs mixed formats, innings stage distribution). These targets are rough guides for T20 internationals.*

### What Matters for Simulations

For Monte Carlo match simulations:
1. **Low MAE/RMSE:** Predictions are close to reality on average
2. **Positive R²:** Model captures contextual factors (overs, wickets, pitch)
3. **Low RMSE/MAE ratio:** Consistent predictions without catastrophic failures
4. **Reasonable median AE:** Typical predictions are useful even if edge cases fail
