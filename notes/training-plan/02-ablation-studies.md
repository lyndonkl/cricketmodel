# Ablation Studies Design

This document outlines the ablation experiment design for validating each component of the CricketHeteroGNN architecture.

---

## Purpose of Ablations

Ablation studies answer: **"Does this component actually help?"**

For each architectural choice, we train:
1. Model WITH the component
2. Model WITHOUT the component

The difference in performance tells us the component's value.

---

## Required Ablation Experiments

### Primary Ablations (9 experiments)

| ID | Configuration | Tests | Hypothesis |
|----|---------------|-------|------------|
| A1 | CricketHeteroGNN (Base) | Baseline | - |
| A2 | CricketHeteroGNNWithPooling | Ball pooling | Explicit ball sequence pooling improves over implicit graph aggregation |
| A3 | CricketHeteroGNNHybrid | Matchup MLP | Explicit matchup modeling beats relying on graph structure alone |
| A4 | CricketHeteroGNNPhaseModulated | FiLM | Phase-conditioned message passing captures game dynamics |
| A5 | CricketHeteroGNNInningsConditional | Innings heads | Separate heads capture first/second innings asymmetry |
| A6 | CricketHeteroGNNFull (all on) | Combined | All components together beat individual components |
| A7 | Full - FiLM (`use_film=False`) | FiLM necessity | FiLM adds value in Full model |
| A8 | Full - Innings (`use_innings_heads=False`) | Innings necessity | Innings heads add value in Full model |
| A9 | Full - Hierarchical (`use_hierarchical=False`) | Cold-start | Hierarchical embeddings improve unseen player handling |

---

## Experiment Details

### A1: Baseline (CricketHeteroGNN)

**What it tests**: Core heterogeneous GNN architecture.

```python
from src.model.hetero_gnn import CricketHeteroGNN

model = CricketHeteroGNN(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Expected behavior**: Reasonable baseline, may struggle with class imbalance.

---

### A2: Ball Pooling (CricketHeteroGNNWithPooling)

**What it tests**: Does explicit attention-weighted ball pooling help?

```python
from src.model.hetero_gnn import CricketHeteroGNNWithPooling

model = CricketHeteroGNNWithPooling(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Hypothesis**: Ball pooling should help because:
- Recent balls matter more than distant ones
- Learned attention can focus on key deliveries
- Supplements query-based aggregation

**Compare to**: A1 (Base)

---

### A3: Matchup MLP (CricketHeteroGNNHybrid)

**What it tests**: Does explicit matchup modeling help?

```python
from src.model.hetero_gnn import CricketHeteroGNNHybrid

model = CricketHeteroGNNHybrid(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Hypothesis**: Matchup MLP should help because:
- Batter-bowler interaction dominates outcomes
- Graph structure captures it implicitly, MLP captures it explicitly
- Non-striker gate models partnership dynamics

**Compare to**: A1 (Base)

---

### A4: FiLM Modulation (CricketHeteroGNNPhaseModulated)

**What it tests**: Does phase-conditional message passing help?

```python
from src.model.hetero_gnn import CricketHeteroGNNPhaseModulated

model = CricketHeteroGNNPhaseModulated(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Hypothesis**: FiLM should help because:
- Cricket dynamics change across phases (powerplay vs death overs)
- Same message should be interpreted differently in different phases
- Affine transformation is expressive yet efficient

**Compare to**: A3 (Hybrid)

---

### A5: Innings Heads (CricketHeteroGNNInningsConditional)

**What it tests**: Do separate prediction heads for innings help?

```python
from src.model.hetero_gnn import CricketHeteroGNNInningsConditional

model = CricketHeteroGNNInningsConditional(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

**Hypothesis**: Innings heads should help because:
- First innings: maximize runs
- Second innings: chase target efficiently
- Different strategies → different prediction patterns

**Compare to**: A3 (Hybrid)

---

### A6: Full Model (All Components)

**What it tests**: Do all components work well together?

```python
from src.model.hetero_gnn import CricketHeteroGNNFull

model = CricketHeteroGNNFull(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    use_film=True,
    use_hierarchical=True,
    use_innings_heads=True
)
```

**Compare to**: A1-A5

---

### A7: Full - FiLM

**What it tests**: Is FiLM necessary in the Full model?

```python
model = CricketHeteroGNNFull(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    use_film=False,  # Disabled
    use_hierarchical=True,
    use_innings_heads=True
)
```

**Compare to**: A6 (Full with all)

---

### A8: Full - Innings Heads

**What it tests**: Are innings heads necessary in the Full model?

```python
model = CricketHeteroGNNFull(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    use_film=True,
    use_hierarchical=True,
    use_innings_heads=False  # Disabled
)
```

**Compare to**: A6 (Full with all)

---

### A9: Full - Hierarchical Embeddings

**What it tests**: Do hierarchical player embeddings help cold-start?

```python
model = CricketHeteroGNNFull(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    use_film=True,
    use_hierarchical=False,  # Disabled
    use_innings_heads=True
)
```

**Additional evaluation**: Performance on unseen players vs known players.

**Compare to**: A6 (Full with all)

---

## Experimental Protocol

### 1. Fixed Hyperparameters

Use the same HPs for ALL ablations (from base tuning):

```python
fixed_config = {
    "hidden_dim": 128,  # From HP tuning
    "num_layers": 3,
    "num_heads": 4,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "batch_size": 64,
    "epochs": 100,
    "focal_gamma": 2.0,
    "patience": 15,
}
```

### 2. Multiple Seeds

Run each configuration with 3 random seeds:

```python
seeds = [42, 123, 456]

for ablation in ablations:
    results = []
    for seed in seeds:
        set_seed(seed)
        result = train_and_evaluate(ablation, fixed_config)
        results.append(result)

    mean = np.mean(results)
    std = np.std(results)
    print(f"{ablation}: {mean:.4f} +/- {std:.4f}")
```

### 3. Statistical Significance

Use paired t-test to compare ablations:

```python
from scipy import stats

# Compare A6 (Full) vs A7 (Full - FiLM)
t_stat, p_value = stats.ttest_rel(results_a6, results_a7)

if p_value < 0.05:
    print(f"FiLM significantly improves performance (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

---

## Results Template

### Raw Results

| ID | Config | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|----|--------|---------|----------|----------|------|-----|
| A1 | Base | | | | | |
| A2 | +Pooling | | | | | |
| A3 | +Hybrid | | | | | |
| A4 | +FiLM | | | | | |
| A5 | +Innings | | | | | |
| A6 | Full | | | | | |
| A7 | -FiLM | | | | | |
| A8 | -Innings | | | | | |
| A9 | -Hierarchical | | | | | |

### Statistical Comparisons

| Comparison | Δ Macro F1 | p-value | Significant? |
|------------|------------|---------|--------------|
| A2 vs A1 | | | |
| A3 vs A1 | | | |
| A4 vs A3 | | | |
| A5 vs A3 | | | |
| A6 vs A1 | | | |
| A6 vs A7 | | | |
| A6 vs A8 | | | |
| A6 vs A9 | | | |

---

## Running Ablations

### Command Line

```bash
# Base experiments
python train.py --model base --seed 42 --run_name A1_base_s42
python train.py --model base --seed 123 --run_name A1_base_s123
python train.py --model base --seed 456 --run_name A1_base_s456

# Pooling
python train.py --model pooling --seed 42 --run_name A2_pooling_s42
# ... etc

# Full ablations
python train.py --model full --use_film --use_hierarchical --use_innings_heads \
    --seed 42 --run_name A6_full_s42

python train.py --model full --no_film --use_hierarchical --use_innings_heads \
    --seed 42 --run_name A7_nofilm_s42
```

### Automated Script

```python
# run_ablations.py
import subprocess

ablations = [
    ("base", {}),
    ("pooling", {}),
    ("hybrid", {}),
    ("phase_modulated", {}),
    ("innings_conditional", {}),
    ("full", {"use_film": True, "use_hierarchical": True, "use_innings_heads": True}),
    ("full", {"use_film": False, "use_hierarchical": True, "use_innings_heads": True}),
    ("full", {"use_film": True, "use_hierarchical": True, "use_innings_heads": False}),
    ("full", {"use_film": True, "use_hierarchical": False, "use_innings_heads": True}),
]

seeds = [42, 123, 456]

for model, flags in ablations:
    for seed in seeds:
        cmd = f"python train.py --model {model} --seed {seed}"
        for flag, value in flags.items():
            cmd += f" --{flag}" if value else f" --no_{flag}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)
```

---

## Expected Outcomes

### Likely Winners

Based on domain knowledge:

1. **A3 (Hybrid) > A1 (Base)**: Matchup is crucial in cricket
2. **A4 (FiLM) > A3 (Hybrid)**: T20 phases matter significantly
3. **A6 (Full) > Individual**: Synergistic effects

### Possible Surprises

1. **A2 (Pooling) ≈ A1 (Base)**: Graph may capture ball sequence implicitly
2. **A5 (Innings) ≈ A3 (Hybrid)**: If dataset is mostly T20, innings effect may be small
3. **A9 (-Hierarchical) ≈ A6 (Full)**: If test set has few unseen players

---

## What to Do with Results

### If Component Helps (p < 0.05)

- Keep it in the final model
- Document the improvement magnitude
- Analyze when it helps most (e.g., FiLM helps more in death overs)

### If Component Doesn't Help (p >= 0.05)

- Remove it (simpler model is better)
- Investigate why (insufficient data? wrong implementation?)
- Consider alternative implementations

### If Results Are Inconsistent

- Check for bugs in implementation
- Increase number of seeds
- Look at per-phase or per-class results for insights
