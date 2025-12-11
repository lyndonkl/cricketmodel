# Derived Features vs Attention Learning

## The Core Question

> Should we compute features like `pressure_index`, `momentum`, `batsman_setness` explicitly, or can multi-head attention learn these patterns implicitly?

## Answer: Both, for Different Reasons

**Compute explicitly**: Features requiring aggregation, domain knowledge, or non-linear computation

**Let attention learn**: Context-dependent weighting, pattern discovery, temporal relationships

## What Attention CAN Learn

Multi-head attention excels at:

| Pattern | How Attention Learns It | Evidence |
|---------|------------------------|----------|
| **Same-bowler focus** | Head learns to match `bowler_id` across sequence | NLP attention learns coreference |
| **Same-batsman focus** | Head learns to match `batter_id` across sequence | Same mechanism |
| **Recency weighting** | Positional encoding + learned decay | GPT-style models do this |
| **Over-boundary patterns** | Attention to specific ball-in-over positions | Positional patterns |
| **Context-dependent importance** | Different attention weights based on input | Attention's core strength |

### Example: Momentum Learning

Momentum is "how well the team has been scoring recently". Attention CAN approximate this:

```
Current ball attends to last 6-12 balls
→ If those balls have high runs, representation is "positive momentum"
→ If those balls have low runs/wickets, representation is "negative momentum"
```

**But there's a catch**: Attention is linear. It computes weighted sums, not complex functions.

## What Attention CANNOT Learn Efficiently

| Feature | Why Attention Struggles | Issue |
|---------|------------------------|-------|
| **Required Run Rate** | `RRR = (target - score) / (balls_remaining / 6)` | Division is non-linear; attention can't compute ratios |
| **Pressure Index** | Combines 4-5 factors with specific weights | Too many interacting factors |
| **Career Statistics** | Data from other matches | Not in the sequence at all |
| **Run Rate** | `runs / overs` | Division operation |
| **Setness** | Non-linear function of balls faced | Non-linear transformation |
| **Phase Boundaries** | "Powerplay ends at over 6" | Domain knowledge, not pattern |

### Example: Required Run Rate

```python
# What we need:
rrr = (target - score) / (balls_remaining / 6)

# What attention can do:
# Weighted sum of: target, score, balls_remaining
# = a*target + b*score + c*balls_remaining

# These are NOT equivalent. Attention cannot learn division.
```

## The Hybrid Strategy

### Principle: Compute What Attention Can't, Let Attention Weight What It Can

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE STRATEGY                                 │
│                                                                      │
│  COMPUTE EXPLICITLY                LET ATTENTION LEARN               │
│  ─────────────────────            ────────────────────               │
│                                                                      │
│  ┌─────────────────────┐          ┌─────────────────────┐           │
│  │ Required run rate   │          │ Which balls matter   │           │
│  │ Pressure index      │ ──────►  │ for prediction      │           │
│  │ Batsman setness     │  (as     │                     │           │
│  │ Consecutive dots    │  node    │ Context-dependent   │           │
│  │ Phase indicators    │  features)│ weighting of        │           │
│  │ Run rate           │          │ computed features   │           │
│  │ Partnership stats   │          │                     │           │
│  └─────────────────────┘          │ Pattern discovery   │           │
│                                   │ (same-bowler, etc)  │           │
│                                   └─────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### Feature-by-Feature Decision Matrix

| Feature | Compute? | Let Attention Weight? | Rationale |
|---------|----------|----------------------|-----------|
| **Pressure Index** | Yes | Yes | Requires aggregation; attention weights its importance |
| **Required Run Rate** | Yes | Yes | Division needed; attention learns when RRR matters |
| **Batsman Setness** | Yes | Yes | Non-linear function; attention learns when setness matters |
| **Momentum Score** | Yes | Yes | Aggregation across balls; attention refines importance |
| **Consecutive Dots** | Yes | Yes | Counting; attention learns pressure effect |
| **Same-Bowler Pattern** | No | Yes | Attention head specializes for this |
| **Same-Batsman Form** | No | Yes | Attention head specializes for this |
| **Recency Effect** | No | Yes | Positional attention handles this |
| **Boundary Clusters** | Partially | Yes | Count explicitly, attention finds patterns |
| **Phase Effects** | Yes (flags) | Yes | Boundaries are domain knowledge; effects are learned |

## Why This Hybrid Works

### 1. Data Efficiency

Without explicit features:
- Model must DISCOVER that "pressure builds when dot balls accumulate"
- Requires many examples to learn this from scratch
- ~100,000+ balls needed

With explicit features:
- Model KNOWS pressure index exists as input
- Only needs to learn WHEN and HOW MUCH it matters
- ~10,000 balls sufficient

### 2. Guaranteed Signal

Without explicit features:
- Model MIGHT learn pressure dynamics
- Or might find spurious correlations
- Depends on architecture, data, luck

With explicit features:
- Model WILL see pressure index
- Cannot ignore this domain knowledge
- Reduces variance in learning

### 3. Interpretability

Without explicit features:
- Attention weights are to raw features
- "Model attended to runs, balls, wickets..."
- Must infer pressure from raw features

With explicit features:
- Attention weights to pressure_index directly
- "Model attended to pressure index (35%)"
- Direct interpretability

### 4. Attention for Refinement

Explicit features are **baseline predictions**.
Attention learns **context-dependent adjustments**.

```
Example:
- Pressure index = 0.7 (high)
- But attention learns: "when batsman is set (high setness),
  pressure matters less"
- Effective pressure weight: 0.7 * 0.6 = 0.42
```

## Implementation: Separate Nodes for Derived Features

In our [graph structure](./02-graph-structure.md), derived features have their own nodes:

| Derived Feature | Node | Why Separate |
|-----------------|------|--------------|
| Pressure Index | Pressure Index Node | Can observe attention to pressure directly |
| Momentum | Batting/Bowling Momentum Nodes | Separate batting and bowling momentum |
| Setness | Part of Batsman State Node | Bundled with batsman state |
| RRR | Chase State Node | Chase-specific features together |

This means attention weights directly show:
- "Model attended 40% to Pressure Index Node"
- Not: "Model attended to some combination of features that approximately equals pressure"

## Ablation Study Design

To validate this hybrid approach, run ablations:

| Configuration | Description | Expected Result |
|---------------|-------------|-----------------|
| **Full Model** | Derived features + attention | Baseline |
| **No Derived** | Raw features only | Worse; model must learn pressure from scratch |
| **No Attention** | Derived features, MLP only | Worse; loses context-dependent weighting |
| **Fixed Weights** | Derived features with fixed importance | Worse; loses situational adaptation |

### Expected Ablation Results

```
Full Model:         Accuracy = 0.42, Calibration = 0.92
No Derived:         Accuracy = 0.35  ← Can't learn pressure efficiently
No Attention:       Accuracy = 0.38  ← Misses context-dependent weighting
Fixed Weights:      Accuracy = 0.40  ← Can't adapt to situation
```

## For LLM Insight Generation

With the hybrid approach, LLM can generate insights like:

**With derived features**:
> "The model attended heavily to the Pressure Index node (35%), which currently shows high pressure (0.78) due to required run rate gap. Combined with low attention to Batting Momentum (-0.2), the model predicts conservative play."

**Without derived features (hypothetical)**:
> "The model attended to runs (25%), wickets (20%), balls (15%), target (10%)... [must manually compute what this means for pressure]"

The hybrid approach makes attention patterns **directly interpretable**.

## Summary

| Question | Answer |
|----------|--------|
| Should we compute derived features? | Yes - for anything requiring aggregation, division, or domain knowledge |
| Should we let attention learn? | Yes - for context-dependent weighting and pattern discovery |
| Are they alternatives? | No - they're complementary |
| Which is more important? | Both; derived features provide signal, attention provides adaptation |

## See Also

- [Full derived features catalog](../data-analysis/04-derived-features-catalog.md)
- [Graph structure with feature nodes](./02-graph-structure.md)
- [Systems analysis of feedback loops](../data-analysis/02-systems-thinking-leverage.md)
