# Applying Transformers to Cricket Ball-by-Ball Prediction

## The Prediction Task

**Goal**: Given the history of a cricket match up to delivery n, predict the outcome of delivery n+1.

**Outcomes to predict** (could be multi-task):
- Runs scored (0, 1, 2, 3, 4, 6)
- Extras (wide, no-ball, bye, leg-bye)
- Wicket (and type: bowled, caught, LBW, etc.)
- Dot ball probability
- Boundary probability

---

## Why Transformers Fit This Problem

### 1. Long-Range Dependencies

Cricket has inherent long-range patterns:

| Dependency | Distance | Example |
|------------|----------|---------|
| Over-level | ~6 balls | Bowler's plan for the over |
| Partnership | 10-100+ balls | Established pair vs. new batsman |
| Match situation | 100+ balls | Chasing target, required rate |
| Batsman form | Across matches | Historical patterns |

Transformers handle these with **O(1) path length** vs. RNNs' O(n).

### 2. Dynamic Relevance

Not all past events are equally relevant:
- The previous ball matters a lot
- The first ball of the same bowler's previous over matters
- A wicket 30 balls ago might be very relevant (new batsman)

**Attention learns which events to focus on** - it doesn't assume recency is most important.

### 3. Multi-Head Attention = Multiple Views

Different heads can learn different relationships:
- Head 1: Same-bowler patterns
- Head 2: Same-batsman patterns
- Head 3: Match situation context
- Head 4: Recent momentum

---

## Mapping Cricket to Transformer Architecture

### Input Representation

Each delivery becomes a token with features:

```python
delivery_features = {
    # Event features
    'runs': 0-6,
    'extras': one-hot encoding,
    'wicket': binary + type,

    # Context features
    'over': 1-20 (T20) or 1-50 (ODI),
    'ball_in_over': 1-6,
    'innings': 1 or 2,

    # Actors
    'batsman_id': embedding,
    'bowler_id': embedding,
    'non_striker_id': embedding,

    # Match state
    'total_runs': int,
    'wickets_fallen': 0-10,
    'required_runs': int (2nd innings),
    'balls_remaining': int,

    # Derived
    'run_rate': float,
    'required_rate': float,
    'partnership_runs': int,
    'partnership_balls': int,
}
```

### Embedding Strategy

```
Token = Concat(
    Embed(batsman),           # d_batsman
    Embed(bowler),            # d_bowler
    Embed(outcome),           # d_outcome
    Linear(numeric_features), # d_numeric
    Positional_Encoding       # d_model
)
```

### Architecture Choices

**Encoder-only** (like BERT) vs **Decoder-only** (like GPT)?

For **autoregressive prediction** (predict next ball given all previous), a **decoder-only** architecture with causal masking is natural:

```
[ball_1, ball_2, ..., ball_n] → [pred_2, pred_3, ..., pred_n+1]
```

Each position predicts the next delivery, using only past context.

---

## Attention Patterns We'd Expect to Learn

### 1. Recent Ball Attention
```
Ball n should attend strongly to balls n-1, n-2, n-3
(momentum, immediate context)
```

### 2. Same-Bowler Attention
```
Ball n (bowler X) should attend to all previous balls by bowler X
(bowler's patterns, setups)
```

### 3. Same-Batsman Attention
```
Ball n should attend to all previous balls faced by current batsman
(batsman's form, weaknesses)
```

### 4. Over Boundary Attention
```
Ball 6 of over should attend to balls 1-5 of same over
(completing the over, bowler's strategy)
```

### 5. Phase Attention
```
Death overs (16-20) should attend more to other death overs
(different patterns than powerplay)
```

---

## Geometric View: The Cricket Manifold

### Input Space

Each delivery is a point in high-dimensional space. The manifold of "all T20 deliveries" has structure:

```
Clusters:
- Powerplay deliveries (1-6)
- Middle overs (7-15)
- Death overs (16-20)
- Spin vs pace
- Left-hand vs right-hand batsman
- High pressure vs low pressure
```

### What Learning Should Do

**Before training**: Deliveries embedded based on raw features

**After training**: The manifold should organize by **outcome predictability**:

```
Region A: "Almost certainly a dot ball"
  - Good bowler, defensive batsman, early innings

Region B: "High boundary probability"
  - Death overs, set batsman, pace bowler

Region C: "Wicket likely"
  - New batsman, quality bowler, pressure situation
```

The final layer should make **outcome prediction linear** - a simple classifier on top of the transformed manifold.

---

## Positional Encoding for Cricket

Standard sinusoidal encoding gives position 1, 2, 3, ... But cricket has **multiple notions of position**:

### Ball Position (Absolute)
```
PE_ball(pos) = standard sinusoidal
```

### Ball-in-Over Position
```
PE_over_ball(pos) = embed(pos mod 6)
# Ball 1-6 of over has special meaning
```

### Over Number
```
PE_over(pos) = embed(pos // 6)
# Which over we're in
```

### Combined Positional Encoding
```
PE_total = PE_ball + PE_over_ball + PE_over + PE_phase
```

This **injects cricket-specific structure** while letting attention learn the rest.

---

## Training Considerations

### Data Structure

From the T20 JSON data, each match provides a sequence of ~240 deliveries (max 120 per innings × 2 innings).

```python
# Training example structure
{
    'match_id': str,
    'innings': int,
    'deliveries': [
        {'over': 0, 'ball': 1, 'batsman': 'X', 'bowler': 'Y', 'runs': 1, ...},
        {'over': 0, 'ball': 2, ...},
        ...
    ]
}
```

### Masking Strategy

- **Causal mask**: Predict ball n+1 from balls 1...n
- **Padding mask**: Handle variable-length innings (all outs, rain)

### Loss Function

Multi-task prediction:
```
Loss = α·CE(runs) + β·BCE(wicket) + γ·CE(extras) + ...
```

Or hierarchical:
```
Loss = CE(outcome_category) + conditional_losses
```

---

## Key Differences from NLP

| Aspect | NLP (Translation) | Cricket Prediction |
|--------|-------------------|-------------------|
| Vocabulary | ~30K-50K tokens | ~100 outcome types + continuous features |
| Sequence length | ~50-200 tokens | ~240 deliveries (T20) |
| Token type | Discrete words | Mixed discrete/continuous |
| Actors | N/A | Batsman, bowler identities matter |
| Structure | Sentences, paragraphs | Overs, innings, phases |
| Goal | Generate sequence | Predict next + probabilities |

---

## Next Steps for Implementation

### 1. Data Exploration
- Understand JSON format structure
- Compute statistics on outcomes
- Visualize patterns in data

### 2. Feature Engineering
- Design input representation
- Create batsman/bowler embeddings
- Define positional encoding scheme

### 3. Architecture Design
- Choose model size (layers, dimensions, heads)
- Define output head(s)
- Implement custom positional encoding

### 4. Training Pipeline
- Data loading and batching
- Train/validation/test split (by match, not by delivery)
- Evaluation metrics (accuracy, log-loss, calibration)

### 5. Analysis
- Visualize attention patterns
- Interpret what model learns
- Compare to baselines (logistic regression, LSTM)

---

## Summary

The Transformer is well-suited for cricket ball-by-ball prediction because:

1. **Long-range dependencies**: Match context matters across hundreds of balls
2. **Dynamic attention**: Model learns which past events are relevant
3. **Parallel computation**: Efficient training on long sequences
4. **Multi-head attention**: Captures multiple types of relationships
5. **Proven architecture**: Foundation of modern sequence models

The key challenges will be:
- Representing the rich, mixed-type cricket features
- Encoding cricket-specific structure (overs, innings, phases)
- Ensuring the model learns cricket-relevant patterns, not just recency
