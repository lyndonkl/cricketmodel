# Manifold Transformations in the Transformer

## The Data Manifold Perspective

### What is a Data Manifold?

Real-world data (text, cricket events, etc.) doesn't fill the entire high-dimensional space uniformly. Instead, it lies on or near a **lower-dimensional manifold** embedded in the high-dimensional representation space.

**Example**: All English sentences form a tiny subset of all possible 512-dimensional vectors. Valid sentences cluster along structured pathways in this space.

### The Learning Goal

The Transformer's job is to **learn transformations of this manifold** that make the prediction task (next word, next ball outcome) easier.

```
Raw Input Manifold → [Transformer Layers] → Task-Optimized Manifold
(tangled, complex)                          (linearly separable)
```

---

## Layer-by-Layer Manifold Evolution

### Initial Embedding: Discrete → Continuous

**Operation**: Token embedding + Positional encoding

**Manifold effect**:
- Discrete tokens are mapped to points in ℝ^512
- Similar tokens (semantically) should be nearby
- Positional encoding adds a "time axis" to the manifold

```
Before: Discrete vocabulary (|V| isolated points)
After:  Continuous manifold in ℝ^(d_model)
        with positional structure added
```

---

### Self-Attention: Dynamic Re-weighting

**Operation**: Each position computes a weighted average of all positions

**Manifold effect**:

1. **Mixing**: Each point moves toward a weighted centroid of all points
2. **Context-dependent**: The weights depend on content (Q·K similarity)
3. **Subspace projections**: Multi-head projects to different subspaces

```
Before attention:
    x₁ •              • x₃
           • x₂

After attention:
    Points move toward contextually relevant neighbors

    x₁' •----→ • x₃'  (if x₁ attends strongly to x₃)
              ↗
           • x₂'
```

**Geometric Interpretation**:

Self-attention performs a **data-dependent convex combination**:
```
x_i' = Σ_j α_ij · V_j

where α_ij ≥ 0 and Σ_j α_ij = 1
```

Each output is a convex combination of value vectors, meaning outputs lie in the **convex hull** of the values (before the output projection).

The attention weights α_ij define a **transport map** on the manifold - moving information from keys/values to queries.

---

### Feed-Forward Network: Local Deformation

**Operation**: Two-layer MLP applied position-wise
```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

**Manifold effect**:

1. **Expansion**: First layer projects 512 → 2048 (4x expansion)
2. **Non-linearity**: ReLU carves the space into linear regions
3. **Compression**: Second layer projects 2048 → 512

```
512-dim → 2048-dim → (ReLU folds space) → 512-dim
  │          │                               │
Manifold → Unfold into   → Fold back but    → Transformed
           higher dim      with new           manifold
                          geometry
```

**Key insight**: The FFN is applied **identically to each position**. It doesn't mix information across positions - that's attention's job. Instead, it **locally deforms** each point on the manifold.

Think of it as: Attention decides **which information to gather**, FFN decides **how to transform** that gathered information.

---

### Residual Connections: Identity + Transformation

**Operation**: `output = x + SubLayer(x)`

**Manifold effect**:

The residual connection ensures the transformation is:
```
x → x + f(x)
```

rather than:
```
x → f(x)
```

**Geometric interpretation**:

1. **Prevents collapse**: Even if f(x) ≈ 0, x is preserved
2. **Incremental refinement**: Each layer makes small adjustments
3. **Gradient highways**: Gradients flow directly through addition

```
Without residuals:
    x₀ → f₁ → f₂ → f₃ → ... → fₙ
    (gradients must traverse all functions)

With residuals:
    x₀ ────────────────────────→ + output
         ↓      ↓      ↓      ↓
        f₁ → + f₂ → + f₃ → + ...
    (direct gradient path through identity)
```

---

### Layer Normalization: Hypersphere Projection

**Operation**: Normalize mean and variance per position
```
LayerNorm(x) = γ · (x - μ) / σ + β
```

**Manifold effect**:

1. **Centering**: Shift mean to origin
2. **Scaling**: Normalize to unit variance
3. **Learnable shift/scale**: γ, β allow recovery

Geometrically, this approximately projects points onto a **hypersphere** (modulo the learnable parameters), preventing representations from drifting to extreme magnitudes.

```
Before LayerNorm: Points can drift far from origin
After LayerNorm:  Points concentrated near unit hypersphere
```

---

## The Full Picture: 6 Layers of Transformation

```
Input tokens
     │
     ▼
[Embedding + Position] ──→ Points on initial manifold
     │
     ▼
┌──────────────────────────────────────────────┐
│ Layer 1                                      │
│   Attention: Mix based on initial features   │
│   FFN: First-level feature extraction        │
│   Result: Basic syntactic patterns emerge    │
└──────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│ Layer 2-3                                    │
│   Attention: Mix based on emerging patterns  │
│   FFN: Compose basic features                │
│   Result: Phrasal structure appears          │
└──────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│ Layer 4-5                                    │
│   Attention: Semantic relationships          │
│   FFN: Higher-level abstractions             │
│   Result: Contextual meanings form           │
└──────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────┐
│ Layer 6                                      │
│   Attention: Task-relevant information       │
│   FFN: Final feature preparation             │
│   Result: Task-ready representations         │
└──────────────────────────────────────────────┘
     │
     ▼
Output (next token prediction)
```

---

## Geometric Deep Learning Principles at Work

### 1. Symmetry and Equivariance

**Principle**: Neural networks should respect the symmetries of the domain.

**In Transformer**:
- Without positions: Fully permutation equivariant
- With positions: Controlled symmetry breaking
- The model learns *which* symmetries to preserve

### 2. Locality and Compositionality

**Principle**: Complex patterns are built from simpler local patterns.

**In Transformer**:
- Attention creates "soft locality" - relevant things become neighbors in feature space
- Layers compose: layer n sees patterns found by layer n-1
- FFN operates locally on each position

### 3. Message Passing

**Principle**: Nodes update by aggregating information from neighbors.

**In Transformer**:
- Attention = message passing on complete graph
- Messages weighted by learned relevance
- Multi-head = multiple message types in parallel

### 4. Hierarchical Representations

**Principle**: Learn representations at multiple scales of abstraction.

**In Transformer**:
- Early layers: local patterns (syntax, adjacent dependencies)
- Later layers: global patterns (semantics, long-range dependencies)
- Each layer refines the manifold for the task

---

## What Learning Does to the Manifold

### Before Training

- Token embeddings are random or pretrained
- Attention weights are essentially random
- The manifold is not optimized for any task

### During Training

Gradient descent adjusts:
1. **W_Q, W_K, W_V, W_O**: How attention weights are computed
2. **W_1, W_2, b_1, b_2**: How FFN transforms locally
3. **Embeddings**: Where tokens sit in the initial manifold

The manifold gradually **unfolds and reorganizes** so that:
- Points needing similar predictions cluster together
- Decision boundaries become simpler (ideally linear)
- Relevant information concentrates; irrelevant information disperses

### After Training

The learned manifold has structure:
- **Clusters**: Similar contexts group together
- **Linear directions**: Often correspond to interpretable features
- **Hierarchy**: Different layers capture different abstraction levels

---

## Implications for Cricket Prediction

### The Cricket Event Manifold

Each ball-by-ball event can be embedded as a point:
- Features: batsman, bowler, runs, wickets, over, match state...
- The manifold of "all possible cricket moments"

### What the Transformer Should Learn

1. **Attention patterns**:
   - Which past deliveries matter for predicting the next?
   - Same bowler's previous balls? Same batsman's recent form?
   - Match situation (chasing, setting)?

2. **Manifold transformations**:
   - Early layers: Encode basic event features
   - Later layers: Encode strategic context
   - Final layer: Prepare for outcome prediction

3. **The learned manifold**:
   - Cluster similar situations (e.g., "pressure situations at death")
   - Separate situations with different likely outcomes
   - Make prediction (runs, wicket probability) linearly extractable

---

## Summary

| Component | Manifold Effect | Geometric Principle |
|-----------|-----------------|---------------------|
| Embedding | Discrete → Continuous | Representation learning |
| Position Encoding | Add temporal structure | Symmetry breaking |
| Self-Attention | Dynamic information mixing | Message passing on complete graph |
| Multi-Head | Multiple parallel views | Ensemble of subspace projections |
| FFN | Local point-wise deformation | Non-linear feature extraction |
| Residual | Incremental refinement | Identity + perturbation |
| Layer Norm | Concentration near hypersphere | Normalization/regularization |
| Stacking Layers | Hierarchical abstraction | Deep composition |
