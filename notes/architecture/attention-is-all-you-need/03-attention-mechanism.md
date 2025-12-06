# Attention Mechanism Deep Dive

## Core Intuition

Attention answers the question:
> **"Given what I'm looking for (Query), which elements (Keys) are relevant, and what information (Values) should I extract?"**

It's a **soft dictionary lookup** where:
- Keys don't need to match exactly
- Multiple entries contribute weighted by similarity
- The "relevance" function is learned

---

## Scaled Dot-Product Attention

### The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

### Step-by-Step Breakdown

**Input**:
- Q (Queries): Matrix of shape (n, d_k) - what we're looking for
- K (Keys): Matrix of shape (m, d_k) - what we're searching over
- V (Values): Matrix of shape (m, d_v) - what we'll retrieve

**Step 1: Compute Similarity Scores**
```
Scores = Q · K^T    # Shape: (n, m)
```
Each entry (i, j) = dot product of query i with key j

**Step 2: Scale by √d_k**
```
Scaled = Scores / √d_k
```
Why? Dot products grow with dimension d_k. If q and k have components ~ N(0,1), then q·k ~ N(0, d_k). Large values push softmax into saturation (tiny gradients).

**Step 3: Apply Softmax (row-wise)**
```
Weights = softmax(Scaled)    # Each row sums to 1
```
Converts scores to a probability distribution over keys.

**Step 4: Weighted Sum of Values**
```
Output = Weights · V    # Shape: (n, d_v)
```
Each output is a weighted combination of all values.

---

## Abstraction Ladder for Attention

### L1: Universal Principle
> Information retrieval should be content-based: what you retrieve depends on what you're looking for and what's available.

### L2: Framework
Attention implements a **differentiable associative memory**:
- Store: Key-Value pairs
- Query: Similarity-weighted retrieval

### L3: Method
Three projections (Q, K, V) from input, dot-product similarity, softmax normalization.

### L4: Implementation
```python
def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```

### L5: Concrete Numbers (Base Model)
- d_k = 64
- √d_k ≈ 8
- For sequence length 100: Attention matrix is 100×100 = 10,000 entries

---

## Multi-Head Attention

### Why Multiple Heads?

Single-head attention computes ONE weighted combination. But relationships are multi-faceted:
- Syntactic: "the" relates to its noun
- Semantic: "bank" relates to "money" or "river"
- Positional: adjacent words often relate

**Solution**: Run h parallel attention operations, each learning different relationships.

### The Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

### Dimensions
- Input: d_model = 512
- Per head: d_k = d_v = d_model / h = 64
- Number of heads: h = 8
- Total computation ≈ same as single full-dimensional head

### Geometric Interpretation

Each head projects Q, K, V into a **different subspace**:

```
Head 1: Projects to subspace S₁ → learns syntactic relationships
Head 2: Projects to subspace S₂ → learns semantic relationships
Head 3: Projects to subspace S₃ → learns positional patterns
...
```

The final projection W^O **recombines** these different "views" back into d_model dimensions.

---

## Geometric Deep Learning Perspective

### Attention as a Dynamic Graph

Traditional GNNs operate on fixed graphs:
```
h_i^(l+1) = σ(Σ_j A_ij · W · h_j^(l))
```
where A is a fixed adjacency matrix.

**Attention creates a data-dependent adjacency**:
```
A_ij = softmax_j(q_i · k_j / √d_k)
```

This is equivalent to a **Graph Attention Network (GAT)** on a complete graph where edge weights are computed from node features.

### The Attention Matrix as a Soft Adjacency

For a sequence of length n, attention produces an n×n matrix:

```
        Key positions
      ┌─────────────────┐
      │ 0.1  0.6  0.2  0.1 │  ← Query 1 attends mostly to position 2
Query │ 0.4  0.3  0.2  0.1 │  ← Query 2 attends somewhat uniformly
pos.  │ 0.0  0.1  0.8  0.1 │  ← Query 3 attends mostly to position 3
      │ 0.2  0.2  0.2  0.4 │  ← Query 4 attends mostly to position 4
      └─────────────────┘
```

Properties:
- **Rows sum to 1** (softmax normalization)
- **Learned, not fixed** (depends on Q, K content)
- **Different per head** (multi-head = multiple graphs)

### Equivariance Properties

**Without positional encoding**: Transformer is **permutation equivariant**
- If you permute input positions, outputs permute identically
- Formally: f(Π·x) = Π·f(x) for any permutation matrix Π

**With positional encoding**: This symmetry is **broken**
- Position information distinguishes otherwise identical tokens
- Model can learn position-dependent patterns

This is a **controlled symmetry breaking** - we start with maximal flexibility (complete graph) and inject structure (positions) as needed.

---

## Positional Encoding

### The Problem

Self-attention is **permutation equivariant** - it treats "cat sat mat" the same as "mat cat sat". But word order matters!

### The Solution: Sinusoidal Encodings

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
1. **Unique per position**: Each position gets a distinct vector
2. **Bounded**: Values in [-1, 1]
3. **Relative positions are linear**: PE(pos+k) is a linear function of PE(pos)
4. **Extrapolation**: Works for sequences longer than training data

### Geometric View

The positional encoding maps discrete positions to points on a **high-dimensional torus**:

```
Position 0 → (sin(0), cos(0), sin(0), cos(0), ...)
Position 1 → (sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/d)), ...)
...
```

Different dimensions oscillate at different frequencies, creating a unique "fingerprint" for each position.

---

## Masking in Attention

### Why Mask?

In the decoder, we must prevent attending to **future positions** to maintain the autoregressive property.

### How It Works

Before softmax, set future positions to -∞:

```
Masked_Scores = Scores + Mask

where Mask[i,j] = 0     if j ≤ i (can attend)
                 = -∞   if j > i (cannot attend)
```

After softmax(-∞) = 0, so future positions contribute nothing.

### Visualization

```
Position:    1  2  3  4
Query 1:   [ ✓  ✗  ✗  ✗ ]  ← Can only see position 1
Query 2:   [ ✓  ✓  ✗  ✗ ]  ← Can see positions 1-2
Query 3:   [ ✓  ✓  ✓  ✗ ]  ← Can see positions 1-3
Query 4:   [ ✓  ✓  ✓  ✓ ]  ← Can see all positions
```

---

## Summary: Attention's Role

| Aspect | What Attention Does |
|--------|-------------------|
| Information routing | Dynamically selects what to look at |
| Relationship learning | Discovers which positions relate |
| Graph construction | Builds soft adjacency from content |
| Global receptive field | O(1) path between any positions |
| Parallelization | All positions computed simultaneously |
