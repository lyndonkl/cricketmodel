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

### Understanding the Dimensions: What are n, m, d_k, d_v?

Before diving into the steps, let's clarify what these dimensions mean:

| Symbol | Meaning | Typical Value | What it represents |
|--------|---------|---------------|-------------------|
| **n** | Number of queries | Sequence length | How many positions are "asking questions" |
| **m** | Number of keys/values | Sequence length | How many positions can be "looked at" |
| **d_k** | Dimension of each query/key | 64 | Size of the vector used for matching |
| **d_v** | Dimension of each value | 64 | Size of the vector that gets retrieved |

#### Concrete Example: Self-Attention on a Sentence

Sentence: "The cat sat" (3 words)

```
n = 3  (3 queries, one per word position)
m = 3  (3 keys/values, one per word position)
d_k = 64 (each query/key is a 64-dimensional vector)
d_v = 64 (each value is a 64-dimensional vector)
```

**Q shape**: (3, 64) = 3 query vectors, each with 64 numbers
**K shape**: (3, 64) = 3 key vectors, each with 64 numbers
**V shape**: (3, 64) = 3 value vectors, each with 64 numbers

#### When n ≠ m: Cross-Attention

In encoder-decoder attention (cross-attention):
- **n** = decoder sequence length (e.g., 5 words generated so far)
- **m** = encoder sequence length (e.g., 10 words in input)

The decoder (n=5 positions) attends to the encoder (m=10 positions).

#### NOT the Vocabulary Size!

Important: n and m are **sequence lengths**, NOT vocabulary size.
- Vocabulary size might be 30,000 tokens
- But a single sentence might only have n=10 tokens
- Q, K, V matrices are about the **positions in the current sequence**

---

### Step-by-Step Breakdown with Concrete Numbers

Let's work through a complete example with actual numbers.

**Setup**: Sentence "The cat sat" with d_k = 4 (small for illustration)

```
Word positions: [0: "The", 1: "cat", 2: "sat"]

After projection, we have:
Q = [[1, 0, 1, 0],    # Query for "The"
     [0, 1, 0, 1],    # Query for "cat"
     [1, 1, 0, 0]]    # Query for "sat"

K = [[1, 0, 0, 1],    # Key for "The"
     [0, 1, 1, 0],    # Key for "cat"
     [1, 0, 1, 0]]    # Key for "sat"

V = [[0.1, 0.2, 0.3, 0.4],   # Value for "The"
     [0.5, 0.6, 0.7, 0.8],   # Value for "cat"
     [0.9, 1.0, 1.1, 1.2]]   # Value for "sat"
```

---

**Step 1: Compute Similarity Scores**

```
Scores = Q · K^T    # Shape: (3, 3)
```

Each entry Scores[i,j] = dot product of Query i with Key j

```
Scores[0,0] = Q[0] · K[0] = [1,0,1,0] · [1,0,0,1] = 1*1 + 0*0 + 1*0 + 0*1 = 1
Scores[0,1] = Q[0] · K[1] = [1,0,1,0] · [0,1,1,0] = 1*0 + 0*1 + 1*1 + 0*0 = 1
Scores[0,2] = Q[0] · K[2] = [1,0,1,0] · [1,0,1,0] = 1*1 + 0*0 + 1*1 + 0*0 = 2

Scores = [[1, 1, 2],     # "The" has highest similarity with "sat"
          [1, 2, 0],     # "cat" has highest similarity with "cat"
          [1, 1, 1]]     # "sat" has equal similarity with all
```

**What this means**: "The" (Query 0) matches best with "sat" (Key 2), getting score 2.

---

**Step 2: Scale by √d_k**

```
d_k = 4
√d_k = 2

Scaled = Scores / 2 = [[0.5, 0.5, 1.0],
                       [0.5, 1.0, 0.0],
                       [0.5, 0.5, 0.5]]
```

#### Why Scale? The Variance Problem Explained

**The Problem**: Dot products get bigger as vectors get longer (more dimensions).

**Intuition with an example**:

Imagine two vectors with random components from N(0,1) (mean 0, variance 1):
```
d_k = 2:  q = [0.5, -0.3],  k = [0.8, 0.2]
          q·k = 0.5*0.8 + (-0.3)*0.2 = 0.4 - 0.06 = 0.34

d_k = 100: q = [0.5, -0.3, 0.1, ..., 0.7]  (100 numbers)
           k = [0.8, 0.2, -0.5, ..., 0.3]  (100 numbers)
           q·k = sum of 100 products ≈ much larger magnitude!
```

**The Math**:
- Each component q_i and k_i has variance 1
- Each product q_i * k_i has variance 1 (product of independent unit-variance variables)
- Sum of d_k such products has variance d_k
- So q·k ~ N(0, d_k) — the variance grows with dimension!

**Why this is bad for softmax**:

```
softmax([1, 2, 3]) = [0.09, 0.24, 0.67]   # Reasonable distribution

softmax([10, 20, 30]) = [0.0000, 0.0000, 1.0000]  # Saturated!
```

When values are large, softmax becomes nearly one-hot (one value ≈ 1, others ≈ 0).

**The gradient problem**:
- softmax'(x) ≈ 0 when softmax is saturated
- No gradient = no learning!

**The fix**: Divide by √d_k to keep variance ≈ 1 regardless of dimension.

```
If q·k ~ N(0, d_k), then (q·k)/√d_k ~ N(0, 1)
```

---

**Step 3: Apply Softmax (row-wise)**

```
Weights = softmax(Scaled, dim=-1)  # Each ROW sums to 1
```

For row 0 (Query "The"):
```
softmax([0.5, 0.5, 1.0]) = [e^0.5, e^0.5, e^1.0] / (e^0.5 + e^0.5 + e^1.0)
                        = [1.65, 1.65, 2.72] / 6.02
                        = [0.27, 0.27, 0.45]
```

Full Weights matrix:
```
Weights = [[0.27, 0.27, 0.45],   # "The" attends: 27% to "The", 27% to "cat", 45% to "sat"
           [0.24, 0.65, 0.11],   # "cat" attends: 24% to "The", 65% to "cat", 11% to "sat"
           [0.33, 0.33, 0.33]]   # "sat" attends: equally to all
```

**What "attends" means**: The weight is how much information to take from each position.
- "cat" attends 65% to itself (position 1) — it takes most of its output from its own value
- "The" attends 45% to "sat" — it takes almost half its information from position 2

---

**Step 4: Weighted Sum of Values**

```
Output = Weights · V    # Shape: (3, 4)
```

**What this computes**: For each query position, a weighted average of ALL value vectors.

For position 0 ("The"):
```
Output[0] = 0.27 * V[0] + 0.27 * V[1] + 0.45 * V[2]
          = 0.27 * [0.1, 0.2, 0.3, 0.4]
          + 0.27 * [0.5, 0.6, 0.7, 0.8]
          + 0.45 * [0.9, 1.0, 1.1, 1.2]
          = [0.027, 0.054, 0.081, 0.108]
          + [0.135, 0.162, 0.189, 0.216]
          + [0.405, 0.450, 0.495, 0.540]
          = [0.567, 0.666, 0.765, 0.864]
```

**The output for "The" is a blend of all three value vectors**, weighted by attention.

Since "The" attended most to "sat" (45%), its output is pulled toward V["sat"].

---

### When Does This Happen? Training vs Inference

**This happens in BOTH training and inference!**

**During training**:
- Forward pass: Compute attention for all positions
- Loss: Compare predicted tokens to actual tokens
- Backward pass: Gradients flow through attention weights
- The W_Q, W_K, W_V matrices get updated

**During inference**:
- Same computation, but no gradient updates
- For autoregressive generation: compute attention, predict next token, add to sequence, repeat

**The attention matrix example** (Query 1 attends to position 2, etc.) shows what happens at **one moment** — it could be during training on a batch, or during inference generating the next token.

---

## Abstraction Ladder for Attention

### L1: Universal Principle
> Information retrieval should be content-based: what you retrieve depends on what you're looking for and what's available.

### L2: Framework
Attention implements a **differentiable associative memory**.

#### What is Associative Memory?

**Traditional dictionary/hash table**:
```python
memory = {"cat": "furry animal", "dog": "loyal pet"}
query = "cat"
result = memory["cat"]  # Returns "furry animal"

# But what if query = "feline"?
result = memory["feline"]  # KeyError! No exact match.
```

**Associative memory** (content-addressable):
```python
# Query by similarity, not exact match
query = "feline"
# Find keys similar to "feline" → "cat" is similar
# Return weighted blend of values based on similarity
```

**Attention as associative memory**:
- **Keys** = addresses/labels for stored information
- **Values** = the actual stored information
- **Query** = what we're looking for
- **Output** = weighted blend of values based on query-key similarity

#### What does "differentiable" mean?

**Differentiable** = we can compute gradients through it.

Traditional dictionary lookup is NOT differentiable:
```python
# This has no gradient - it's a discrete operation
if key == "cat":
    return value_cat
else:
    return None
```

Attention IS differentiable:
```python
# Soft lookup - every value contributes a little
output = 0.7 * value_cat + 0.2 * value_dog + 0.1 * value_bird
# Gradients can flow to all values and to the weights!
```

**Why this matters**: Gradient descent can optimize the keys and values to store useful information, and optimize queries to retrieve relevant information.

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

### L5: Concrete Numbers (Base Model) - Explained

```
d_model = 512    # Each token is represented as a 512-dim vector
d_k = 64         # Each query/key is 64-dim (512 / 8 heads)
d_v = 64         # Each value is 64-dim
h = 8            # 8 parallel attention heads
```

**For sequence length 100:**
```
Q shape: (100, 64)   # 100 queries, each 64-dim
K shape: (100, 64)   # 100 keys, each 64-dim
V shape: (100, 64)   # 100 values, each 64-dim

Attention matrix: Q @ K^T = (100, 64) @ (64, 100) = (100, 100)
                 = 10,000 entries!

Each entry = similarity between one query and one key
```

**Memory/compute implications:**
- 10,000 entries per head × 8 heads = 80,000 attention weights
- This is why Transformers struggle with very long sequences (quadratic in length)

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

---

### How Do Heads Learn Different Things? (IMPORTANT)

**This is NOT enforced! It's EMERGENT.**

#### There's No Explicit Constraint

The architecture does NOT say:
- "Head 1, you must learn syntax"
- "Head 2, you must learn semantics"

Each head has its own W_Q, W_K, W_V matrices that are randomly initialized and trained by gradient descent. Nothing forces them to specialize.

#### Why Specialization Happens Anyway

**Reason 1: Different initializations**

Each head starts with random W_Q, W_K, W_V matrices:
```
Head 1: W_Q^1 = random, W_K^1 = random, W_V^1 = random
Head 2: W_Q^2 = random (different!), W_K^2 = random, W_V^2 = random
```

Different starting points → different local minima → different learned patterns.

**Reason 2: Gradient diversity**

The loss function is:
```
Loss = -log P(correct next word)
```

To minimize loss, the model needs to capture MANY types of relationships. If all heads learned the same thing, they'd be redundant — the gradient would push them toward different patterns.

**Reason 3: Capacity pressure**

Each head only has 64 dimensions (vs 512 for the full model). It CAN'T capture everything. So each head learns a **subset** of useful patterns.

Think of it like:
- 8 people (heads) need to watch a football game
- Each person can only focus on a small part of the field
- Naturally, they'll spread out to cover more ground

#### What Heads Actually Learn (Empirically Observed)

Researchers have visualized attention patterns and found:
```
Layer 1, Head 3: Often attends to previous word (positional)
Layer 2, Head 7: Often attends to syntactic dependencies (subject-verb)
Layer 5, Head 2: Often attends to semantic relations (word meanings)
```

But these patterns are:
- **Emergent** (not programmed)
- **Fuzzy** (heads aren't perfectly specialized)
- **Task-dependent** (different tasks → different patterns)
- **Layer-dependent** (early layers vs late layers behave differently)

#### Could Heads Learn the Same Thing?

Yes! And sometimes they do. This is called **redundancy**.

Some research has shown you can **prune** (remove) many attention heads with minimal performance loss — they were redundant.

---

### Geometric Interpretation

Each head projects Q, K, V into a **different subspace**:

```
Original space: 512 dimensions

Head 1: Projects to 64-dim subspace S₁
Head 2: Projects to 64-dim subspace S₂ (different from S₁!)
Head 3: Projects to 64-dim subspace S₃
...
Head 8: Projects to 64-dim subspace S₈
```

**Analogy**: Looking at a 3D object from 8 different camera angles
- Each camera sees a 2D projection
- Different angles reveal different features
- Combining all views gives richer information than any single view

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

## Positional Encoding (First Principles)

### The Problem

Self-attention is **permutation equivariant** - it treats "cat sat mat" the same as "mat cat sat". But word order matters!

We need a way to tell the model: "This is position 1, this is position 2, ..."

---

### Approach 1 (Simple but Bad): Just Use Position Numbers

```
Position 0: Add 0 to embedding
Position 1: Add 1 to embedding
Position 2: Add 2 to embedding
...
Position 1000: Add 1000 to embedding
```

**Problems:**
1. **Unbounded**: Position 1000 has value 1000, but embeddings might be in range [-1, 1]. This would dominate the embedding!
2. **No structure**: The model can't easily learn that position 5 and position 6 are "close"
3. **Training dependency**: If you only trained on sequences up to length 100, what happens at position 500?

---

### Approach 2 (Better): Normalize to [0, 1]

```
Position 0: Add 0.0
Position 1: Add 0.01  (if max length is 100)
Position 2: Add 0.02
...
Position 100: Add 1.0
```

**Problems:**
1. **Depends on sequence length**: Position 50 in a 100-word sequence ≠ position 50 in a 200-word sequence
2. **Still no structure**: Doesn't capture that positions 1 and 2 are adjacent

---

### Approach 3 (The Solution): Sinusoidal Encodings

**Key insight**: Use **waves of different frequencies**, like a radio signal.

Think of how you might describe a time uniquely:
- "It's 3 o'clock" (hour hand position)
- "It's 3:45" (hour + minute hand)
- "It's 3:45:30" (hour + minute + second hand)

Each "hand" cycles at a different frequency. Together, they uniquely identify ANY time.

---

### Understanding Sinusoidal Encoding Step by Step

#### The Formula

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Let me unpack this piece by piece.

#### What is 2i and 2i+1?

These are the **dimension indices** of the positional encoding vector.

If d_model = 512:
- Dimension 0: use sin (2i where i=0)
- Dimension 1: use cos (2i+1 where i=0)
- Dimension 2: use sin (2i where i=1)
- Dimension 3: use cos (2i+1 where i=1)
- ...
- Dimension 510: use sin (2i where i=255)
- Dimension 511: use cos (2i+1 where i=255)

Each pair of dimensions (0-1, 2-3, 4-5, ...) uses sin and cos at the same frequency.

#### What is the Frequency?

The denominator `10000^(2i/d_model)` determines the **wavelength**.

```
Dimension 0-1:   wavelength = 10000^(0/512) = 10000^0 = 1
                 → cycles every 2π ≈ 6.28 positions

Dimension 2-3:   wavelength = 10000^(2/512) = 10000^0.0039 ≈ 1.036
                 → cycles every 6.5 positions

Dimension 254-255: wavelength = 10000^(254/512) ≈ 316
                   → cycles every ~2000 positions

Dimension 510-511: wavelength = 10000^(510/512) ≈ 9950
                   → cycles every ~62,500 positions
```

**The first dimensions oscillate FAST (distinguish nearby positions)**
**The last dimensions oscillate SLOW (distinguish distant positions)**

#### Concrete Example: d_model = 4 (simplified)

```
pos = 0:
  dim 0: sin(0 / 10000^0) = sin(0) = 0
  dim 1: cos(0 / 10000^0) = cos(0) = 1
  dim 2: sin(0 / 10000^0.5) = sin(0) = 0
  dim 3: cos(0 / 10000^0.5) = cos(0) = 1
  PE(0) = [0, 1, 0, 1]

pos = 1:
  dim 0: sin(1 / 1) = sin(1) ≈ 0.84
  dim 1: cos(1 / 1) = cos(1) ≈ 0.54
  dim 2: sin(1 / 100) = sin(0.01) ≈ 0.01
  dim 3: cos(1 / 100) = cos(0.01) ≈ 1.0
  PE(1) = [0.84, 0.54, 0.01, 1.0]

pos = 2:
  dim 0: sin(2) ≈ 0.91
  dim 1: cos(2) ≈ -0.42
  dim 2: sin(0.02) ≈ 0.02
  dim 3: cos(0.02) ≈ 1.0
  PE(2) = [0.91, -0.42, 0.02, 1.0]
```

Notice:
- Dimensions 0-1 change a lot between positions (fast oscillation)
- Dimensions 2-3 change slowly (slow oscillation)

---

### Why Sin AND Cos?

Using both sin and cos makes it easy to represent **relative positions** as linear transformations.

**Mathematical fact:**
```
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
```

This means:
```
PE(pos + k) = some_linear_transform(PE(pos))
```

The model can learn to compute "what's 5 positions ahead?" with a simple matrix multiplication!

---

### Geometric View: The Torus Explained

#### What is a Torus?

A **torus** is a donut shape:

```
     ___---___
   /           \
  |    ___     |
  |   /   \    |
   \_|     |__/
      \___/
```

A point on a torus can be described by two angles:
- θ: angle around the "big circle" (around the donut)
- φ: angle around the "small circle" (around the tube)

Each (θ, φ) pair uniquely identifies a point on the torus.

#### How Positional Encoding Relates to a Torus

Each pair of dimensions (sin, cos) defines a **circle**:
```
(sin(θ), cos(θ))  ← this traces out a circle as θ varies!
```

With multiple pairs at different frequencies:
```
Pair 1: (sin(pos·f₁), cos(pos·f₁))     → circle in dims 0-1
Pair 2: (sin(pos·f₂), cos(pos·f₂))     → circle in dims 2-3
Pair 3: (sin(pos·f₃), cos(pos·f₃))     → circle in dims 4-5
```

**Each pair is a circle.** Multiple circles at different frequencies = **high-dimensional torus**.

#### Visualization in 2D (One Pair)

```
Position 0: (sin(0), cos(0)) = (0, 1)       ↑ top of circle
Position 1: (sin(1), cos(1)) ≈ (0.84, 0.54) ↗ moving clockwise
Position 2: (sin(2), cos(2)) ≈ (0.91, -0.42) → right side
...
Position 6: (sin(6), cos(6)) ≈ (-0.28, 0.96) ↑ back near top

After ~6.28 positions, we're back where we started (one full circle)!
```

#### Why Multiple Frequencies Matter

With ONLY one frequency:
```
Position 0 and Position 6.28 would look identical! (wrapped around)
```

With multiple frequencies:
```
Position 0:    fast_circle = (0, 1),      slow_circle = (0, 1)
Position 6.28: fast_circle = (0, 1),      slow_circle = (0.0006, 1.0)  ← DIFFERENT!
```

The slow dimensions haven't wrapped around yet, so we can still distinguish them.

**With enough frequencies spanning from fast to slow, every position up to ~10,000 has a unique encoding.**

---

### Why Not Just Learn Positional Embeddings?

Some models (like BERT) use **learned positional embeddings**:
```python
position_embedding = nn.Embedding(max_length, d_model)
```

This works! But sinusoidal has advantages:
1. **Extrapolation**: Works for sequences longer than training data
2. **No parameters**: One less thing to learn
3. **Relative positions**: The linear relationship property

The original Transformer paper tested both — they performed similarly. But sinusoidal is more principled.

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
