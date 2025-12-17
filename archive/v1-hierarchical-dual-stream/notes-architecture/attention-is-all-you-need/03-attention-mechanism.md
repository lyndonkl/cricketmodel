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

### CRITICAL: How Does Attention Produce a NEW Word? (Not in Query/Keys)

This is a common confusion! Let me clarify the full pipeline.

#### The Confusion

You might think:
```
Input: "The cat"
Attention looks at: "The", "cat" (Q, K, V all from these 2 words)
Output: ??? How can it produce "sat" if "sat" was never in the input?
```

#### The Answer: Attention Output ≠ The Next Word

Attention outputs are **dense vectors**, NOT words!

```
Input words:     ["The", "cat"]
                     ↓
Embeddings:      [[0.1, 0.2, ...], [0.3, 0.4, ...]]   # Shape: (2, 512)
                     ↓
After Attention: [[0.15, 0.25, ...], [0.35, 0.45, ...]]  # Shape: (2, 512)
                                              ↑
                          Still 512-dimensional vectors!
                          NOT words!
```

#### The Missing Step: Output Projection to Vocabulary

After all the Transformer layers, there's a **final linear layer**:

```
Attention output for last position: [0.35, 0.45, ..., 0.82]  # 512 numbers

Final Linear Layer: W_vocab of shape (512, vocab_size)
                   vocab_size = 30,000 (all possible words!)

Logits = attention_output @ W_vocab
       = [512 numbers] @ [512 × 30,000]
       = [30,000 numbers]  # One score for EVERY word in vocabulary!

Probabilities = softmax(logits)
              = [0.001, 0.0005, ..., 0.15, ..., 0.002]
                  ↑        ↑           ↑
                "a"     "aardvark"   "sat"
```

**The 30,000-dimensional output gives a probability for EVERY possible next word**, including words that were never in the input!

#### Complete Example

```
Step 1: Input
        Words: ["The", "cat"]

Step 2: Embed
        Embeddings: [[0.1, 0.2, ...512 dims...],
                     [0.3, 0.4, ...512 dims...]]

Step 3: Add Positional Encoding
        With positions: [[0.1+PE₀], [0.3+PE₁]]

Step 4: Pass through N Transformer layers (attention + FFN)
        Output: [[new 512 dims for "The"],
                 [new 512 dims for "cat"]]  ← we use THIS one

Step 5: Take LAST position's output
        Last output: [0.35, 0.45, ...512 dims...]

Step 6: Project to vocabulary
        W_vocab: learned matrix of shape (512, 30000)
        Logits = last_output @ W_vocab = [30,000 scores]

Step 7: Softmax
        Probs = softmax(logits)

        Word        Probability
        ----        -----------
        "the"       0.001
        "sat"       0.15      ← highest!
        "ran"       0.08
        "dog"       0.02
        "meowed"    0.05
        ...         ...

Step 8: Sample or Argmax
        Next word = "sat" (highest probability)

Step 9: Append and repeat
        New input: ["The", "cat", "sat"]
        Go back to Step 2...
```

#### Key Insight: The Vocabulary Projection

The magic is in `W_vocab`:

```python
# This is learned during training!
W_vocab = nn.Linear(d_model, vocab_size)  # (512 → 30,000)
```

This matrix learns to map the 512-dimensional "meaning" space to probabilities over all 30,000 words.

- If the 512-dim vector is "close to" the concept of a verb → high probability for verbs
- If it captures "animal doing something" → "sat", "ran", "jumped" get boosted

**The model doesn't just copy from input! It maps to the ENTIRE vocabulary.**

#### Analogy

Think of it like a translator's brain:

```
Input (French): "Le chat"

Translator's brain:
1. Understands "Le chat" → forms internal representation
2. Internal representation activates concepts: [small, furry, animal, pet, ...]
3. Maps concepts to ALL English words
4. "cat" has highest activation
5. Output: "cat"
```

The translator doesn't search through the French input for English words. They map their understanding to the entire English vocabulary.

Similarly:
```
Input: "The cat"

Transformer:
1. Processes "The cat" → rich 512-dim representation
2. This representation encodes: [subject established, animal, expecting action, ...]
3. Maps to ALL 30,000 vocabulary words
4. "sat", "ran", "slept" have high probability (verbs for animals)
5. Samples → "sat"
```

#### Where Q, K, V Fit In

Q, K, V are for **mixing information between positions**, not for outputting words.

```
"The cat" → Attention → "The cat" (but now each word knows about the other)
                              ↓
                       "cat" representation now includes:
                       - Its own meaning
                       - That "The" came before it (article + noun pattern)
                       - That it's the subject (likely to be followed by verb)
                              ↓
                       Project to 30,000 words
                              ↓
                       "sat" has high probability
```

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

## Why Use the LAST Embedding During Inference?

### The Question

During autoregressive generation, we have:
```
Input: ["The", "cat", "sat"]
       Position 0, Position 1, Position 2

After Transformer: Three output embeddings
       [emb_0', emb_1', emb_2']

We project emb_2' (the LAST one) to vocabulary → predict next word

But wait - why not use emb_0' and emb_1' too?
```

### The Answer: The Last Position Already Contains All Information

**The last embedding ISN'T ignoring the previous embeddings. It HAS SEEN them all through attention!**

```
What each output embedding knows:

emb_0' (output for "The"):
  - Attended to: itself only (or itself + "cat" + "sat" in encoder)
  - Knows: "The" in context of sentence
  - Useful for: Nothing about what comes after position 2

emb_1' (output for "cat"):
  - Attended to: "The", "cat" (in causal/decoder setup)
  - Knows: "cat" follows "The"
  - Useful for: Would predict what comes after position 1

emb_2' (output for "sat"):  ← THIS IS WHAT WE WANT
  - Attended to: "The", "cat", "sat" (ALL previous positions!)
  - Knows: Full context up to position 2
  - Useful for: Predicting what comes at position 3
```

### Visual Explanation

```
Layer 1 Attention:
                 Attends to
    "The"    →   ["The"]
    "cat"    →   ["The", "cat"]
    "sat"    →   ["The", "cat", "sat"]  ← sees everything!

Layer 2 Attention:
                 Attends to (now with Layer 1's enriched representations)
    "The"    →   [enriched "The"]
    "cat"    →   [enriched "The", enriched "cat"]
    "sat"    →   [enriched "The", enriched "cat", enriched "sat"]

...after N layers...

Final "sat" embedding contains:
  ├── Information from "The" (flowed through attention)
  ├── Information from "cat" (flowed through attention)
  ├── Its own "sat" information
  └── All relationships between them!
```

### The Last Position is a SUMMARY

Think of it like a meeting:
```
Person 1 speaks → takes notes on what they said
Person 2 speaks → takes notes on Person 1 + 2
Person 3 speaks → takes notes on ALL speakers

At the end, Person 3's notes are the most complete!
```

The last position's embedding is a **compressed summary** of the entire sequence, enriched by attention across all layers.

### Why Not Concatenate All Embeddings?

You COULD average or concatenate all output embeddings:
```
final_repr = Average(emb_0', emb_1', emb_2')
           or
final_repr = Concat(emb_0', emb_1', emb_2')
```

But this has problems:
1. **Variable length**: Different sequences have different lengths → different output sizes
2. **Redundant**: emb_2' already contains information from positions 0 and 1
3. **Less focused**: emb_0' is optimized for "what follows The" not "what follows sat"

### For Classification vs Generation

**Generation (next token prediction)**:
- Use LAST position → it summarizes everything, predicts what's NEXT
```
["The", "cat", "sat"] → use emb_2' → predict "on"
```

**Classification (e.g., sentiment)**:
- Often use a SPECIAL token like [CLS] at position 0
- OR average all embeddings
- OR use the last token's embedding
```
["[CLS]", "I", "love", "this", "movie"] → use emb_0' (CLS) → predict "positive"
```

### Critical Clarification: Attention IS How Information Flows

**The last position doesn't start with all the information. It GAINS information through attention!**

```
BEFORE attention (just embeddings):
  emb_0 = embedding("The") + PE(0)   → knows only about "The"
  emb_1 = embedding("cat") + PE(1)   → knows only about "cat"
  emb_2 = embedding("sat") + PE(2)   → knows only about "sat"

  At this point, emb_2 knows NOTHING about "The" or "cat"!

AFTER attention:
  emb_0' = attention(emb_0, [emb_0])              → still just "The"
  emb_1' = attention(emb_1, [emb_0, emb_1])       → "cat" + info from "The"
  emb_2' = attention(emb_2, [emb_0, emb_1, emb_2]) → "sat" + info from ALL

  NOW emb_2' contains information from the whole sequence!
```

**Attention is not optional during inference — it IS the mechanism that gathers information.**

### Why We Must Run Attention During Inference

You might think: "If we just want emb_2', and it needs info from emb_0 and emb_1, can't we cache that?"

**Answer**: We DO cache, but we still need to compute attention.

```
Inference step-by-step (generating "The cat sat on"):

Step 1: Input ["The"]
  - Compute attention for position 0
  - Get emb_0'
  - Project to vocab → predict "cat"
  - Cache: K_0, V_0 for position 0

Step 2: Input ["The", "cat"]
  - Position 0: Use cached K_0, V_0 (don't recompute!)
  - Position 1: Compute new K_1, V_1
  - Compute attention for position 1: attends to positions 0,1
  - Get emb_1'
  - Project to vocab → predict "sat"
  - Cache: K_0, V_0, K_1, V_1

Step 3: Input ["The", "cat", "sat"]
  - Positions 0,1: Use cached keys/values
  - Position 2: Compute new K_2, V_2
  - Compute attention for position 2: attends to positions 0,1,2
  - Get emb_2'
  - Project to vocab → predict "on"
```

**Key insight**: We cache the Keys and Values, but we MUST compute attention for the new position. The new position needs to "look at" all previous positions to gather their information.

### Training vs Inference: What's Different?

| Aspect | Training | Inference |
|--------|----------|-----------|
| Input | Full sequence known | Build sequence token-by-token |
| Attention computed for | All positions (in parallel) | New position only (using cached K,V) |
| Outputs used | ALL positions (each predicts next) | Only LAST position |
| Gradients | Yes (backprop through attention) | No |
| KV Cache | Not needed (see full sequence) | Essential (avoid recomputation) |

**Training example**:
```
Input:  ["The", "cat", "sat", "on"]
Target: ["cat", "sat", "on", "the"]  (shifted by 1)

Compute attention for ALL positions simultaneously:
  Position 0 → predicts "cat" (loss computed)
  Position 1 → predicts "sat" (loss computed)
  Position 2 → predicts "on"  (loss computed)
  Position 3 → predicts "the" (loss computed)

Total loss = sum of all position losses
Backpropagate through everything
```

**Inference example**:
```
Start: ["The"]
  Attention for pos 0 → predict "cat"

Now: ["The", "cat"]
  Attention for pos 1 (using cached pos 0) → predict "sat"

Now: ["The", "cat", "sat"]
  Attention for pos 2 (using cached pos 0,1) → predict "on"

...continue until done
```

### The KV Cache Explained

Why cache Keys and Values specifically?

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

For position n attending to positions 0...n:
  Q_n = new (depends on current token)
  K_{0:n} = [K_0, K_1, ..., K_n]  ← K_0 to K_{n-1} are UNCHANGED
  V_{0:n} = [V_0, V_1, ..., V_n]  ← V_0 to V_{n-1} are UNCHANGED

Only K_n and V_n are new. So cache K_{0:n-1} and V_{0:n-1}!
```

This is why large language models need significant memory during inference — they're storing K and V for every layer, every head, for the entire context.

```
Memory for KV cache:
  = 2 (K and V) × num_layers × num_heads × seq_length × head_dim × bytes_per_float

For a 32-layer model with 32 heads, d_head=128, seq_len=4096, float16:
  = 2 × 32 × 32 × 4096 × 128 × 2 bytes
  = 2.1 GB just for KV cache!
```

### Cricket Implication

For ball-by-ball prediction:
```
Balls: [ball_1, ball_2, ..., ball_n]

After Transformer: [emb_1', emb_2', ..., emb_n']

To predict ball n+1:
  → Use emb_n' (it has seen balls 1 through n through attention!)
  → Project to outcome probabilities
```

The last ball's embedding knows about:
- The first ball of the innings
- Every wicket that fell
- The current batsman's form (all their balls)
- The bowler's pattern (all their balls)
- Everything! (weighted by learned relevance)

---

## Sinusoidal Embeddings: A Complete Worked Example

### The Goal

We need to give each position a unique "fingerprint" that:
1. Is bounded (values between -1 and 1)
2. Allows nearby positions to be similar
3. Allows the model to learn "position 5 is 3 steps after position 2"
4. Works for any sequence length

### Setup: Tiny Example

Let's use:
- d_model = 4 (just 4 dimensions, for clarity)
- Positions 0, 1, 2, 3, 4

The formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Computing the Wavelengths

First, let's understand the denominators:

```
d_model = 4

Dimension 0 (i=0): 10000^(2*0/4) = 10000^0 = 1
Dimension 1 (i=0): 10000^(2*0/4) = 10000^0 = 1
  → wavelength = 2π ≈ 6.28 positions

Dimension 2 (i=1): 10000^(2*1/4) = 10000^0.5 = 100
Dimension 3 (i=1): 10000^(2*1/4) = 10000^0.5 = 100
  → wavelength = 2π * 100 ≈ 628 positions
```

So dimensions 0-1 cycle every ~6 positions (FAST)
And dimensions 2-3 cycle every ~628 positions (SLOW)

### Computing Each Position's Encoding

```
Position 0:
  dim 0: sin(0 / 1)   = sin(0)    = 0.000
  dim 1: cos(0 / 1)   = cos(0)    = 1.000
  dim 2: sin(0 / 100) = sin(0)    = 0.000
  dim 3: cos(0 / 100) = cos(0)    = 1.000

  PE(0) = [0.000, 1.000, 0.000, 1.000]

Position 1:
  dim 0: sin(1 / 1)   = sin(1)    = 0.841
  dim 1: cos(1 / 1)   = cos(1)    = 0.540
  dim 2: sin(1 / 100) = sin(0.01) = 0.010
  dim 3: cos(1 / 100) = cos(0.01) = 1.000

  PE(1) = [0.841, 0.540, 0.010, 1.000]

Position 2:
  dim 0: sin(2)       = 0.909
  dim 1: cos(2)       = -0.416
  dim 2: sin(0.02)    = 0.020
  dim 3: cos(0.02)    = 1.000

  PE(2) = [0.909, -0.416, 0.020, 1.000]

Position 3:
  dim 0: sin(3)       = 0.141
  dim 1: cos(3)       = -0.990
  dim 2: sin(0.03)    = 0.030
  dim 3: cos(0.03)    = 1.000

  PE(3) = [0.141, -0.990, 0.030, 1.000]

Position 4:
  dim 0: sin(4)       = -0.757
  dim 1: cos(4)       = -0.654
  dim 2: sin(0.04)    = 0.040
  dim 3: cos(0.04)    = 0.999

  PE(4) = [-0.757, -0.654, 0.040, 0.999]
```

### Visualizing the Pattern

```
Position | Dim 0   Dim 1  | Dim 2   Dim 3
         | (fast)  (fast) | (slow)  (slow)
---------|----------------|----------------
    0    |  0.00   1.00   |  0.00   1.00
    1    |  0.84   0.54   |  0.01   1.00
    2    |  0.91  -0.42   |  0.02   1.00
    3    |  0.14  -0.99   |  0.03   1.00
    4    | -0.76  -0.65   |  0.04   1.00
    5    | -0.96   0.28   |  0.05   1.00
    6    | -0.28   0.96   |  0.06   1.00  ← dims 0-1 nearly back to start!
```

**Key observation**:
- Dims 0-1 cycle rapidly (sin/cos complete a cycle around position 6)
- Dims 2-3 barely change (they'd take ~628 positions to cycle)

### Property 1: All Values Bounded

Every value is between -1 and 1 (sin and cos always are).

This means positional encodings don't dominate the learned embeddings:
```
Word embedding for "cat": [0.5, -0.3, 0.8, 0.1, ...]  (values typically small)
Positional encoding:      [0.84, 0.54, 0.01, 1.0]    (also small!)

Combined: [0.5+0.84, -0.3+0.54, ...]  ← both contribute roughly equally
```

### Property 2: Nearby Positions Are Similar

Let's compute distances:
```
Distance(PE(0), PE(1)) = √[(0-0.84)² + (1-0.54)² + (0-0.01)² + (1-1)²]
                       = √[0.71 + 0.21 + 0.0001 + 0]
                       = √0.92 ≈ 0.96

Distance(PE(0), PE(3)) = √[(0-0.14)² + (1-(-0.99))² + (0-0.03)² + (1-1)²]
                       = √[0.02 + 3.96 + 0.001 + 0]
                       = √3.98 ≈ 2.0

Distance(PE(0), PE(100)) = √[... + (0-0.86)² + (1-0.51)²]  (slow dims differ too!)
                         ≈ much larger
```

**Nearby positions have similar encodings; distant positions have different encodings.**

### Property 3: Relative Positions via Linear Transform

Here's the magical property. For any fixed offset k, there exists a matrix M_k such that:

```
PE(pos + k) = M_k × PE(pos)
```

This is because of the trigonometric identities:
```
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
```

**For k=1 (shifting by one position)**:

```
PE(pos + 1) for dims 0-1:

sin(pos + 1) = sin(pos)cos(1) + cos(pos)sin(1)
cos(pos + 1) = cos(pos)cos(1) - sin(pos)sin(1)

In matrix form:
[sin(pos+1)]   [cos(1)   sin(1) ] [sin(pos)]
[cos(pos+1)] = [-sin(1)  cos(1) ] [cos(pos)]

             = [0.54   0.84] [sin(pos)]
               [-0.84  0.54] [cos(pos)]
```

**This is a ROTATION MATRIX!** Shifting by 1 position = rotating in the (sin, cos) plane.

The model can learn this matrix to reason about relative positions:
- "What's 3 positions back?" → multiply by M_{-3}
- "Is position j close to position i?" → compare rotated vectors

### Implications for Learning

**The model doesn't need to memorize all position relationships!**

Instead of learning:
```
"position 5 relates to position 2 in this way"
"position 100 relates to position 97 in this way"
"position 1000 relates to position 997 in this way"
```

The model can learn ONE transformation ("shift by 3") that works for ALL positions:
```
"k positions back" = apply M_{-k}
```

This is why sinusoidal encodings **generalize to longer sequences** than seen during training.

---

## Application to Cricket Ball-by-Ball Prediction

### The Standard NLP Positional Encoding

In language:
```
Position 0 → "The"
Position 1 → "cat"
Position 2 → "sat"
```

Position just means "index in sequence."

### Cricket Has Richer Structure

In cricket, a ball has multiple "positions":

```
Ball example:
- Absolute position: Ball 47 of the innings
- Ball in over: 5th ball of the over (ball 47 = over 7, ball 5)
- Over number: Over 7 of 20
- Phase: Death overs (overs 16-20)
- Batsman's ball: This is the 23rd ball the batsman has faced
- Bowler's ball: This is the 18th ball the bowler has bowled
```

Each of these "positions" provides different information!

### Multi-Dimensional Positional Encoding for Cricket

Instead of single sinusoidal encoding, we could use:

```python
def cricket_positional_encoding(ball):
    PE = []

    # 1. Absolute ball position (standard sinusoidal)
    PE_absolute = sinusoidal(ball.absolute_position, d=128)
    PE.append(PE_absolute)

    # 2. Ball-in-over position (1-6, learned embedding might be better)
    PE_ball_in_over = sinusoidal(ball.ball_in_over, d=32)
    # or: PE_ball_in_over = learned_embedding(ball.ball_in_over)
    PE.append(PE_ball_in_over)

    # 3. Over number (0-19 for T20)
    PE_over = sinusoidal(ball.over_number, d=64)
    PE.append(PE_over)

    # 4. Phase encoding (powerplay=0, middle=1, death=2)
    PE_phase = learned_embedding(ball.phase)  # Just 3 values, learn them
    PE.append(PE_phase)

    # Concatenate all
    return concat(PE)  # Total: 128 + 32 + 64 + 32 = 256 dims
```

### Why This Helps

**Standard sinusoidal for absolute position**:
- Ball 47 can attend to ball 41 and know "that was 6 balls ago" (same bowler's last over)
- The linear transform property helps!

**Ball-in-over position (1-6)**:
- Ball 1 of an over is different from ball 6
- First ball: batsman facing new bowler, adjusting
- Last ball: bowler's final chance for the over

```
Attention might learn:
"If I'm ball 6, attend strongly to ball 1 of same over"
  → understand the bowler's plan for the over
  → detect dot-ball pressure building
```

**Over number (0-19)**:
- Over 1 (powerplay) is very different from over 18 (death)
- Even if same batsman/bowler combo, strategy differs

```
Same batsman + bowler:
  Over 2: Batsman defensive, survival mode
  Over 18: Same batsman attacking, looking for boundaries

The over positional encoding helps distinguish these!
```

**Phase encoding**:
- Explicit signal: "we're in death overs"
- Model doesn't need to LEARN that overs 16-20 are special

### What Attention Patterns Might Emerge

With these rich positional encodings, attention heads might specialize:

```
Head 1: "Same over" attention
  - Ball 6 attends strongly to balls 1-5 of same over
  - Uses: Ball-in-over encoding to identify same over

Head 2: "Same bowler" attention
  - Current ball attends to all balls by same bowler
  - Uses: Absolute position to know "6 balls ago = same bowler"

Head 3: "Same batsman" attention
  - Current ball attends to all balls faced by current batsman
  - Uses: Batsman embedding (not positional, but relevant)

Head 4: "Phase context" attention
  - Powerplay balls attend to other powerplay balls
  - Death overs attend to other death-over patterns
  - Uses: Phase encoding
```

### Learned vs Fixed Encodings for Cricket

**Sinusoidal (fixed) works well for**:
- Absolute ball position (hundreds of positions)
- Over number (20 distinct values, but linear relationship matters)

**Learned embeddings might work better for**:
- Ball-in-over (only 6 values, each qualitatively different)
- Phase (only 3 values, might have complex meaning)

```python
class CricketPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Fixed sinusoidal for absolute position
        self.register_buffer('PE_absolute',
                             create_sinusoidal(max_balls=300, d=256))

        # Learned for ball-in-over (6 positions)
        self.ball_in_over_embed = nn.Embedding(6, 64)

        # Learned for over (20 overs in T20)
        self.over_embed = nn.Embedding(20, 64)

        # Learned for phase (3 phases)
        self.phase_embed = nn.Embedding(3, 32)

        # Project to d_model
        self.proj = nn.Linear(256 + 64 + 64 + 32, d_model)

    def forward(self, ball_positions, balls_in_over, overs, phases):
        pe_abs = self.PE_absolute[ball_positions]  # (batch, seq, 256)
        pe_bio = self.ball_in_over_embed(balls_in_over)  # (batch, seq, 64)
        pe_over = self.over_embed(overs)  # (batch, seq, 64)
        pe_phase = self.phase_embed(phases)  # (batch, seq, 32)

        combined = torch.cat([pe_abs, pe_bio, pe_over, pe_phase], dim=-1)
        return self.proj(combined)  # (batch, seq, d_model)
```

### Key Insight for Cricket

**Cricket has multiple natural notions of "position"** — unlike text where position is just index.

By giving the model rich positional information:
1. It can learn attention patterns specific to overs (balls 1-6)
2. It can recognize phase-specific strategies
3. It can connect balls by the same bowler (6 positions apart in same over)

This is **domain knowledge encoded as inductive bias** — we're telling the model "these positional relationships matter" rather than making it discover them from scratch.

---

## Summary: Attention's Role

| Aspect | What Attention Does |
|--------|-------------------|
| Information routing | Dynamically selects what to look at |
| Relationship learning | Discovers which positions relate |
| Graph construction | Builds soft adjacency from content |
| Global receptive field | O(1) path between any positions |
| Parallelization | All positions computed simultaneously |
