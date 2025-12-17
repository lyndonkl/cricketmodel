# Transformer Architecture Overview

## The Big Picture

The Transformer is an **encoder-decoder** architecture for sequence-to-sequence tasks:

```
Input Sequence → [ENCODER] → Context Representation → [DECODER] → Output Sequence
   (x₁...xₙ)                       (z₁...zₙ)                        (y₁...yₘ)
```

Both encoder and decoder are built from stacks of identical layers, each containing:
1. **Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network**
3. **Residual connections + Layer Normalization**

---

## Architecture Abstraction Ladder

### L1: Universal Principle
> **Every position in a sequence should be able to directly gather information from every other position, weighted by relevance.**

### L2: Framework - The Encoder-Decoder Pattern
- **Encoder**: Builds a rich contextual representation of the input
- **Decoder**: Generates output autoregressively, attending to both its own outputs and the encoder's representation

### L3: Method - Layer Composition
Each layer applies:
```
output = LayerNorm(x + SubLayer(x))
```
Where SubLayer is either attention or feed-forward.

### L4: Components - The Building Blocks

| Component | Purpose | Geometric Interpretation |
|-----------|---------|-------------------------|
| Multi-Head Attention | Learn relationships | Dynamic graph construction |
| Feed-Forward Network | Transform representations | Point-wise manifold deformation |
| Positional Encoding | Inject sequence order | Break permutation symmetry |
| Residual Connections | Enable gradient flow | Identity mapping baseline |
| Layer Normalization | Stabilize training | Normalize on hypersphere |

### L5: Concrete Dimensions (Base Model)
- `d_model = 512` (embedding/representation dimension)
- `d_ff = 2048` (feed-forward inner dimension)
- `h = 8` (number of attention heads)
- `d_k = d_v = 64` (key/value dimension per head)
- `N = 6` (number of encoder/decoder layers)

---

## Encoder Stack

```
┌─────────────────────────────────┐
│         Encoder Layer ×N        │
│  ┌───────────────────────────┐  │
│  │    Multi-Head Attention    │  │  ← Self-attention over input
│  │    (Self-Attention)        │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │   Add & Norm (Residual)   │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │  Position-wise Feed-Fwd   │  │  ← Independent transformation
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │   Add & Norm (Residual)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

**Key property**: Each position can attend to ALL positions in the input.

---

## Decoder Stack

```
┌─────────────────────────────────┐
│         Decoder Layer ×N        │
│  ┌───────────────────────────┐  │
│  │   Masked Multi-Head Attn   │  │  ← Self-attention (causal mask)
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │   Add & Norm (Residual)   │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │    Multi-Head Attention    │  │  ← Cross-attention to encoder
│  │    (Encoder-Decoder)       │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │   Add & Norm (Residual)   │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │  Position-wise Feed-Fwd   │  │
│  └───────────────────────────┘  │
│              ↓                   │
│  ┌───────────────────────────┐  │
│  │   Add & Norm (Residual)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

**Key property**: Decoder self-attention is **masked** to prevent attending to future positions (preserving autoregressive property).

---

## Three Types of Attention in the Transformer

### 1. Encoder Self-Attention
- **Query, Key, Value**: All from encoder
- **Purpose**: Each input position attends to all input positions
- **Masking**: None

### 2. Decoder Self-Attention (Masked)
- **Query, Key, Value**: All from decoder
- **Purpose**: Each output position attends to previous output positions only
- **Masking**: Future positions masked with -∞

### 3. Encoder-Decoder Attention (Cross-Attention)
- **Query**: From decoder
- **Key, Value**: From encoder output
- **Purpose**: Decoder attends to relevant parts of input
- **Masking**: None

---

## Geometric Deep Learning View (First Principles)

### Part 1: What is a Permutation?

A **permutation** is a reordering of elements.

**Example**: Given the list [A, B, C], some permutations are:
```
Original:    [A, B, C]
Permutation 1: [B, A, C]  (swap first two)
Permutation 2: [C, B, A]  (reverse)
Permutation 3: [A, C, B]  (swap last two)
Permutation 4: [B, C, A]  (rotate left)
Permutation 5: [C, A, B]  (rotate right)
```

We can represent a permutation as a mapping:
```
π: position → new_position

For [A, B, C] → [C, A, B]:
π(1) = 2  (position 1 goes to position 2)
π(2) = 3  (position 2 goes to position 3)
π(3) = 1  (position 3 goes to position 1)
```

---

### Part 2: What is Permutation Equivariance?

A function f is **permutation equivariant** if:
```
f(permute(input)) = permute(f(input))
```

In words: **Permute then process = Process then permute**

#### Concrete Example: Element-wise Squaring

The function f(x) = x² applied element-wise:
```
Input:           [2, 3, 5]
Output:          [4, 9, 25]

Permuted input:  [5, 2, 3]
Output:          [25, 4, 9]

Is this equal to permuting the original output?
Original output: [4, 9, 25]
Permuted:        [25, 4, 9] ✓ YES!
```

Element-wise operations are permutation equivariant.

#### Another Example: Sum (Permutation Invariant)

The function f(x) = sum(x):
```
Input:           [2, 3, 5]  →  sum = 10
Permuted input:  [5, 2, 3]  →  sum = 10

Same output regardless of order!
```

This is **permutation invariant** (a special case of equivariance where the output doesn't change at all).

---

### Part 3: Why is Self-Attention Permutation Equivariant?

Let's trace through attention step by step.

**Setup**: Three words as vectors (ignoring position for now)
```
Input: [dog, cat, sat]  →  vectors [v_dog, v_cat, v_sat]
```

**Attention computation for "dog"**:
1. Query from dog: q_dog = v_dog · W_Q
2. Keys from all: k_dog, k_cat, k_sat
3. Attention scores: [q_dog·k_dog, q_dog·k_cat, q_dog·k_sat]
4. Softmax → weights: [0.2, 0.5, 0.3]
5. Output: 0.2·v_dog + 0.5·v_cat + 0.3·v_sat

**Key observation**: The attention weights depend ONLY on the content of the vectors, not their positions!

**Now permute the input**: [cat, sat, dog]
```
Input: [cat, sat, dog]  →  vectors [v_cat, v_sat, v_dog]
```

**Attention computation for "dog" (now at position 3)**:
1. Query from dog: q_dog = v_dog · W_Q  (SAME query, dog is still dog)
2. Keys from all: k_cat, k_sat, k_dog  (same keys, different order)
3. Attention scores: [q_dog·k_cat, q_dog·k_sat, q_dog·k_dog]
4. Softmax → weights: [0.5, 0.3, 0.2]  (SAME weights, reordered!)
5. Output: 0.5·v_cat + 0.3·v_sat + 0.2·v_dog = SAME as before!

**The output for "dog" is identical**, just now it appears at position 3 instead of position 1.

**This is permutation equivariance**: Reordering inputs → Reordering outputs (same values, different positions).

---

### Part 4: What Does "Breaking Symmetry" Mean?

**Breaking symmetry does NOT mean:**
- Creating "negative" symmetry
- Creating "opposite" behavior
- Creating anti-symmetry

**Breaking symmetry MEANS:**
- The symmetry **no longer holds**
- The property that was true before is now false
- The function can now **distinguish** between different orderings

#### Analogy: Breaking the Symmetry of a Circle

A circle has rotational symmetry - rotate it any amount, it looks the same.

**How to break this symmetry**: Put a dot on the circle.
```
Before (symmetric):     After (broken symmetry):
      ○                       ○
                              •

Rotate 90°:             Rotate 90°:
      ○                       •
                            ○
   (looks same)          (looks DIFFERENT!)
```

The dot **breaks** the symmetry because now rotations produce visibly different results.

**Breaking ≠ Negating. Breaking = Removing the property.**

---

### Part 5: Why Pure Permutation Equivariance is a Problem for Language

If attention is permutation equivariant, then:
```
"dog bites man"  and  "man bites dog"
```
Would produce the SAME outputs (just reordered)!

But these sentences have completely different meanings:
- "dog bites man" → dog is the biter, man is bitten
- "man bites dog" → man is the biter, dog is bitten

**We NEED the model to distinguish word order.** Pure permutation equivariance destroys this information.

---

### Part 6: Examples of Functions That Are NOT Permutation Equivariant

#### Example 1: "Return the first element"

```python
def first(x):
    return x[0]
```

Test:
```
Input:           [A, B, C]  →  first = A
Permuted input:  [C, A, B]  →  first = C

Is f(permute(x)) = permute(f(x))?
f(permute(x)) = C
permute(f(x)) = permute(A) = A  (single element, permutation is identity)

C ≠ A  ✗ NOT equivariant!
```

The function "get first element" can **distinguish** between different orderings.

#### Example 2: "Weighted sum with position-dependent weights"

```python
def weighted_sum(x):
    # Position 1 gets weight 0.5, position 2 gets 0.3, position 3 gets 0.2
    return 0.5*x[0] + 0.3*x[1] + 0.2*x[2]
```

Test:
```
Input:           [10, 20, 30]
Output:          0.5*10 + 0.3*20 + 0.2*30 = 5 + 6 + 6 = 17

Permuted input:  [30, 10, 20]
Output:          0.5*30 + 0.3*10 + 0.2*20 = 15 + 3 + 4 = 22

17 ≠ 22  ✗ Different outputs for permuted inputs!
```

Position-dependent weights make the function **not** permutation equivariant.

#### Example 3: RNNs

An RNN processes left-to-right:
```
h₁ = f(x₁, h₀)
h₂ = f(x₂, h₁)
h₃ = f(x₃, h₂)
```

For input [A, B, C]:
- h₃ depends on seeing A first, then B, then C

For input [C, A, B]:
- h₃ depends on seeing C first, then A, then B
- This creates a DIFFERENT h₃!

**RNNs are NOT permutation equivariant** because order of processing matters.

---

### Part 7: How Positional Encoding Breaks Permutation Equivariance

#### Without Positional Encoding

```
Input words:    [dog, cat, sat]
Embeddings:     [e_dog, e_cat, e_sat]

Permuted words: [cat, sat, dog]
Embeddings:     [e_cat, e_sat, e_dog]
```

The embeddings are just reordered. Attention (being equivariant) produces reordered outputs. The model cannot tell which word came first.

#### With Positional Encoding

```
Input words:    [dog, cat, sat]
Position codes: [PE₁, PE₂, PE₃]
Final input:    [e_dog + PE₁, e_cat + PE₂, e_sat + PE₃]

Permuted words: [cat, sat, dog]
Position codes: [PE₁, PE₂, PE₃]  ← SAME positions!
Final input:    [e_cat + PE₁, e_sat + PE₂, e_dog + PE₃]
```

**Now compare the inputs for "dog":**
- Original: e_dog + PE₁ (dog at position 1)
- Permuted: e_dog + PE₃ (dog at position 3)

**These are DIFFERENT vectors!**

```
e_dog + PE₁ ≠ e_dog + PE₃
```

Since the inputs are different, the attention outputs will be different. The model can now **distinguish** where each word appeared.

#### Worked Example

Let's use simple numbers:
```
e_dog = [1, 0]
e_cat = [0, 1]
PE₁ = [0.1, 0]
PE₂ = [0.2, 0]
```

Original input [dog at pos 1, cat at pos 2]:
```
dog: [1, 0] + [0.1, 0] = [1.1, 0]
cat: [0, 1] + [0.2, 0] = [0.2, 1]
```

Permuted input [cat at pos 1, dog at pos 2]:
```
cat: [0, 1] + [0.1, 0] = [0.1, 1]   ← Different from [0.2, 1]!
dog: [1, 0] + [0.2, 0] = [1.2, 0]   ← Different from [1.1, 0]!
```

**The actual vectors fed into attention are different**, so the outputs will be different. Order now matters.

---

### Part 8: What the Model Can Now Do

**Without positional encoding**, the model could answer:
- "Is there a dog in this sentence?" ✓
- "Is there a cat in this sentence?" ✓
- "What words are in this sentence?" ✓

**But it could NOT answer:**
- "What is the first word?" ✗
- "What word comes after 'the'?" ✗
- "Is this 'dog bites man' or 'man bites dog'?" ✗

**With positional encoding**, the model CAN distinguish order:
- "dog at position 1" is a different input than "dog at position 3"
- The model can learn that position matters for meaning

---

### Part 9: Permutation Equivariance vs. Other Symmetries

| Symmetry Type | What it means | Example |
|---------------|---------------|---------|
| **Translation equivariant** | Shift input → shift output | CNNs on images |
| **Rotation equivariant** | Rotate input → rotate output | Some specialized vision models |
| **Permutation equivariant** | Reorder input → reorder output | Set functions, vanilla attention |
| **Permutation invariant** | Reorder input → same output | Sum, max, mean over sets |

**Self-attention without positions**: Permutation equivariant
**Self-attention with positions**: NOT permutation equivariant (can distinguish order)

---

### Part 10: Summary - Why This Design?

The Transformer's design is intentional:

1. **Start with maximum flexibility**: Attention can look at anything (complete graph)

2. **Inject minimal structure**: Add positional encoding to distinguish order

3. **Let the model learn the rest**: Which positions actually matter for each task

This is different from RNNs which **bake in** sequential processing. The Transformer **adds** position information as a feature, keeping the architecture flexible.

```
RNN approach:    Process sequentially (order is in the algorithm)
Transformer:     Process in parallel + add position as data (order is in the input)
```

---

### The Architecture as a Graph Neural Network

From a GDL perspective, the Transformer can be viewed as a special case of a **Graph Attention Network (GAT)** where:
- The graph is **complete** (all-to-all connections)
- Edge weights are **computed dynamically** from node features
- Multiple **parallel graphs** exist (multi-head)

```
Standard GNN:  Fixed graph adjacency A
Transformer:   Learned A = softmax(QK^T/√d_k)
```

---

## Summary Table

| Component | Input → Output | Key Insight |
|-----------|---------------|-------------|
| Input Embedding | Tokens → d_model vectors | Discrete → continuous |
| Positional Encoding | Add position info | Break permutation symmetry |
| Encoder | Sequence → Context | Build all-to-all relationships |
| Decoder | Context + Prefix → Next token | Autoregressive generation |
| Output Linear + Softmax | d_model → Vocabulary | Probability distribution |
