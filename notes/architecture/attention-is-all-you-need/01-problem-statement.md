# The Problem: Sequential Bottleneck in Sequence Modeling

## The Core Issue

Before Transformers, sequence-to-sequence models (like machine translation) relied on:

1. **Recurrent Neural Networks (RNNs/LSTMs/GRUs)**
2. **Convolutional Neural Networks (CNNs)**

Both have fundamental limitations for learning long-range dependencies.

---

## Problem Abstraction Ladder

### L1: Abstract Principle
> **Learning relationships between distant elements in a sequence should not require traversing all intermediate elements.**

This is the fundamental insight. If word 1 relates to word 100, why should information have to flow through words 2-99?

### L2: Framework Level
The bottleneck manifests in two ways:
- **Computational**: Sequential processing prevents parallelization
- **Learning**: Long paths make gradient flow difficult (vanishing/exploding gradients)

### L3: Method Level - Comparing Approaches

| Architecture | Path Length | Sequential Ops | Parallelizable |
|-------------|-------------|----------------|----------------|
| RNN/LSTM | O(n) | O(n) | No |
| CNN | O(log_k(n)) | O(1) | Yes |
| **Self-Attention** | **O(1)** | **O(1)** | **Yes** |

### L4: Concrete Impact
- RNN: To connect position 1 to position n, information flows through n hidden states
- CNN: Requires O(n/k) layers with kernel size k, or O(log_k(n)) with dilated convolutions
- Self-Attention: Every position directly attends to every other position in one operation

### L5: Specific Example
Consider the sentence: "The cat, which was sitting on the mat that my grandmother bought, **was** sleeping."

The verb "was" must agree with "cat" (singular), not "mat" or "grandmother".

- **RNN**: Signal must propagate through ~12 time steps
- **Transformer**: Direct attention from "was" → "cat" in O(1)

---

## Geometric Deep Learning Perspective (First Principles)

This section builds up the key concepts from scratch. If you understand these, you'll have deep intuition for why Transformers work.

---

### Part 1: What is a Symmetry?

A **symmetry** is a transformation you can apply to something that leaves it "essentially unchanged."

#### Concrete Examples of Symmetries

**Example 1: A Square**
```
    A --- B
    |     |
    D --- C
```

What transformations leave this square "the same"?
- Rotate 90°, 180°, 270°, 360° (back to start)
- Flip horizontally, vertically, or diagonally

After any of these, you still have a square in the same position - it looks identical. These are the **symmetries of a square**.

**Example 2: A Circle**

A circle has infinitely many symmetries:
- Rotate by ANY angle (1°, 45°, 89.7°, anything)
- Flip across any diameter

**Example 3: The Letter "A"**
```
    /\
   /  \
  /----\
 /      \
```
Only ONE symmetry: flip across the vertical axis. Rotating it would make it look different.

---

### Part 2: What is a Symmetry Group?

A **symmetry group** is the complete set of all symmetries of an object, along with rules for combining them.

#### The Symmetry Group of a Square

The square has 8 symmetries:
1. Do nothing (identity)
2. Rotate 90° clockwise
3. Rotate 180°
4. Rotate 270° clockwise
5. Flip horizontally
6. Flip vertically
7. Flip along main diagonal
8. Flip along other diagonal

This is called the **dihedral group D₄**.

**Key property**: If you apply one symmetry, then another, you get a third symmetry that's also in the group.
- Example: Rotate 90° + Rotate 90° = Rotate 180° ✓ (still in the group)

---

### Part 3: The Symmetry Group of a Chain (Sequence)

Now consider a **chain** or **sequence**:

```
Position:    1      2      3      4      5
             ●------●------●------●------●
```

What are the symmetries of this chain?

**Translation (Shift)**: Move everything one position to the right:
```
Before:  [A, B, C, D, E]
After:   [?, A, B, C, D]  (E falls off, ? is unknown)
```

This is the main symmetry of a chain: **things that are true at position k should also be true at position k+1** (with appropriate adjustments).

**The Chain's Symmetry Group** consists of:
- Shift by 1 position
- Shift by 2 positions
- Shift by n positions
- etc.

This is called the **translation group** along the chain.

---

### Part 4: What is Equivariance?

**Equivariance** means: if you transform the input, the output transforms in a corresponding way.

#### Formal Definition
A function f is **equivariant** to a transformation T if:
```
f(T(input)) = T(f(input))
```

In words: Transform then process = Process then transform.

#### Concrete Example: Image Classification

**Invariance** (a special case of equivariance):
- If you shift a photo of a cat 10 pixels to the right, it's still a cat
- The label "cat" doesn't change
- Classification should be **invariant** to translation

**Equivariance**:
- If you shift an image, then apply edge detection, you get shifted edges
- If you apply edge detection, then shift, you also get shifted edges
- Edge detection is **equivariant** to translation

```
Original:        Shifted Right:
  ████              ████
  █  █              █  █
  ████              ████

Edge detection gives you the same edges, just shifted.
```

---

### Part 5: How RNNs Are Equivariant to the Chain's Symmetry

An RNN processes sequences like this:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ...
      ↑            ↑            ↑
     x₁           x₂           x₃
```

The SAME RNN function is applied at every position. This is equivariance to translation along the chain:

**If you shift the input sequence by one position, the hidden states shift by one position.**

```
Input:       [A, B, C, D]
Hidden:      [h₁, h₂, h₃, h₄]

Shifted input:  [B, C, D, E]
Shifted hidden: [h'₁, h'₂, h'₃, h'₄]

If we had seen [B, C, D] before, h'₁ = what h₂ was.
```

**The RNN "respects" the chain structure.** It assumes that:
- Position 1 connects to position 2
- Position 2 connects to position 3
- There's no direct connection from position 1 to position 5

---

### Part 6: What is an Inductive Bias?

An **inductive bias** is an assumption built into a model about what kinds of patterns it should look for.

#### Why Do We Need Inductive Biases?

Imagine you have 100 data points and you want to learn a function. There are infinitely many functions that pass through those 100 points! How do you choose?

**Inductive biases** constrain the search:
- "The function should be smooth" (not wildly oscillating)
- "Nearby inputs should have similar outputs"
- "The function should be simple" (Occam's razor)

Without inductive biases, learning is impossible (you can't generalize from finite data to infinite possibilities).

#### Examples of Inductive Biases in Neural Networks

| Architecture | Inductive Bias |
|--------------|----------------|
| Fully Connected | None specific (very flexible, needs lots of data) |
| CNN | Local patterns matter, translation equivariance |
| RNN | Sequential order matters, recent past is most relevant |
| Graph Neural Network | Data has graph structure, neighbors influence each other |

#### The RNN's Inductive Bias

The RNN assumes:
1. **Sequential processing matters**: Information flows left-to-right
2. **Chain connectivity**: Position k only directly connects to position k-1 and k+1
3. **Recency bias**: Recent information is in the hidden state; old information may be forgotten

**This is the "chain topology" inductive bias.**

---

### Part 7: When the Inductive Bias Doesn't Match Reality

#### The Problem: Language Isn't Really a Chain

Consider this sentence:
```
"The cat, which was sitting on the mat, was sleeping."
```

The RNN sees this as a chain:
```
The → cat → which → was → sitting → on → the → mat → was → sleeping
 1     2      3      4       5      6    7     8     9      10
```

But the **true grammatical relationships** are:

```
"The cat" (positions 1-2) ←→ "was sleeping" (positions 9-10)
    ↓
"which was sitting on the mat" (positions 3-8) is a PARENTHETICAL
```

The verb "was sleeping" agrees with "cat" (singular), NOT with "mat" (also singular, but irrelevant grammatically).

**In chain form:**
- Distance from "cat" to "was sleeping" = 7 positions
- The RNN must remember "cat" through 7 steps
- Gradient must flow backward through 7 steps

**In true grammatical form:**
- "cat" and "was sleeping" are DIRECTLY connected
- They're part of the same clause: "The cat was sleeping"
- The stuff in between is a detour

---

### Part 8: What is the "True Relational Structure"?

The **true relational structure** of data is how elements actually relate to each other, independent of how they're presented.

#### Example 1: Sentence Structure

Surface form (chain):
```
Position: 1   2   3   4   5   6   7   8   9   10
Words:    The cat who ate the fish was still hungry
```

True structure (tree):
```
              [was hungry]
               /        \
          [cat]        [still]
          /    \
       [The]  [who ate the fish]
                    |
                  [ate]
                 /     \
              [who]   [the fish]
```

"cat" and "was" are DIRECTLY connected in the tree (parent-child), even though they're 7 positions apart in the chain.

#### Example 2: Cricket

Surface form (chain of deliveries):
```
Ball: 1   2   3   4   5   6   7   8   ... 50   51   52
      ●───●───●───●───●───●───●───●───...───●────●────●
```

True relational structure:
```
Ball 52 relates to:
├── Ball 51 (previous ball) - CLOSE in chain
├── Ball 46 (same bowler's last ball) - FAR in chain
├── Ball 30 (when this batsman came in) - FAR in chain
├── Ball 1 (start of innings, sets context) - VERY FAR in chain
└── Many balls in between are IRRELEVANT
```

The chain assumes all balls connect only to their neighbors. But ball 52 might have almost nothing to do with ball 40, and everything to do with ball 30 (when the batsman arrived).

---

### Part 9: The RNN's Failure Mode

Because the RNN is equivariant to the chain's symmetry group, it's forced to:

1. **Process sequentially**: Can't skip ahead to see what's coming
2. **Compress into hidden state**: All information about positions 1...k must fit in a fixed-size vector hₖ
3. **Struggle with long gaps**: Information from position 1 gets "washed out" by the time we reach position 100

**Concrete failure**:

If "cat" (position 2) and "was sleeping" (position 9) need to match, the RNN must:
- Encode "cat" into h₂
- Carry this information through h₃, h₄, h₅, h₆, h₇, h₈
- Still have it available at h₉

Seven steps is manageable. But what about 100 steps? The hidden state can't perfectly preserve all information for that long.

---

### Part 10: What Transformers Do Instead

**Transformers have a different inductive bias:**

Instead of assuming chain connectivity, they assume:
1. **Any position might relate to any other position**
2. **The relevance should be learned from data**
3. **Multiple types of relationships can coexist** (multi-head attention)

#### Attention Creates a Learned Graph

For the sentence "The cat was sleeping":

RNN sees:
```
The → cat → was → sleeping  (fixed chain)
```

Transformer learns:
```
The ──→ cat ←──────────→ was ←── sleeping
 │       ↑                ↑         │
 └───────┴────────────────┴─────────┘
         (all-to-all, weighted by relevance)
```

The attention weights LEARN that "cat" and "was" should connect strongly, even though they're not adjacent.

#### The Key Insight

**Self-attention creates a complete graph with learned, data-dependent edge weights.**

```
Complete graph (all-to-all connections):

    x₁ ─────── x₂
    │ \     / │
    │  \   /  │
    │   \ /   │
    │    ╳    │
    │   / \   │
    │  /   \  │
    │ /     \ │
    x₃ ─────── x₄
```

Every position can attend to every other position. The attention weights (learned) determine how strongly each connection matters.

**The attention weights act as a soft adjacency matrix:**

|          | The | cat | was | sleeping |
|----------|-----|-----|-----|----------|
| **The**  | 0.1 | 0.8 | 0.05| 0.05     |
| **cat**  | 0.1 | 0.1 | 0.6 | 0.2      |
| **was**  | 0.05| 0.7 | 0.1 | 0.15     |
| **sleeping** | 0.05 | 0.3 | 0.4 | 0.25 |

Each row sums to 1. "was" attends strongly (0.7) to "cat" even though they're not adjacent.

---

### Summary: Chain vs. Learned Graph

| Aspect | RNN (Chain Bias) | Transformer (Learned Graph) |
|--------|------------------|----------------------------|
| Connectivity | Fixed: only neighbors | Learned: any-to-any |
| Symmetry | Translation along chain | Permutation (with position encoding) |
| Inductive bias | "Sequence order is everything" | "Let data determine connections" |
| Long-range | Hard (O(n) path) | Easy (O(1) path) |
| Failure mode | Forgets distant past | Needs more data to learn patterns |

---

## Why This Matters for Cricket Ball-by-Ball Prediction

In cricket:
- Ball 1 of an over might relate strongly to ball 6 (completing the over)
- A dot ball early in an innings affects strategy 50 balls later
- Batsman-bowler matchups create long-range dependencies

A model that can learn **which past events matter for predicting the next ball** - without being constrained by sequential distance - has significant advantages.

---

## Summary

| Aspect | RNN Problem | Transformer Solution |
|--------|------------|---------------------|
| Long-range deps | Hard to learn (path = O(n)) | Easy (path = O(1)) |
| Parallelization | Sequential by nature | Fully parallel |
| Inductive bias | Rigid chain structure | Flexible, learned graph |
| Gradient flow | Vanishing/exploding | Direct paths everywhere |
