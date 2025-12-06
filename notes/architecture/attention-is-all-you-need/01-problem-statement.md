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

## Geometric Deep Learning Perspective

### The Sequence as a Manifold

Sequences can be viewed as data living on a **1D chain graph**:

```
x₁ --- x₂ --- x₃ --- ... --- xₙ
```

**RNNs** respect this chain structure rigidly - they are **equivariant to the chain's symmetry group** (translation along the sequence, but only in one direction).

**Problem**: The chain topology is an **inductive bias** that may not match the true relational structure of the data.

### What We Actually Want

The true relationships in language (or cricket sequences) form a **much richer graph**:

```
     x₁ -------- x₅
    /  \        /
   x₂   x₃ --- x₄
          \   /
           x₆
```

Different words relate to each other based on semantics, syntax, and context - not just proximity.

### The Transformer's Solution

**Self-attention creates a complete graph with learned, data-dependent edge weights.**

```
Every xᵢ <--attention--> Every xⱼ
```

The attention weights act as a **soft adjacency matrix** that the model learns to construct based on the input itself.

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
