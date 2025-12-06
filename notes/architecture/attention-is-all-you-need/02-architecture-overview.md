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

## Geometric Deep Learning View

### Symmetry and Equivariance

The Transformer's attention mechanism is:
- **Permutation equivariant** (without positional encoding): Reordering inputs reorders outputs identically
- **Position-aware** (with positional encoding): Positional encodings break pure permutation symmetry

This is a design choice: we inject just enough structure (position) while maintaining flexibility.

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
