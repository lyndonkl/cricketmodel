# Graph Attention Networks (GAT) - Study Notes

**Paper**: Veličković et al. (2018) - "Graph Attention Networks"
**Published**: ICLR 2018
**arXiv**: 1710.10903v3
**Authors**: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio

## Overview

This paper introduces **Graph Attention Networks (GATs)** - neural network architectures that operate on graph-structured data using masked self-attention. Unlike prior graph convolution methods that use fixed aggregation weights based on graph structure, GAT learns to weight neighbors based on their features, enabling different importance to be assigned to different nodes in a neighborhood.

**Key Achievement**: GAT achieves state-of-the-art results on citation network benchmarks (Cora, Citeseer, Pubmed) and protein-protein interaction datasets, while being computationally efficient, inductive (works on unseen graphs), and interpretable (attention weights show what matters).

## Notes Structure

| File | Topic |
|------|-------|
| [01-problem-statement.md](./01-problem-statement.md) | Why graphs are hard, spectral vs non-spectral approaches |
| [02-architecture-overview.md](./02-architecture-overview.md) | GAT layer, attention mechanism, multi-head attention |
| [03-manifold-transformations.md](./03-manifold-transformations.md) | Geometric view: isotropic vs anisotropic smoothing |
| [04-cricket-application.md](./04-cricket-application.md) | Modeling cricket as a graph, potential applications |

## Key Takeaways (Preview)

1. **Learned Neighbor Weights**: GAT computes attention coefficients that determine how much each neighbor contributes - learned from features, not fixed by structure
2. **Masked Self-Attention**: Attention is computed only over neighbors (not all nodes), injecting graph structure into the mechanism
3. **Multi-Head Stability**: Multiple independent attention heads stabilize learning, similar to Transformers
4. **Inductive Capability**: Shared attention mechanism generalizes to unseen graphs at test time
5. **Interpretability**: Attention weights can be inspected to understand model decisions

## Abstraction Ladder Summary

```
L1 (Abstract):   Learn which relationships matter based on content, not just structure
L2 (Framework):  Self-attention over graph neighborhoods with learnable importance weights
L3 (Method):     Compute attention scores, normalize with softmax, aggregate weighted neighbors
L4 (Component):  e_ij = LeakyReLU(a^T[Wh_i || Wh_j]), α_ij = softmax(e_ij)
L5 (Concrete):   h'_i = σ(Σ_{j∈N_i} α_ij · W·h_j), with K=8 heads, F'=8 features each
```

## Comparison: GCN vs GAT

| Aspect | GCN | GAT |
|--------|-----|-----|
| Neighbor weights | Fixed (degree-based) | Learned (content-based) |
| Aggregation | Isotropic (uniform smoothing) | Anisotropic (selective) |
| Inductive | No (needs full graph) | Yes (shared mechanism) |
| Interpretability | Low | High (attention weights) |
| Complexity | O(\|V\|FF' + \|E\|F') | O(\|V\|FF' + \|E\|F') - same! |
| Model capacity | Limited | Higher (different neighbor importance) |

## Comparison: Transformer vs GAT

| Aspect | Transformer | GAT |
|--------|-------------|-----|
| Input structure | Sequence | Graph |
| Attention scope | All-to-all | Neighbors only (masked) |
| Position encoding | Explicit (sinusoidal/learned) | Implicit (graph structure) |
| Computational cost | O(n²) in sequence length | O(\|E\|) in edges |
| Natural for | Text, time series | Social networks, molecules |

## The Core Innovation

**Before GAT** (GCN and spectral methods):
```
h'_i = σ(Σ_j (1/√(d_i·d_j)) · W·h_j)
            ↑
    Fixed weight based on node degrees

All neighbors treated equally regardless of content
```

**GAT**:
```
h'_i = σ(Σ_j α_ij · W·h_j)
            ↑
    Learned weight based on FEATURES

Model learns which neighbors matter for each node
```

## Why This Matters for Cricket

GAT's strengths map to potential cricket applications:

1. **Relational modeling**: Cricket has natural graph structures (player matchups, team networks)
2. **Learned importance**: Not all relationships equally important (some bowler-batsman matchups matter more)
3. **Interpretability**: Can see which relationships the model considers important
4. **Inductive**: Can generalize to new players not seen during training

However, cricket is primarily **temporal/sequential**, while GAT is designed for **relational/graph** data. Likely best used in combination with temporal models.
