# Attention Is All You Need - Study Notes

**Paper**: Vaswani et al. (2017) - "Attention Is All You Need"
**arXiv**: 1706.03762v7
**Conference**: NeurIPS 2017

## Overview

This paper introduces the **Transformer** architecture - the first sequence transduction model based entirely on attention mechanisms, eliminating recurrence and convolutions. It has become the foundation for modern language models (BERT, GPT, etc.) and is highly relevant for sequential prediction tasks like cricket ball-by-ball modeling.

## Notes Structure

| File | Topic |
|------|-------|
| [01-problem-statement.md](./01-problem-statement.md) | The conceptual problem being solved |
| [02-architecture-overview.md](./02-architecture-overview.md) | High-level architecture and abstraction ladder |
| [03-attention-mechanism.md](./03-attention-mechanism.md) | Deep dive into attention with geometric DL lens |
| [04-manifold-transformations.md](./04-manifold-transformations.md) | What happens to the data manifold during learning |
| [05-cricket-application.md](./05-cricket-application.md) | Relevance to ball-by-ball prediction |

## Key Takeaways (Preview)

1. **Self-attention** enables O(1) path length between any two sequence positions (vs O(n) for RNNs)
2. **Multi-head attention** learns multiple "views" of relationships in parallel
3. **Positional encodings** inject sequence order without requiring sequential computation
4. From a geometric DL perspective: attention creates a **dynamic, data-dependent graph** over the sequence

## Abstraction Ladder Summary

```
L1 (Abstract):   Learn relationships between any elements regardless of distance
L2 (Framework):  Use attention as a soft, learnable adjacency matrix
L3 (Method):     Query-Key-Value mechanism with scaled dot-product
L4 (Component):  Multi-head attention + FFN + residuals + layer norm
L5 (Concrete):   Attention(Q,K,V) = softmax(QK^T/√d_k)V
```
