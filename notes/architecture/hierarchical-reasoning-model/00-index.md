# Hierarchical Reasoning Model - Study Notes

**Paper**: Wang et al. (2025) - "Hierarchical Reasoning Model"
**arXiv**: 2506.21734v3
**Organization**: Sapient Intelligence, Singapore

## Overview

This paper introduces the **Hierarchical Reasoning Model (HRM)** - a brain-inspired recurrent architecture that achieves significant computational depth while maintaining training stability and efficiency. Unlike Transformers (which are "paradoxically shallow" despite being called "deep learning"), HRM can perform extensive iterative reasoning through two coupled recurrent modules operating at different timescales.

**Key Achievement**: With only **27M parameters** and **~1000 training examples**, HRM solves complex reasoning tasks (Sudoku-Extreme, 30x30 Mazes, ARC-AGI) that defeat even the largest CoT models like o3-mini-high.

## Notes Structure

| File | Topic |
|------|-------|
| [01-problem-statement.md](./01-problem-statement.md) | Why Transformers fail at deep reasoning |
| [02-architecture-overview.md](./02-architecture-overview.md) | H-module, L-module, and their interaction |
| [03-hierarchical-convergence.md](./03-hierarchical-convergence.md) | The key innovation: avoiding premature convergence |
| [04-manifold-transformations.md](./04-manifold-transformations.md) | Geometric DL perspective and dimensionality hierarchy |
| [05-cricket-application.md](./05-cricket-application.md) | Relevance to ball-by-ball prediction |

## Key Takeaways (Preview)

1. **Computational Depth Problem**: Transformers are fixed-depth (in complexity classes AC⁰ or TC⁰), preventing them from solving problems requiring polynomial time
2. **Chain-of-Thought is a Crutch**: CoT externalizes reasoning to tokens, but is brittle, slow, and data-hungry
3. **Hierarchical Convergence**: HRM's two-timescale design prevents the premature convergence that plagues standard RNNs
4. **O(1) Memory Training**: One-step gradient approximation avoids expensive BPTT while remaining biologically plausible
5. **Brain Correspondence**: The trained model exhibits dimensionality hierarchy similar to mouse cortex

## Abstraction Ladder Summary

```
L1 (Abstract):   Enable arbitrarily deep reasoning without sequential bottlenecks
L2 (Framework):  Two recurrent modules at different timescales (slow planning + fast execution)
L3 (Method):     H-module guides L-module, L-module converges then resets for new context
L4 (Component):  Encoder-only Transformers as modules + deep supervision + ACT halting
L5 (Concrete):   z_L^i = f_L(z_L^{i-1}, z_H^{i-1}, x̃; θ_L), z_H updates every T steps
```

## Comparison: Transformer vs HRM

| Aspect | Transformer | HRM |
|--------|-------------|-----|
| Depth | Fixed (# layers) | Adaptive (N×T×M segments) |
| Complexity Class | AC⁰/TC⁰ | Turing-complete |
| Long reasoning | Requires CoT tokens | Internal latent space |
| Training | Standard backprop | One-step gradient (O(1) memory) |
| Data efficiency | Needs massive data | ~1000 examples |
| Biological plausibility | Low | High (cortical hierarchy) |

## Brain Inspiration

HRM is inspired by three principles from neuroscience:

1. **Hierarchical Processing**: Higher cortical areas integrate over longer timescales
2. **Temporal Separation**: Different brain rhythms (theta 4-8Hz, gamma 30-100Hz) for different functions
3. **Recurrent Connectivity**: Feedback loops refine representations without requiring BPTT
