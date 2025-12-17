# V2 Architecture: Unified Heterogeneous Graph

## Executive Summary

This document describes the second version of the cricket ball prediction model. The key innovation is representing **all information as a single heterogeneous graph**, eliminating the separation between spatial and temporal processing.

### What Changed from V1

| Aspect | V1 (Dual Stream) | V2 (Unified Graph) |
|--------|------------------|-------------------|
| Architecture | HierarchicalGAT + TemporalTransformer | Single HeteroGNN |
| Temporal handling | Sequence attention (O(n²)) | Graph edges (O(n)) |
| History length | Fixed 24 balls | Full innings |
| Hierarchy | Sequential layer processing | Edge-based conditioning |
| Same-bowler patterns | Soft attention bias | Explicit edge connections |
| Framework | PyTorch + partial PyG | Full PyTorch Geometric |

### Why This is Better

1. **Full History**: Graph attention scales linearly with edges, not quadratically with sequence length
2. **Explicit Structure**: Same-bowler/same-batsman relationships are edges, not soft biases
3. **Unified Processing**: One message-passing framework instead of two parallel streams
4. **Interpretability**: Edge attention weights show exactly how information flows
5. **PyG Native**: Leverage optimized sparse operations, batching, and sampling

---

## The Core Idea

### Everything is a Node

Instead of treating "current state" and "ball history" differently, we represent everything as nodes in one graph:

```
MATCH GRAPH
├── Context Nodes (21 total)
│   ├── Global (3): venue, batting_team, bowling_team
│   ├── State (5): score, chase, phase, time, wickets
│   ├── Actor (7): striker_id, striker_state, nonstriker_id, nonstriker_state,
│   │              bowler_id, bowler_state, partnership
│   └── Dynamics (4): batting_momentum, bowling_momentum, pressure, dots
│
├── Ball Nodes (N balls, entire innings history)
│   └── Each ball: 15 features (runs, wicket type, over, extras, etc.)
│   └── Player embeddings: who bowled, who faced, who partnered
│
└── Query Node (1 node)
    └── "What happens on the next ball?"
```

### Geometric Deep Learning Principles

The architecture respects key symmetries in cricket data:

1. **Temporal Ordering**: `precedes` edges with TransformerConv + temporal distance edge features
2. **Correct Player Attribution**: Cross-domain edges connect balls to the players who ACTUALLY faced/bowled them (not current players)
3. **Z2 Striker/Non-striker Symmetry**: Partnership dynamics and `partnered_by` edges capture the relationship
4. **Set Membership**: `same_bowler`/`same_batsman`/`same_matchup` create symmetric cliques

### Relationships are Edges

The graph structure encodes all relationships explicitly:

```
EDGE TYPES
├── Hierarchical Conditioning
│   ├── (global) --[conditions]--> (state)
│   ├── (state) --[conditions]--> (actor)
│   └── (actor) --[conditions]--> (dynamics)
│
├── Intra-Layer Relationships
│   ├── (global) --[relates_to]--> (global)
│   ├── (state) --[relates_to]--> (state)
│   ├── (actor) --[matchup]--> (actor)      # striker↔bowler, id↔state
│   └── (dynamics) --[relates_to]--> (dynamics)
│
├── Temporal Structure
│   ├── (ball) --[precedes]--> (ball)       # Causal ordering
│   ├── (ball) --[same_bowler]--> (ball)    # Same bowler delivered both
│   └── (ball) --[same_batsman]--> (ball)   # Same batsman faced both
│
├── Cross-Domain Links
│   ├── (ball) --[bowled_by]--> (actor)     # Ball → bowler identity
│   ├── (ball) --[faced_by]--> (actor)      # Ball → striker identity
│   └── (dynamics) --[reflects]--> (ball)   # Recent balls → momentum
│
└── Prediction
    └── (query) --[attends_to]--> (all)     # Query reads everything
```

---

## Why Full History Now Works

### The Quadratic Problem (V1)

In V1's TemporalTransformer, every ball attends to every other ball:
- 24 balls → 24 × 24 = 576 attention computations
- 120 balls → 120 × 120 = 14,400 attention computations
- **Scales O(n²)** - prohibitive for full innings

### The Linear Solution (V2)

In V2's graph, attention only flows along edges:
- Same-bowler edges: ~6 bowlers × ~20 balls each = ~120 edges
- Same-batsman edges: ~5 batsmen × ~25 balls each = ~125 edges
- Temporal edges: ~120 balls × 1 (previous) = ~120 edges
- **Scales O(n)** - full innings is feasible!

### Concrete Example

For a 120-ball innings:
- V1 Transformer: 14,400 attention pairs
- V2 Graph: ~500 edges (same-bowler + same-batsman + temporal)
- **30x reduction in computation**

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CRICKET HETERO-GNN                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: HeteroData object with all nodes and edges                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. Node Feature Encoding                                         │   │
│  │    - Learned embeddings for entities (venue, team, player)       │   │
│  │    - Linear projections for numeric features                     │   │
│  │    - All nodes → hidden_dim (128)                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 2. Heterogeneous Message Passing (× N layers)                    │   │
│  │    - HeteroConv with edge-type-specific convolutions             │   │
│  │    - Each edge type can use different attention mechanism        │   │
│  │    - Information flows: context↔context, balls↔balls,            │   │
│  │      context↔balls, everything→query                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 3. Readout                                                       │   │
│  │    - Query node has aggregated all information                   │   │
│  │    - MLP: hidden_dim → 7 classes                                 │   │
│  │    - Softmax → probabilities                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: [P(dot), P(1), P(2), P(3), P(4), P(6), P(wicket)]             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Document Structure

1. **01-overview.md** (this document) - High-level architecture
2. **02-node-types.md** - Detailed node type specifications
3. **03-edge-types.md** - Edge type specifications and semantics
4. **04-model-architecture.md** - HeteroGNN implementation details
5. **05-data-pipeline.md** - Dataset and DataLoader design
6. **06-training.md** - Training procedure and loss functions
