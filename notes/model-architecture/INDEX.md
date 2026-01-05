# CricketHeteroGNN Model Architecture Documentation

## Overview

This documentation provides a comprehensive guide to the CricketHeteroGNN model architecture, a heterogeneous graph neural network designed for cricket ball-by-ball outcome prediction.

**Task**: Predict the outcome of the next ball in a T20 cricket match
**Output Classes**: Dot (0), Single (1), Two (2), Three (3), Four (4), Six (5), Wicket (6)

## Architecture at a Glance

```
                    ┌─────────────────────────────────────────────────┐
                    │              CricketHeteroGNN                   │
                    └─────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              ┌─────▼─────┐       ┌───────▼───────┐     ┌───────▼───────┐
              │  Encoders │       │  Conv Stack   │     │   Readout &   │
              │           │       │  (Message     │     │   Prediction  │
              │ - Entity  │──────▶│   Passing)    │────▶│               │
              │ - Feature │       │               │     │ - Query Node  │
              │ - Ball    │       │ - HeteroConv  │     │ - Matchup MLP │
              │ - Query   │       │ - FiLM Cond.  │     │ - MLP Head    │
              └───────────┘       └───────────────┘     └───────────────┘
```

## The Hierarchical Graph Structure

The model operates on a 6-layer heterogeneous graph:

```
Layer 1: GLOBAL      [venue] [batting_team] [bowling_team]
              │              │              │
              └──────────────┼──────────────┘
                             │ conditions
                             ▼
Layer 2: STATE       [score] [chase] [phase] [time_pressure] [wicket_buffer]
              │       │       │         │           │
              └───────┴───────┼─────────┴───────────┘
                              │ conditions
                              ▼
Layer 3: ACTOR       [striker_id] [striker_state] [nonstriker_id] [nonstriker_state]
                     [bowler_id]  [bowler_state]  [partnership]
              │              │              │
              └──────────────┼──────────────┘
                             │ conditions
                             ▼
Layer 4: DYNAMICS    [batting_momentum] [bowling_momentum] [pressure_index] [dot_pressure]
              │              │              │              │
              └──────────────┼──────────────┴──────────────┘
                             │ informs (from balls)
                             ▼
Layer 5: BALL        [ball_0] [ball_1] ... [ball_n-1]  (historical deliveries)
              │              │              │
              └──────────────┼──────────────┘
                             │ attends
                             ▼
Layer 6: QUERY       [query]  ──────────────────────▶  PREDICTION
```

## Documentation Files

### Core Architecture
| File | Description |
|------|-------------|
| [01-architecture-overview.md](./01-architecture-overview.md) | High-level architecture, design philosophy, model variants |
| [02-graph-structure.md](./02-graph-structure.md) | Node types (21), edge types (~150), hierarchical layers |
| [09-forward-pass-walkthrough.md](./09-forward-pass-walkthrough.md) | Step-by-step data flow through the model |

### Components
| File | Description |
|------|-------------|
| [03-encoders.md](./03-encoders.md) | EntityEncoder, HierarchicalPlayerEncoder, FeatureEncoder, BallEncoder, QueryEncoder |
| [04-conv-layers.md](./04-conv-layers.md) | Per-edge convolution operators: GATv2Conv, TransformerConv, SAGEConv |
| [05-message-passing.md](./05-message-passing.md) | HeteroConvBlock, PhaseModulatedConvBlock, FiLM conditioning |
| [06-readout-and-prediction.md](./06-readout-and-prediction.md) | Query aggregation, hybrid readout, innings-conditional heads |

### Reference
| File | Description |
|------|-------------|
| [07-model-variants.md](./07-model-variants.md) | All 6 model variants: base, pooling, hybrid, phase-modulated, innings-conditional, full |
| [08-features-reference.md](./08-features-reference.md) | Complete feature dimensions, meanings, and computation |

## Reading Order

**For a complete understanding**, read in this order:
1. `01-architecture-overview.md` - Get the big picture
2. `02-graph-structure.md` - Understand the data representation
3. `08-features-reference.md` - Know what features exist
4. `03-encoders.md` - How raw features become embeddings
5. `04-conv-layers.md` - Why specific operators for specific edges
6. `05-message-passing.md` - How information flows
7. `06-readout-and-prediction.md` - How predictions are made
8. `07-model-variants.md` - Different model configurations
9. `09-forward-pass-walkthrough.md` - Tie it all together

**For quick reference**, jump directly to:
- `08-features-reference.md` for feature dimensions
- `07-model-variants.md` for choosing a model variant
- `04-conv-layers.md` for understanding edge-specific operators

## Key Source Files

| File | Purpose |
|------|---------|
| `src/model/hetero_gnn.py` | Main model classes (6 variants) |
| `src/model/encoders.py` | Node encoders (5 types) |
| `src/model/conv_builder.py` | Conv operators and FiLM |
| `src/data/edge_builder.py` | Graph structure definition |
| `src/data/feature_utils.py` | Feature computation |
| `src/data/entity_mapper.py` | Entity ID mapping |
| `src/data/hetero_data_builder.py` | HeteroData construction |

## Quick Reference

### Model Configuration (ModelConfig)

```python
@dataclass
class ModelConfig:
    # Entity counts (from EntityMapper)
    num_venues: int
    num_teams: int
    num_players: int
    
    # Architecture
    hidden_dim: int = 128        # Common embedding dimension
    num_layers: int = 3          # Message passing layers
    num_heads: int = 4           # Attention heads
    
    # Embedding dimensions
    venue_embed_dim: int = 32
    team_embed_dim: int = 32
    player_embed_dim: int = 64
    role_embed_dim: int = 16     # For hierarchical player encoder
    
    # Output
    num_classes: int = 7         # Dot, Single, Two, Three, Four, Six, Wicket
```

### Node Types (21 total)

| Layer | Node Types |
|-------|------------|
| Global | venue, batting_team, bowling_team |
| State | score_state, chase_state, phase_state, time_pressure, wicket_buffer |
| Actor | striker_identity, striker_state, nonstriker_identity, nonstriker_state, bowler_identity, bowler_state, partnership |
| Dynamics | batting_momentum, bowling_momentum, pressure_index, dot_pressure |
| Ball | ball (N nodes, one per historical delivery) |
| Query | query (1 node, learned embedding) |

### Convolution Operators by Edge Type

| Edge Relation | Operator | Why |
|--------------|----------|-----|
| conditions | GATv2Conv | Attention for hierarchical importance |
| relates_to | GATv2Conv | Attention for intra-layer relationships |
| matchup | GATv2Conv | Attention for matchup importance weighting |
| recent_precedes | TransformerConv | Position-aware with temporal distance |
| medium_precedes | TransformerConv | Position-aware with temporal distance |
| distant_precedes | SAGEConv | Simple aggregation for sparse connections |
| same_bowler | TransformerConv | Temporal decay for spell recency |
| same_batsman | TransformerConv | Temporal decay for form recency |
| same_matchup | GATv2Conv | Attention over matchup history |
| same_over | TransformerConv | Ball-in-over position features |
| faced_by, bowled_by | GATv2Conv | Attention for player history |
| informs | GATv2Conv | Attention-weighted dynamics |
| attends | GATv2Conv | Attention-weighted query aggregation |
| drives | GATv2Conv | Momentum influence on prediction |

## Why Heterogeneous GNN?

Cricket ball prediction is fundamentally about **typed relationships**:

1. **Players are different from venues** - Need different embedding strategies
2. **Matchup matters differently than momentum** - Edge semantics vary
3. **Temporal structure is meaningful** - Recent balls vs historical balls
4. **Hierarchy is natural** - Venue conditions team, team conditions players

A homogeneous GNN would treat all nodes and edges the same, losing this rich structure.

## Why These Specific Architectural Choices?

| Choice | Rationale |
|--------|-----------|
| **Hierarchical Player Embeddings** | Cold-start problem: unknown players fall back to team+role |
| **Per-edge Conv Operators** | Different edge semantics need different aggregation strategies |
| **TransformerConv for Temporal** | Supports edge_dim for temporal distance features |
| **FiLM Modulation** | Phase-conditional message passing (powerplay vs death) |
| **Hybrid Readout** | Combines edge-level matchup with graph-level context |
| **Innings-Conditional Heads** | 1st vs 2nd innings are different prediction tasks |

---

*Documentation generated for CricketHeteroGNN model architecture*
*Source: `/Users/kushaldsouza/Documents/Projects/cricketmodel/src/model/`*
