# Architecture Overview

## The Problem: Ball-by-Ball Prediction

Cricket ball prediction is a complex sequential decision problem:
- **Input**: Complete game state before a delivery (venue, players, score, momentum, history)
- **Output**: Probability distribution over 7 outcomes (Dot, Single, Two, Three, Four, Six, Wicket)

The challenge is that the outcome depends on many interacting factors:
- Who is batting vs who is bowling (matchup)
- Current game situation (phase, pressure, chase target)
- Recent momentum (form of batsman/bowler)
- Historical patterns (same matchup earlier, same over dynamics)

## Why Heterogeneous Graph Neural Network?

### The Natural Structure of Cricket

Cricket data has inherent hierarchical and typed relationships:

```
VENUE (Melbourne Cricket Ground)
  └─── conditions ───▶ TEAMS (Australia vs India)
                           └─── conditions ───▶ PLAYERS (Kohli facing Starc)
                                                    └─── conditions ───▶ DYNAMICS (momentum, pressure)
                                                                             └─── informed by ───▶ BALLS (recent deliveries)
```

A heterogeneous GNN respects this structure by:
1. **Different node types** for entities with different semantics
2. **Different edge types** for relationships with different meanings
3. **Different convolution operators** for edges requiring different aggregation

### Alternative Approaches and Why GNN Wins

| Approach | Limitation |
|----------|------------|
| Tabular ML (XGBoost, etc.) | Loses graph structure, can't model variable-length history |
| Sequence Models (LSTM, Transformer) | Treats all context as sequence, loses typed relationships |
| Homogeneous GNN | Treats all nodes/edges same, loses semantic distinctions |
| **Heterogeneous GNN** | Preserves all structure, allows type-specific processing |

## High-Level Architecture

```python
class CricketHeteroGNN(nn.Module):
    """
    Architecture:
    1. Node Encoders: Project each node type to common hidden_dim
    2. Message Passing: Stack of HeteroConvBlocks  
    3. Readout: Extract query node representation
    4. Prediction: MLP classifier
    """
```

### Stage 1: Node Encoding

Each of the 21 node types needs to be projected to a common `hidden_dim`:

```
Raw Features ────▶ Type-Specific Encoder ────▶ [hidden_dim] vector

Examples:
- venue (ID: 42) ────▶ EntityEncoder ────▶ [128] embedding
- phase_state ([1,0,0,0.5,0,0]) ────▶ FeatureEncoder ────▶ [128] embedding
- ball (18 features + player IDs) ────▶ BallEncoder ────▶ [128] embedding
```

### Stage 2: Message Passing

Information flows through the graph via typed edges:

```
For each layer (default: 3 layers):
    For each edge type:
        messages = ConvOperator[edge_type](source_nodes, target_nodes, edge_index)
    
    For each node type:
        new_embedding = Aggregate(incoming_messages) + residual
        new_embedding = LayerNorm(new_embedding)
        new_embedding = Dropout(new_embedding)
```

The key insight: **different edge types use different convolution operators**:
- `GATv2Conv` for attention-weighted relationships
- `TransformerConv` for temporal edges with distance features
- `SAGEConv` for simple aggregation

### Stage 3: Readout

The **query node** serves as a learned aggregation point:

```
After message passing:
    query_embedding = x_dict['query']  # [batch_size, hidden_dim]
    
    # All context has flowed into query via 'attends' edges
```

In advanced variants (Hybrid, Full), readout also uses:
- **Matchup MLP**: Concatenate striker + bowler embeddings
- **Non-striker Gate**: Modulate for running outcomes
- **Context Combination**: Merge matchup with query aggregation

### Stage 4: Prediction

```python
self.predictor = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.LayerNorm(hidden_dim // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes),  # 7 classes
)
```

## Design Philosophy

### 1. Respect the Data Structure

The model architecture mirrors the natural structure of cricket:
- Hierarchical conditioning (venue → teams → players → dynamics)
- Typed relationships (matchup is different from momentum)
- Temporal ordering (recent balls matter more)

### 2. Handle Cold Start Gracefully

Unknown players are a reality (debuts, rare players):
- **Hierarchical Player Embeddings**: Fall back to team + role when player unknown
- This means even a debutant gets meaningful representation

### 3. Allow Phase-Specific Behavior

Cricket phases have different dynamics:
- **Powerplay** (overs 0-5): Aggressive batting, fielding restrictions
- **Middle** (overs 6-15): Rotation, building innings
- **Death** (overs 16-19): High risk/reward

**FiLM Modulation** allows the model to adapt its message passing based on phase.

### 4. Distinguish Innings Tasks

First and second innings are fundamentally different:
- **1st Innings**: Maximize score with wickets in hand (no target)
- **2nd Innings**: Chase target with risk/reward tradeoffs

**Innings-Conditional Heads** use separate prediction heads for each task.

## Model Variants

The codebase provides 6 model variants with increasing sophistication:

| Variant | Key Addition | Use Case |
|---------|--------------|----------|
| `CricketHeteroGNN` | Base model | Baseline, simple deployment |
| `CricketHeteroGNNWithPooling` | Ball attention pooling | Alternative aggregation |
| `CricketHeteroGNNHybrid` | Matchup + query readout | Better matchup modeling |
| `CricketHeteroGNNPhaseModulated` | FiLM conditioning | Phase-aware predictions |
| `CricketHeteroGNNInningsConditional` | Separate innings heads | Chase-aware predictions |
| `CricketHeteroGNNFull` | All features combined | **Production model** |

See [07-model-variants.md](./07-model-variants.md) for detailed comparison.

## Key Hyperparameters

```python
ModelConfig(
    # Architecture
    hidden_dim=128,      # Embedding dimension for all node types
    num_layers=3,        # Number of message passing layers
    num_heads=4,         # Attention heads in GATv2Conv/TransformerConv
    
    # Embeddings
    venue_embed_dim=32,   # Venues are simpler entities
    team_embed_dim=32,    # Teams have moderate complexity
    player_embed_dim=64,  # Players are most complex (many individuals)
    role_embed_dim=16,    # Roles are coarse categories
    
    # Regularization
    dropout=0.1,
    
    # FiLM Conditioning (for phase-modulated variants)
    phase_dim=6,          # Phase state features
    chase_dim=7,          # Chase state features  
    resource_dim=2,       # Wicket buffer features
    condition_dim=15,     # Total conditioning dimension
)
```

## Information Flow Summary

```
1. ENCODE: Raw features → [hidden_dim] embeddings
   ├── Entity nodes: ID → Embedding Table → MLP → [hidden_dim]
   ├── Feature nodes: [features] → MLP → [hidden_dim]
   ├── Ball nodes: [features, bowler_id, batsman_id] → Embedding + MLP → [hidden_dim]
   └── Query node: Learned parameter → [hidden_dim]

2. MESSAGE PASS (x3 layers):
   ├── Hierarchical: Global → State → Actor → Dynamics
   ├── Intra-layer: Within each layer (matchups, relationships)
   ├── Temporal: Ball-to-ball (precedes, same_bowler, same_over)
   ├── Cross-domain: Ball → Player identities, Ball → Dynamics
   └── Query: All context → Query node

3. READOUT:
   ├── Base: Query embedding only
   └── Hybrid: Matchup MLP + Query projection + Combination

4. PREDICT:
   └── MLP: [hidden_dim] → [hidden_dim//2] → [7 classes]
```

---

*Next: [02-graph-structure.md](./02-graph-structure.md) - Detailed node and edge type definitions*
