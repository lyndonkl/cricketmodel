# Convolution Layer Selection

## The Key Insight

Different edge types represent different kinds of relationships. **The convolution operator should match the semantic meaning of the edge**.

| Relationship Type | What It Needs | Operator |
|------------------|---------------|----------|
| Importance-weighted | Learn which edges matter more | GATv2Conv |
| Position/distance-aware | Use edge features for attention | TransformerConv |
| Simple aggregation | Just combine neighbors | SAGEConv |

## Available Operators

The model uses three convolution operators from PyTorch Geometric:

### GATv2Conv (Graph Attention v2)

**What it does**: Learns attention weights for each edge, then aggregates neighbor features weighted by attention.

```python
# Pseudocode
for each target node t:
    for each source node s connected to t:
        attention_score = LeakyReLU(a^T [W_s h_s || W_t h_t])
    attention_weights = Softmax(attention_scores)
    output[t] = sum(attention_weights[s] * W_s h_s)
```

**Parameters**:
```python
GATv2Conv(
    in_channels=hidden_dim,      # Input dimension
    out_channels=head_dim,       # Output per head (hidden_dim // num_heads)
    heads=num_heads,             # Number of attention heads
    add_self_loops=False,        # No self-loops (heterogeneous graph)
    concat=True,                 # Concatenate heads (output = num_heads * head_dim)
    dropout=dropout,             # Attention dropout
)
```

**When to use**: When the model should learn which edges are most important.

**Cricket examples**:
- Striker-bowler matchup: "Is this a favorable matchup?"
- Query aggregation: "Which context nodes are most relevant?"
- Dynamics influence: "Is batting momentum more important than pressure right now?"

### TransformerConv

**What it does**: Attention-based aggregation that can incorporate edge features.

```python
# Pseudocode (with edge features)
for each target node t:
    for each source node s with edge_attr e_st:
        attention_score = (Q_t)^T (K_s) / sqrt(d)
        # Edge features are incorporated into key/value
        attention_score += edge_projection(e_st)
    attention_weights = Softmax(attention_scores)
    output[t] = sum(attention_weights[s] * V_s)
```

**Parameters**:
```python
TransformerConv(
    in_channels=hidden_dim,
    out_channels=head_dim,
    heads=num_heads,
    concat=True,
    dropout=dropout,
    edge_dim=1,              # Dimension of edge features
)
```

**When to use**: When edges have features (like temporal distance) that should influence attention.

**Cricket examples**:
- Temporal edges: "Recent balls should be weighted more heavily"
- Same-bowler edges: "Recent balls in spell matter more"
- Same-over edges: "Ball position in over affects relevance"

### SAGEConv

**What it does**: Simple mean/sum/max aggregation of neighbor features.

```python
# Pseudocode (mean aggregation)
for each target node t:
    neighbors = all nodes connected to t
    aggregated = Mean(h_s for s in neighbors)
    output[t] = W * Concat(h_t, aggregated)
```

**Parameters**:
```python
SAGEConv(
    in_channels=hidden_dim,
    out_channels=hidden_dim,
    aggr='mean',             # 'mean', 'sum', or 'max'
)
```

**When to use**: When all edges are equally important, simple aggregation suffices.

**Cricket examples**:
- Distant temporal edges: Sparse historical connections, don't need fine-grained attention
- Default fallback: Unknown edge types

## Edge Type to Operator Mapping

**Source**: `src/model/conv_builder.py:97-293`

### Hierarchical Edges (`conditions`)

```python
if rel == 'conditions':
    convs[edge_type] = GATv2Conv(
        hidden_dim, head_dim, heads=num_heads,
        add_self_loops=False, concat=True, dropout=dropout,
    )
```

**Why GATv2Conv?**
- Venue conditions teams, but some venue effects matter more (pitch type, boundaries)
- Teams condition players, but team's batting style affects striker more than non-striker
- Attention learns these importance weights

### Intra-Layer Edges (`relates_to`)

```python
elif rel == 'relates_to':
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Score-chase relationship: More important in 2nd innings
- Phase-time pressure: Death overs amplify time pressure
- Attention captures these varying relationships

### Actor Matchup Edges (`matchup`)

```python
elif rel == 'matchup':
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Striker vs bowler: THE key relationship for prediction
- Identity to state: Current form modulates identity
- Partnership connections: Both batsmen contribute differently
- Attention weights these appropriately

### Multi-Scale Temporal Edges

```python
elif rel in ['recent_precedes', 'medium_precedes']:
    convs[edge_type] = TransformerConv(
        hidden_dim, head_dim, heads=num_heads,
        concat=True, dropout=dropout,
        edge_dim=1,  # Temporal distance feature
    )
```

**Why TransformerConv?**
- Edges have temporal distance as edge attribute
- Model should learn decay: recent balls matter more
- Edge features allow position-aware attention

```python
elif rel == 'distant_precedes':
    convs[edge_type] = SAGEConv(
        hidden_dim, hidden_dim, aggr='mean',
    )
```

**Why SAGEConv for distant?**
- Distant edges are sparse (every 6 balls)
- Historical patterns don't need fine-grained attention
- Simple mean aggregation is efficient

### Same-Actor Edges

```python
elif rel in ['same_bowler', 'same_batsman']:
    convs[edge_type] = TransformerConv(
        hidden_dim, head_dim, heads=num_heads,
        concat=True, dropout=dropout,
        edge_dim=1,  # Temporal distance
    )
```

**Why TransformerConv?**
- Same-bowler connects all balls by same bowler
- Temporal distance matters: recent spell performance > early spell
- Edge feature encodes this decay

```python
elif rel == 'same_matchup':
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Same-matchup edges are causal (no future info)
- No edge attributes (all matchup history is relevant)
- Attention learns which historical encounters matter

### Same-Over Edges

```python
elif rel == 'same_over':
    convs[edge_type] = TransformerConv(
        hidden_dim, head_dim, heads=num_heads,
        concat=True, dropout=dropout,
        edge_dim=1,  # Ball-in-over position
    )
```

**Why TransformerConv?**
- Over boundaries are significant (new bowler, batsmen swap)
- Within-over has tight local coherence
- Ball position matters: ball 1 vs ball 6 differ significantly
- Edge feature encodes position

### Cross-Domain Edges

```python
elif rel in ['faced_by', 'bowled_by', 'partnered_by']:
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Connect historical balls to current players
- Attention learns which balls are most relevant
- Recent performance weighted appropriately

### Dynamics Edges

```python
elif rel == 'informs':
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Recent balls inform momentum/pressure
- Boundaries and wickets contribute more than dots
- Attention learns these importance weights

### Query Aggregation

```python
elif rel == 'attends':
    convs[edge_type] = GATv2Conv(...)

elif rel == 'drives':
    convs[edge_type] = GATv2Conv(...)
```

**Why GATv2Conv?**
- Query aggregates from all context
- Should weight different contexts by relevance
- `drives` captures feedback loop: momentum → prediction

### Default Fallback

```python
else:
    convs[edge_type] = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
```

Simple mean aggregation for any unknown edge types.

## HeteroConv Assembly

All operators are assembled into a single `HeteroConv` layer:

```python
def build_hetero_conv(hidden_dim, num_heads, dropout, edge_types):
    convs = {}
    
    for edge_type in edge_types:
        src_type, rel, tgt_type = edge_type
        
        if rel == 'conditions':
            convs[edge_type] = GATv2Conv(...)
        elif rel in ['recent_precedes', 'medium_precedes']:
            convs[edge_type] = TransformerConv(..., edge_dim=1)
        # ... etc
    
    return HeteroConv(convs, aggr='sum')
```

**The `aggr='sum'` parameter**: When a node receives messages from multiple edge types, they are summed. This is processed downstream with residual + LayerNorm.

## Summary Table

| Edge Relation | Operator | edge_dim | Rationale |
|--------------|----------|----------|-----------|
| conditions | GATv2Conv | - | Learn hierarchical importance |
| relates_to | GATv2Conv | - | Learn intra-layer relevance |
| matchup | GATv2Conv | - | Learn matchup importance |
| recent_precedes | TransformerConv | 1 | Temporal distance decay |
| medium_precedes | TransformerConv | 1 | Temporal distance decay |
| distant_precedes | SAGEConv | - | Simple aggregation (sparse) |
| same_bowler | TransformerConv | 1 | Spell recency decay |
| same_batsman | TransformerConv | 1 | Form recency decay |
| same_matchup | GATv2Conv | - | Attention over matchup history |
| same_over | TransformerConv | 1 | Ball-in-over position |
| faced_by | GATv2Conv | - | Attention over batsman history |
| bowled_by | GATv2Conv | - | Attention over bowler history |
| partnered_by | GATv2Conv | - | Attention over partnership history |
| informs | GATv2Conv | - | Attention-weighted dynamics |
| attends | GATv2Conv | - | Context aggregation |
| drives | GATv2Conv | - | Momentum influence |
| (default) | SAGEConv | - | Simple fallback |

## Why Not Just Use GATv2Conv for Everything?

1. **Temporal edges need edge features**: GATv2Conv doesn't support edge_dim
2. **Distant temporal doesn't need attention**: Simple mean is faster and sufficient
3. **TransformerConv is slower**: Only use where edge features add value

## Why Not Just Use TransformerConv for Everything?

1. **Many edges don't have edge features**: Would waste the edge_dim parameter
2. **Computational cost**: TransformerConv is heavier than GATv2Conv
3. **Simpler is better**: GATv2Conv is well-understood and efficient

---

*Next: [05-message-passing.md](./05-message-passing.md) - HeteroConvBlock and FiLM conditioning*
