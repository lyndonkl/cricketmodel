# Model Variants

## Overview

The codebase provides 6 model variants with increasing sophistication:

| Variant | Key Features | Parameters (approx) |
|---------|--------------|---------------------|
| `CricketHeteroGNN` | Base model | ~2M |
| `CricketHeteroGNNWithPooling` | + Ball attention pooling | ~2.1M |
| `CricketHeteroGNNHybrid` | + Matchup + Query readout | ~2.3M |
| `CricketHeteroGNNPhaseModulated` | + FiLM conditioning | ~2.8M |
| `CricketHeteroGNNInningsConditional` | + Separate innings heads | ~2.5M |
| `CricketHeteroGNNFull` | All features combined | ~3M |

## Variant 1: CricketHeteroGNN (Base)

**Source**: `src/model/hetero_gnn.py:60-196`

**Architecture**:
```
Encoders → Conv Stack → Query Readout → MLP Predictor
```

**Key Characteristics**:
- Uses `NodeEncoderDict` for all node types
- Standard `HeteroConvBlock` (no FiLM)
- Query-only readout
- Single prediction head

**Use Case**: Baseline, simple deployment, ablation studies

```python
class CricketHeteroGNN(nn.Module):
    def __init__(self, config: ModelConfig):
        self.encoders = NodeEncoderDict(...)
        self.conv_stack = build_conv_stack(...)
        self.predictor = nn.Sequential(...)
    
    def forward(self, data):
        x_dict = self.encoders.encode_nodes(data)
        for conv in self.conv_stack:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        query_repr = x_dict['query']
        return self.predictor(query_repr)
```

## Variant 2: CricketHeteroGNNWithPooling

**Source**: `src/model/hetero_gnn.py:198-278`

**Inherits from**: `CricketHeteroGNN`

**Additional Components**:
```python
self.ball_attention = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.Tanh(),
    nn.Linear(hidden_dim // 2, 1),
)

self.combiner = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

**Key Difference**: Explicit attention pooling over ball nodes, combined with query.

**Use Case**: When you want more explicit control over ball aggregation

```python
def forward(self, data):
    # ... message passing ...
    
    query_repr = x_dict['query']
    ball_repr = x_dict['ball']
    
    # Attention pooling
    attn_scores = self.ball_attention(ball_repr)
    attn_weights = torch.softmax(attn_scores, dim=0)
    pooled_balls = (attn_weights * ball_repr).sum(dim=0)
    
    # Combine
    combined = torch.cat([query_repr, pooled_balls], dim=-1)
    combined = self.combiner(combined)
    
    return self.predictor(combined)
```

**Limitation**: Simplified batch handling (assumes single sample per batch in pooling). In practice, need proper batch indexing.

## Variant 3: CricketHeteroGNNHybrid

**Source**: `src/model/hetero_gnn.py:281-384`

**Inherits from**: `CricketHeteroGNN`

**Additional Components**:
```python
# Matchup MLP
self.matchup_mlp = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)

# Query projection
self.query_proj = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
)

# Matchup + Context combiner
self.combiner = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)

# Non-striker gate (P1.1)
self.nonstriker_gate = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 4),
    nn.GELU(),
    nn.Linear(hidden_dim // 4, hidden_dim),
    nn.Sigmoid(),
)
```

**Key Insight**: Cricket prediction is an EDGE-level task (striker vs bowler).

**Flow**:
```
Striker ─┐
         ├─▶ Matchup MLP ──┬──▶ Matchup (modulated by NS gate)
Bowler ──┘                  │
                            │
Non-striker ──▶ Gate ───────┘

Query ──▶ Query Proj ────────────┐
                                 │
Matchup + Query ──▶ Combiner ────▶ Predictor
```

**Use Case**: Better matchup modeling, respects the structure of cricket

## Variant 4: CricketHeteroGNNPhaseModulated

**Source**: `src/model/hetero_gnn.py:387-498`

**Inherits from**: `CricketHeteroGNNHybrid`

**Key Difference**: Uses `PhaseModulatedConvBlock` instead of `HeteroConvBlock`

```python
def __init__(self, config):
    # Override conv_stack with FiLM version
    self.conv_stack = build_phase_modulated_conv_stack(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        phase_dim=config.phase_dim,  # 6
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
```

**Forward Pass**:
```python
def forward(self, data):
    x_dict = self.encoders.encode_nodes(data)
    
    # Get phase condition
    phase_condition = data['phase_state'].x  # [batch, 6]
    
    # Message passing with FiLM
    for conv_block in self.conv_stack:
        x_dict = conv_block(x_dict, edge_index_dict, phase_condition, edge_attr_dict)
    
    # ... hybrid readout ...
```

**Use Case**: Phase-aware predictions (powerplay vs death)

**Note**: This variant uses ONLY phase_state (6 dims) for conditioning, not the full 15-dim condition.

## Variant 5: CricketHeteroGNNInningsConditional

**Source**: `src/model/hetero_gnn.py:501-629`

**Inherits from**: `CricketHeteroGNNHybrid`

**Additional Components**:
```python
# First innings head
self.first_innings_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    ...
    nn.Linear(hidden_dim // 2, num_classes),
)

# Second innings head (with chase state injection)
chase_state_dim = 7
self.second_innings_head = nn.Sequential(
    nn.Linear(hidden_dim + chase_state_dim, hidden_dim),
    ...
    nn.Linear(hidden_dim // 2, num_classes),
)
```

**Forward Pass**:
```python
def forward(self, data):
    # ... hybrid readout to get 'combined' ...
    
    chase_state = data['chase_state'].x  # [batch, 7]
    is_chase = data.is_chase             # [batch] bool
    
    # Compute both
    first_logits = self.first_innings_head(combined)
    second_logits = self.second_innings_head(
        torch.cat([combined, chase_state], dim=-1)
    )
    
    # Select based on innings
    logits = torch.where(is_chase, second_logits, first_logits)
    return logits
```

**Use Case**: Chase-aware predictions (different strategy for chasing)

**Why Two Heads?**
- 1st innings: No target, maximize score
- 2nd innings: Known target, balance risk vs required rate
- Chase state features only make sense in 2nd innings

## Variant 6: CricketHeteroGNNFull (Production)

**Source**: `src/model/hetero_gnn.py:632-850`

**Inherits from**: `nn.Module` (standalone)

**Combines ALL features**:
1. Hierarchical player embeddings
2. Phase-modulated message passing (FiLM with full 15-dim condition)
3. Hybrid matchup + query readout
4. Non-striker gate
5. Innings-conditional heads

```python
class CricketHeteroGNNFull(nn.Module):
    def __init__(self, config):
        # Encoders with hierarchical player
        self.encoders = NodeEncoderDict(
            use_hierarchical_player=config.use_hierarchical_player,
            ...
        )
        
        # Phase-modulated conv stack
        if config.use_phase_modulation:
            self.conv_stack = build_phase_modulated_conv_stack(
                condition_dim=config.condition_dim,  # 15 = phase(6) + chase(7) + wicket(2)
                ...
            )
        
        # Hybrid readout
        self.matchup_mlp = ...
        self.query_proj = ...
        self.combiner = ...
        self.nonstriker_gate = ...
        
        # Innings-conditional heads
        if config.use_innings_conditional:
            self.first_innings_head = ...
            self.second_innings_head = ...
```

**Forward Pass**:
```python
def forward(self, data):
    # 1. Encode (hierarchical player if enabled)
    x_dict = self.encoders.encode_nodes(data)
    
    # 2. Message passing (FiLM if enabled)
    if self.config.use_phase_modulation:
        condition = torch.cat([phase_state, chase_state, wicket_buffer], dim=-1)
        for conv in self.conv_stack:
            x_dict = conv(x_dict, edge_index_dict, condition, edge_attr_dict)
    
    # 3. Hybrid readout
    matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))
    matchup = matchup * (1.0 + 0.1 * self.nonstriker_gate(nonstriker))
    query = self.query_proj(x_dict['query'])
    combined = self.combiner(torch.cat([matchup, query], dim=-1))
    
    # 4. Predict (innings-conditional if enabled)
    if self.config.use_innings_conditional:
        first_logits = self.first_innings_head(combined)
        second_logits = self.second_innings_head(torch.cat([combined, chase_state], dim=-1))
        logits = torch.where(is_chase, second_logits, first_logits)
    else:
        logits = self.predictor(combined)
    
    return logits
```

**Configuration Options**:
```python
ModelConfig(
    use_hybrid_readout=True,          # Always True in Full
    use_innings_conditional=True,     # Separate heads
    use_hierarchical_player=True,     # Cold-start handling
    use_phase_modulation=True,        # FiLM conditioning
)
```

**Use Case**: Production deployment, best accuracy

## Feature Comparison Matrix

| Feature | Base | Pooling | Hybrid | Phase | Innings | Full |
|---------|------|---------|--------|-------|---------|------|
| Standard encoders | Y | Y | Y | Y | Y | Y |
| Hierarchical player | N | N | N | N | N | Y |
| Query-only readout | Y | - | - | - | - | - |
| Ball attention pooling | N | Y | N | N | N | N |
| Matchup MLP | N | N | Y | Y | Y | Y |
| Non-striker gate | N | N | Y | Y | Y | Y |
| FiLM conditioning | N | N | N | Y | N | Y |
| Innings-conditional | N | N | N | N | Y | Y |
| Condition dim | - | - | - | 6 | - | 15 |

## When to Use Each Variant

| Variant | Use When |
|---------|----------|
| Base | Baseline, ablation studies, simple deployment |
| WithPooling | Need explicit ball-level attention (experimental) |
| Hybrid | Want matchup-focused predictions without FiLM overhead |
| PhaseModulated | Phase-aware predictions, no innings distinction needed |
| InningsConditional | Chase-aware predictions, no phase modulation needed |
| **Full** | **Production: best accuracy, handles all scenarios** |

## Creating Models

### From Configuration
```python
config = ModelConfig(
    num_venues=100,
    num_teams=30,
    num_players=2000,
    hidden_dim=128,
    num_layers=3,
    use_phase_modulation=True,
    use_innings_conditional=True,
)

model = CricketHeteroGNNFull(config)
```

### From Dataset Metadata
```python
# Get metadata from dataset
metadata = dataset.get_metadata()
# {'num_venues': 87, 'num_teams': 24, 'num_players': 1856}

# Create model
model = CricketHeteroGNNFull.from_dataset_metadata(
    metadata,
    hidden_dim=128,
    num_layers=3,
)
```

---

*Next: [08-features-reference.md](./08-features-reference.md) - Complete feature dimensions and meanings*
