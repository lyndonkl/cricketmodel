# Readout and Prediction

## Overview

After message passing, we need to extract a fixed-size representation for prediction. The model provides several readout strategies:

1. **Query-only**: Use query node embedding directly
2. **Pooling**: Pool ball representations with attention
3. **Hybrid**: Combine matchup interaction with query context
4. **Innings-conditional**: Separate heads for 1st vs 2nd innings

## Query-Only Readout (Base Model)

The simplest approach: after message passing, the query node has aggregated information from the entire graph.

**Source**: `src/model/hetero_gnn.py:119-150`

```python
class CricketHeteroGNN(nn.Module):
    def forward(self, data):
        # 1. Encode nodes
        x_dict = self.encoders.encode_nodes(data)
        
        # 2. Message passing
        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict)
        
        # 3. Readout from query node
        query_repr = x_dict['query']  # [batch_size, hidden_dim]
        
        # 4. Predict
        logits = self.predictor(query_repr)
        return logits
```

**Why does this work?**

The query node receives messages from ALL context nodes via `attends` edges and from dynamics via `drives` edges. After 3 layers of message passing, it has "seen" information propagated from the entire graph.

```
All Context ──attends──▶ Query
Dynamics ───drives───▶ Query
                          │
                          ▼
                    [hidden_dim]
                          │
                          ▼
                    MLP Predictor
                          │
                          ▼
                    [7 classes]
```

## Pooling Readout (WithPooling Variant)

Adds explicit attention pooling over ball nodes.

**Source**: `src/model/hetero_gnn.py:198-278`

```python
class CricketHeteroGNNWithPooling(CricketHeteroGNN):
    def __init__(self, config):
        super().__init__(config)
        
        # Attention over ball nodes
        self.ball_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        # Combine query and pooled balls
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
```

**Forward Pass**:
```python
def forward(self, data):
    # ... encoding and message passing ...
    
    query_repr = x_dict['query']        # [batch, hidden_dim]
    ball_repr = x_dict['ball']          # [total_balls, hidden_dim]
    
    # Attention-weighted pooling over balls
    attn_scores = self.ball_attention(ball_repr)  # [total_balls, 1]
    attn_weights = torch.softmax(attn_scores, dim=0)
    pooled_balls = (attn_weights * ball_repr).sum(dim=0, keepdim=True)
    
    # Combine with query
    combined = torch.cat([query_repr, pooled_balls], dim=-1)
    combined = self.combiner(combined)
    
    logits = self.predictor(combined)
    return logits
```

**When to use?**
- When you want explicit control over ball aggregation
- When query-only doesn't capture enough ball-level detail

## Hybrid Readout (Hybrid Variant)

The key insight: **cricket prediction is fundamentally an edge-level task**. The outcome depends on the specific striker-bowler interaction, modulated by context.

**Source**: `src/model/hetero_gnn.py:281-384`

```python
class CricketHeteroGNNHybrid(CricketHeteroGNN):
    def __init__(self, config):
        super().__init__(config)
        
        # Matchup MLP: Striker + Bowler → Matchup representation
        self.matchup_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Query projection
        self.query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        # Combine matchup + context
        self.combiner = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Non-striker gate (P1.1)
        self.nonstriker_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim),
            nn.Sigmoid(),
        )
```

### Why Hybrid?

The prediction depends on:
1. **Matchup**: How does THIS striker fare against THIS bowler?
2. **Context**: What's the game situation (phase, pressure, momentum)?

Hybrid explicitly models both:

```
                    ┌─────────────────────┐
Striker embedding ──┤                     │
                    │   Matchup MLP       ├──▶ Matchup Repr
Bowler embedding ───┤                     │
                    └─────────────────────┘
                              │
                              │ (modulated by non-striker gate)
                              ▼
                    ┌─────────────────────┐
                    │                     │
                    │     Combiner        ├──▶ Combined Repr ──▶ Predictor
                    │                     │
                    └─────────────────────┘
                              ▲
                              │
Query embedding ──────────────┘
```

### Non-Striker Gate (P1.1)

**Why?** Running outcomes (Singles, Twos, Threes, Run-outs) depend on BOTH batsmen.

```python
# Base matchup from striker-bowler
base_matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))

# Non-striker modulates for running outcomes
ns_gate = self.nonstriker_gate(nonstriker)  # [batch, hidden_dim], range [0,1]
matchup = base_matchup * (1.0 + 0.1 * ns_gate)  # Small multiplicative modulation
```

The gate is small (0.1 factor) because:
- Non-striker doesn't affect boundaries (Fours, Sixes)
- Non-striker doesn't affect dots or wickets (mostly)
- But non-striker IS crucial for running decisions

### Forward Pass

```python
def forward(self, data):
    # ... encoding and message passing ...
    
    # Matchup interaction
    striker = x_dict['striker_identity']    # [batch, hidden_dim]
    bowler = x_dict['bowler_identity']      # [batch, hidden_dim]
    nonstriker = x_dict['nonstriker_identity']
    
    base_matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))
    
    # Non-striker gate modulation
    ns_gate = self.nonstriker_gate(nonstriker)
    matchup = base_matchup * (1.0 + 0.1 * ns_gate)
    
    # Context from query
    query = self.query_proj(x_dict['query'])
    
    # Combine matchup + context
    combined = torch.cat([matchup, query], dim=-1)
    combined = self.combiner(combined)
    
    logits = self.predictor(combined)
    return logits
```

## Innings-Conditional Heads

**The Problem**: 1st and 2nd innings are fundamentally different prediction tasks.

| Innings | Goal | Key Factors |
|---------|------|-------------|
| 1st | Maximize score | Wickets in hand, overs remaining |
| 2nd | Chase target | Required rate, balls remaining, wickets |

**Source**: `src/model/hetero_gnn.py:501-629`

```python
class CricketHeteroGNNInningsConditional(CricketHeteroGNNHybrid):
    def __init__(self, config):
        super().__init__(config)
        
        # First innings head (standard MLP)
        self.first_innings_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )
        
        # Second innings head (with chase state injection)
        chase_state_dim = 7
        self.second_innings_head = nn.Sequential(
            nn.Linear(config.hidden_dim + chase_state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )
```

### Why Two Heads?

1. **Different input**: 2nd innings head receives chase_state features directly
2. **Different decision boundaries**: Chase scenarios need different thresholds
3. **Shared encoder**: Both use the same message passing (identical context understanding)

### Forward Pass

```python
def forward(self, data):
    # ... encoding, message passing, hybrid readout ...
    combined = self.combiner(torch.cat([matchup, query], dim=-1))
    
    # Get chase info
    chase_state = data['chase_state'].x  # [batch, 7]
    is_chase = data.is_chase             # [batch] bool
    
    # Compute both predictions
    first_innings_logits = self.first_innings_head(combined)
    
    # Second innings: inject chase state
    combined_with_chase = torch.cat([combined, chase_state], dim=-1)
    second_innings_logits = self.second_innings_head(combined_with_chase)
    
    # Select based on innings
    is_chase_expanded = is_chase.unsqueeze(-1).expand_as(first_innings_logits)
    logits = torch.where(is_chase_expanded, second_innings_logits, first_innings_logits)
    
    return logits
```

## The Full Model (Production)

`CricketHeteroGNNFull` combines all features:
- Hierarchical player embeddings (cold-start)
- Phase-modulated message passing (FiLM)
- Hybrid readout (matchup + query)
- Non-striker gate
- Innings-conditional heads

**Source**: `src/model/hetero_gnn.py:632-850`

```python
class CricketHeteroGNNFull(nn.Module):
    def forward(self, data):
        # 1. Encode (with hierarchical player embeddings)
        x_dict = self.encoders.encode_nodes(data)
        
        # 2. Message passing (with FiLM conditioning if enabled)
        if self.config.use_phase_modulation:
            condition = torch.cat([phase_state, chase_state, wicket_buffer], dim=-1)
            for conv_block in self.conv_stack:
                x_dict = conv_block(x_dict, edge_index_dict, condition, edge_attr_dict)
        else:
            for conv_block in self.conv_stack:
                x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict)
        
        # 3. Hybrid readout with non-striker gate
        striker = x_dict['striker_identity']
        bowler = x_dict['bowler_identity']
        nonstriker = x_dict['nonstriker_identity']
        
        base_matchup = self.matchup_mlp(torch.cat([striker, bowler], dim=-1))
        ns_gate = self.nonstriker_gate(nonstriker)
        matchup = base_matchup * (1.0 + 0.1 * ns_gate)
        
        query = self.query_proj(x_dict['query'])
        combined = self.combiner(torch.cat([matchup, query], dim=-1))
        
        # 4. Innings-conditional prediction
        if self.config.use_innings_conditional:
            first_logits = self.first_innings_head(combined)
            combined_chase = torch.cat([combined, chase_state], dim=-1)
            second_logits = self.second_innings_head(combined_chase)
            logits = torch.where(is_chase, second_logits, first_logits)
        else:
            logits = self.predictor(combined)
        
        return logits
```

## MLP Predictor Architecture

All prediction heads use a 3-layer MLP:

```python
self.predictor = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),        # [128] → [128]
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),   # [128] → [64]
    nn.LayerNorm(hidden_dim // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes),  # [64] → [7]
)
```

**Why this architecture?**
- **Two hidden layers**: Sufficient capacity for class boundaries
- **Dimension reduction**: 128 → 64 → 7 gradually reduces
- **LayerNorm + GELU**: Stable training, smooth activations
- **Dropout**: Regularization at each layer

## Summary: Readout Strategies

| Variant | Readout Strategy | Best For |
|---------|------------------|----------|
| Base | Query only | Simple baseline |
| WithPooling | Query + pooled balls | Explicit ball aggregation |
| Hybrid | Matchup MLP + Query | Matchup-focused prediction |
| InningsConditional | Separate heads by innings | Chase-aware prediction |
| Full | All of the above | **Production deployment** |

## Output Format

All variants produce:
```
logits: [batch_size, 7]
```

Where each of the 7 dimensions corresponds to:
- 0: Dot
- 1: Single
- 2: Two
- 3: Three
- 4: Four
- 5: Six
- 6: Wicket

Apply `softmax(logits)` for probabilities.

---

*Next: [07-model-variants.md](./07-model-variants.md) - Detailed comparison of all 6 model variants*
