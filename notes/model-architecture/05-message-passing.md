# Message Passing

## Overview

Message passing is the core of the GNN: information flows along edges, and nodes update based on aggregated messages from neighbors. The model uses 3 layers of message passing (configurable via `num_layers`).

## HeteroConvBlock

The basic message passing unit wraps HeteroConv with residual connections and normalization.

**Source**: `src/model/conv_builder.py:424-519`

```python
class HeteroConvBlock(nn.Module):
    """
    Structure:
    1. HeteroConv message passing
    2. Residual connection (skip from input)
    3. Layer normalization per node type
    4. Dropout
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_types: Optional[List] = None,
        node_types: Optional[List[str]] = None,
    ):
        self.conv = build_hetero_conv(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            edge_types=edge_types,
        )
        
        # Layer norm per node type
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
        self.dropout = nn.Dropout(dropout)
```

### Forward Pass

```python
def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
    # 1. Message passing
    if edge_attr_dict is not None:
        out_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
    else:
        out_dict = self.conv(x_dict, edge_index_dict)
    
    # 2. Residual + Norm + Dropout per node type
    result = {}
    for node_type in out_dict:
        h = out_dict[node_type]
        
        # Residual connection
        if node_type in x_dict and x_dict[node_type].shape[0] > 0:
            h = h + x_dict[node_type]
        
        # Layer norm
        if node_type in self.norms:
            h = self.norms[node_type](h)
        
        # Dropout
        h = self.dropout(h)
        result[node_type] = h
    
    # Preserve nodes not updated by conv
    for node_type in x_dict:
        if node_type not in result:
            result[node_type] = x_dict[node_type]
    
    return result
```

### Why These Components?

**Residual Connection**:
- Prevents gradient vanishing in deep networks
- Allows identity mapping if message passing isn't helpful
- Critical for training stability

**Layer Normalization** (per node type):
- Normalizes across feature dimension
- Each node type has its own normalization statistics
- More stable than batch normalization for graphs

**Dropout**:
- Regularization to prevent overfitting
- Applied after normalization

## PhaseModulatedConvBlock (FiLM)

The advanced message passing unit adds **Feature-wise Linear Modulation (FiLM)** conditioning.

**Source**: `src/model/conv_builder.py:296-421`

### What is FiLM?

FiLM modulates neural network activations based on a conditioning signal:

```
output = gamma * input + beta
```

Where `gamma` and `beta` are learned functions of the condition.

**Paper**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018

### Why FiLM for Cricket?

Different game situations require different attention patterns:

| Phase | Characteristics | Message Passing Behavior |
|-------|----------------|-------------------------|
| Powerplay | Fielding restrictions, aggressive | Weight recent boundaries more |
| Middle | Building innings, rotation | Weight partnership stability |
| Death | High risk/reward | Weight pressure heavily |
| Chase (tight) | Need runs quickly | Weight RRR, urgency |
| Chase (easy) | Wickets more valuable | Weight wickets in hand |

FiLM allows the model to adapt its message passing based on phase WITHOUT needing separate networks for each phase.

### FiLM Layer Implementation

**Source**: `src/model/conv_builder.py:21-83`

```python
class FiLMLayer(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int):
        # Generate gamma and beta from conditioning signal
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma and beta
        )
        
        # Initialize close to identity: gamma=1, beta=0
        with torch.no_grad():
            self.film_generator[-1].weight.fill_(0.0)
            self.film_generator[-1].bias.zero_()
            self.film_generator[-1].bias[:hidden_dim].fill_(1.0)  # gamma=1
    
    def forward(self, x, condition):
        # Generate gamma and beta
        film_params = self.film_generator(condition)  # [batch, hidden_dim * 2]
        gamma, beta = film_params.chunk(2, dim=-1)    # Each: [batch, hidden_dim]
        
        # Broadcast to match nodes
        if gamma.shape[0] == 1 and x.shape[0] > 1:
            gamma = gamma.expand(x.shape[0], -1)
            beta = beta.expand(x.shape[0], -1)
        
        # Apply modulation
        return gamma * x + beta
```

**Key Design: Identity Initialization**
- `gamma=1, beta=0` means FiLM initially does nothing
- Model can learn to modulate as needed
- Prevents disruption at the start of training

### PhaseModulatedConvBlock Structure

```python
class PhaseModulatedConvBlock(nn.Module):
    """
    Structure:
    1. HeteroConv message passing
    2. Residual connection
    3. Layer normalization
    4. FiLM modulation (per node type)
    5. Dropout
    """
    
    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int = 14,  # phase(5) + chase(7) + wicket_buffer(2)
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        self.conv = build_hetero_conv(...)
        
        self.norms = nn.ModuleDict({...})
        
        # FiLM modulation per node type
        self.film_layers = nn.ModuleDict({
            node_type: FiLMLayer(condition_dim, hidden_dim)
            for node_type in node_types
        })
        
        self.dropout = nn.Dropout(dropout)
```

### Forward Pass with FiLM

```python
def forward(self, x_dict, edge_index_dict, condition, edge_attr_dict=None):
    # 1. Message passing
    out_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
    
    # 2. Residual + Norm + FiLM + Dropout
    result = {}
    for node_type in out_dict:
        h = out_dict[node_type]
        
        # Residual
        if node_type in x_dict and x_dict[node_type].shape[0] > 0:
            h = h + x_dict[node_type]
        
        # Layer norm
        if node_type in self.norms:
            h = self.norms[node_type](h)
        
        # FiLM modulation
        if node_type in self.film_layers:
            h = self.film_layers[node_type](h, condition)
        
        # Dropout
        h = self.dropout(h)
        result[node_type] = h
    
    return result
```

### Conditioning Signal

The condition is constructed from three sources:

```python
# In CricketHeteroGNNFull.forward():
phase_state = data['phase_state'].x      # [batch, 6]
chase_state = data['chase_state'].x      # [batch, 7]
wicket_buffer = data['wicket_buffer'].x  # [batch, 2]
condition = torch.cat([phase_state, chase_state, wicket_buffer], dim=-1)  # [batch, 15]
```

| Component | Features | What It Captures |
|-----------|----------|------------------|
| phase_state (6) | is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over | Match phase |
| chase_state (7) | runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem | Chase pressure |
| wicket_buffer (2) | wickets_in_hand, is_tail | Resource state |

## Building Conv Stacks

**Source**: `src/model/conv_builder.py:522-586`

### Standard Stack

```python
def build_conv_stack(num_layers, hidden_dim, num_heads, dropout):
    return nn.ModuleList([
        HeteroConvBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])
```

### Phase-Modulated Stack

```python
def build_phase_modulated_conv_stack(num_layers, hidden_dim, condition_dim, num_heads, dropout):
    return nn.ModuleList([
        PhaseModulatedConvBlock(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])
```

## Using the Conv Stack in the Model

### Standard Model

```python
class CricketHeteroGNN(nn.Module):
    def __init__(self, config):
        self.conv_stack = build_conv_stack(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
    
    def forward(self, data):
        x_dict = self.encoders.encode_nodes(data)
        edge_index_dict = data.edge_index_dict
        
        # Extract edge attributes
        edge_attr_dict = {}
        for edge_type in edge_index_dict.keys():
            if hasattr(data[edge_type], 'edge_attr'):
                edge_attr_dict[edge_type] = data[edge_type].edge_attr
        
        # Message passing
        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, edge_attr_dict or None)
        
        # Readout and predict
        query_repr = x_dict['query']
        logits = self.predictor(query_repr)
        return logits
```

### Phase-Modulated Model

```python
class CricketHeteroGNNFull(nn.Module):
    def __init__(self, config):
        if config.use_phase_modulation:
            self.conv_stack = build_phase_modulated_conv_stack(
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                condition_dim=config.condition_dim,  # 15
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
    
    def forward(self, data):
        x_dict = self.encoders.encode_nodes(data)
        
        # Construct conditioning signal
        phase_state = data['phase_state'].x
        chase_state = data['chase_state'].x
        wicket_buffer = data['wicket_buffer'].x
        condition = torch.cat([phase_state, chase_state, wicket_buffer], dim=-1)
        
        # Message passing with conditioning
        for conv_block in self.conv_stack:
            x_dict = conv_block(x_dict, edge_index_dict, condition, edge_attr_dict)
        
        # ... readout and predict
```

## Information Flow Visualization

```
Layer 0 (After Encoding):
┌────────────────────────────────────────────────────────┐
│ venue: [128]  batting_team: [128]  bowling_team: [128] │
│ score: [128]  chase: [128]  phase: [128]  ...          │
│ striker_id: [128]  striker_state: [128]  ...           │
│ batting_mom: [128]  bowling_mom: [128]  ...            │
│ ball_0: [128]  ball_1: [128]  ...  ball_n: [128]       │
│ query: [128]                                           │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   HeteroConvBlock #1     │
            │   ├── HeteroConv         │
            │   ├── Residual + Norm    │
            │   └── (FiLM if enabled)  │
            └──────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   HeteroConvBlock #2     │
            └──────────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   HeteroConvBlock #3     │
            └──────────────────────────┘
                          │
                          ▼
Layer 3 (After Message Passing):
┌────────────────────────────────────────────────────────┐
│ All nodes now contain aggregated neighborhood info    │
│                                                        │
│ query: [128]  ← Contains info from entire graph       │
│ striker_id: [128] ← Contains matchup context          │
│ bowler_id: [128] ← Contains spell context             │
└────────────────────────────────────────────────────────┘
```

## Summary

| Component | Purpose |
|-----------|---------|
| HeteroConv | Per-edge-type message aggregation |
| Residual | Gradient flow, identity mapping |
| LayerNorm | Feature normalization per node type |
| FiLM | Phase-conditional modulation |
| Dropout | Regularization |

**Standard Stack**: Good baseline, phase-agnostic
**Phase-Modulated Stack**: Adapts to game situation, better for production

---

*Next: [06-readout-and-prediction.md](./06-readout-and-prediction.md) - Query aggregation and prediction heads*
