# Forward Pass Walkthrough

This document traces data through the `CricketHeteroGNNFull` model step by step.

## Setup

**Scenario**: Predicting the 61st ball of a T20 innings.

```python
# Sample dimensions
batch_size = 1
num_balls = 60  # 60 historical balls (balls 0-59)
hidden_dim = 128
num_heads = 4
num_classes = 7
```

## Step 1: Input Data (HeteroData)

The input is a PyTorch Geometric `HeteroData` object:

```python
data = HeteroData()

# Entity nodes (IDs)
data['venue'].x = tensor([[42]])                     # [1, 1]
data['batting_team'].x = tensor([[7]])               # [1, 1]
data['bowling_team'].x = tensor([[12]])              # [1, 1]
data['striker_identity'].x = tensor([[156]])         # [1, 1]
data['striker_identity'].team_id = tensor([[7]])     # [1, 1]
data['striker_identity'].role_id = tensor([[4]])     # [1, 1] (finisher)
# ... similar for nonstriker_identity, bowler_identity

# State nodes (features)
data['score_state'].x = tensor([[0.48, 0.2, 0.5, 0.0, 0.0]])  # [1, 5]
data['chase_state'].x = tensor([[0.0, 0.0, 0.0, ...]])         # [1, 7]
data['phase_state'].x = tensor([[0.0, 1.0, 0.0, 0.33, 0.0, 0.0]])  # [1, 6]
# ... similar for time_pressure, wicket_buffer

# Actor state nodes
data['striker_state'].x = tensor([[0.42, 0.35, 0.60, ...]])    # [1, 8]
# ... similar for nonstriker_state, bowler_state, partnership

# Dynamics nodes
data['batting_momentum'].x = tensor([[0.15]])         # [1, 1]
data['bowling_momentum'].x = tensor([[-0.15]])        # [1, 1]
data['pressure_index'].x = tensor([[0.25]])           # [1, 1]
data['dot_pressure'].x = tensor([[0.17, 0.25, 0.8, 0.3, 0.1]])  # [1, 5]

# Ball nodes (60 historical balls)
data['ball'].x = tensor([[...], [...], ...])          # [60, 18]
data['ball'].bowler_ids = tensor([45, 45, 45, 67, ...])  # [60]
data['ball'].batsman_ids = tensor([156, 156, 189, ...])  # [60]

# Query node (placeholder)
data['query'].x = tensor([[0.0]])                     # [1, 1]

# Edge indices (many edge types)
data['venue', 'conditions', 'score_state'].edge_index = tensor([[0], [0]])  # [2, 1]
data['ball', 'recent_precedes', 'ball'].edge_index = tensor([[...], [...]])  # [2, ~250]
data['ball', 'recent_precedes', 'ball'].edge_attr = tensor([[...]])  # [~250, 1]
# ... many more edge types

# Labels and metadata
data.y = tensor([1])  # Target: Single
data.is_chase = tensor([False])
```

## Step 2: Node Encoding

```python
x_dict = self.encoders.encode_nodes(data)
```

### Entity Nodes

```python
# venue: ID → Embedding → Projection
venue_id = 42
venue_emb = venue_encoder.embedding(42)      # [32]
venue_hidden = venue_encoder.projection(venue_emb)  # [128]
x_dict['venue'] = venue_hidden.unsqueeze(0)  # [1, 128]

# striker_identity: Hierarchical encoding
player_id = 156  # Known player
# Since player_id != 0, use player embedding (not fallback)
player_emb = player_encoder.player_embed(156)  # [64]
striker_hidden = player_encoder.player_projection(player_emb)  # [128]
x_dict['striker_identity'] = striker_hidden.unsqueeze(0)  # [1, 128]
```

### Feature Nodes

```python
# phase_state: MLP projection
phase_features = tensor([0.0, 1.0, 0.0, 0.33, 0.0, 0.0])  # [6]
phase_hidden = feature_encoders['phase_state'](phase_features)  # [128]
x_dict['phase_state'] = phase_hidden.unsqueeze(0)  # [1, 128]
```

### Ball Nodes

```python
# ball: Features + player embeddings
ball_features = data['ball'].x  # [60, 18]
bowler_ids = data['ball'].bowler_ids  # [60]
batsman_ids = data['ball'].batsman_ids  # [60]

bowler_embs = ball_encoder.bowler_embed(bowler_ids)  # [60, 64]
batsman_embs = ball_encoder.batsman_embed(batsman_ids)  # [60, 64]
combined = torch.cat([ball_features, bowler_embs, batsman_embs], dim=-1)  # [60, 146]
ball_hidden = ball_encoder.projection(combined)  # [60, 128]
x_dict['ball'] = ball_hidden  # [60, 128]
```

### Query Node

```python
# query: Learned parameter (NOT from input)
x_dict['query'] = query_encoder.embedding.expand(1, -1)  # [1, 128]
```

### After Encoding

```python
x_dict = {
    'venue': [1, 128],
    'batting_team': [1, 128],
    'bowling_team': [1, 128],
    'striker_identity': [1, 128],
    'nonstriker_identity': [1, 128],
    'bowler_identity': [1, 128],
    'score_state': [1, 128],
    'chase_state': [1, 128],
    'phase_state': [1, 128],
    'time_pressure': [1, 128],
    'wicket_buffer': [1, 128],
    'striker_state': [1, 128],
    'nonstriker_state': [1, 128],
    'bowler_state': [1, 128],
    'partnership': [1, 128],
    'batting_momentum': [1, 128],
    'bowling_momentum': [1, 128],
    'pressure_index': [1, 128],
    'dot_pressure': [1, 128],
    'ball': [60, 128],
    'query': [1, 128],
}
```

## Step 3: Construct FiLM Condition

```python
phase_state = data['phase_state'].x      # [1, 6]
chase_state = data['chase_state'].x      # [1, 7]
wicket_buffer = data['wicket_buffer'].x  # [1, 2]
condition = torch.cat([phase_state, chase_state, wicket_buffer], dim=-1)  # [1, 15]
```

## Step 4: Message Passing (3 layers)

### Layer 1

```python
# HeteroConv: Apply per-edge convolutions
out_dict = conv.conv(x_dict, edge_index_dict, edge_attr_dict)

# For each edge type, messages flow:
# (venue, conditions, score_state): GATv2Conv
#   venue [1, 128] → score_state receives message
# (ball, recent_precedes, ball): TransformerConv with edge_dim=1
#   ball[i] → ball[j] with temporal distance feature
# (striker_identity, matchup, bowler_identity): GATv2Conv
#   striker ← → bowler (bidirectional matchup)
# ... ~150 edge types total

# After aggregation (sum):
# Each node has received messages from all incoming edges
```

```python
# Residual + LayerNorm
for node_type in out_dict:
    h = out_dict[node_type]
    h = h + x_dict[node_type]  # Residual
    h = norms[node_type](h)     # LayerNorm
```

```python
# FiLM Modulation (per node type)
for node_type in out_dict:
    h = out_dict[node_type]
    film_params = film_layers[node_type].film_generator(condition)  # [1, 256]
    gamma, beta = film_params.chunk(2, dim=-1)  # Each: [1, 128]
    h = gamma * h + beta  # Modulation
```

```python
# Dropout
for node_type in out_dict:
    h = dropout(h)
```

### Layers 2 and 3

Same structure, but now nodes have richer representations from previous layers.

**After 3 layers**: All nodes contain aggregated neighborhood information.

## Step 5: Hybrid Readout

```python
# Extract player representations after message passing
striker = x_dict['striker_identity']      # [1, 128]
bowler = x_dict['bowler_identity']        # [1, 128]
nonstriker = x_dict['nonstriker_identity']  # [1, 128]

# Matchup MLP
matchup_input = torch.cat([striker, bowler], dim=-1)  # [1, 256]
base_matchup = matchup_mlp(matchup_input)  # [1, 128]

# Non-striker gate
ns_gate = nonstriker_gate(nonstriker)  # [1, 128], range [0, 1]
matchup = base_matchup * (1.0 + 0.1 * ns_gate)  # [1, 128]

# Query projection
query = query_proj(x_dict['query'])  # [1, 128]

# Combine matchup + context
combined = torch.cat([matchup, query], dim=-1)  # [1, 256]
combined = combiner(combined)  # [1, 128]
```

## Step 6: Innings-Conditional Prediction

```python
# Get chase info
chase_state = data['chase_state'].x  # [1, 7]
is_chase = data.is_chase  # [1] = False (1st innings)

# Compute both heads
first_logits = first_innings_head(combined)  # [1, 7]

combined_with_chase = torch.cat([combined, chase_state], dim=-1)  # [1, 135]
second_logits = second_innings_head(combined_with_chase)  # [1, 7]

# Select based on innings
# is_chase = False, so use first_innings_head
logits = torch.where(
    is_chase.unsqueeze(-1).expand_as(first_logits),
    second_logits,
    first_logits
)  # [1, 7]
```

## Step 7: Output

```python
logits = tensor([[-0.5, 1.2, 0.3, -0.8, 0.7, -0.2, -1.1]])  # [1, 7]

# To get probabilities:
probs = torch.softmax(logits, dim=-1)
# tensor([[0.08, 0.42, 0.17, 0.06, 0.18, 0.07, 0.02]])

# Prediction:
pred = logits.argmax(dim=-1)  # tensor([1]) = Single
```

## Shape Summary

| Step | Component | Input Shape | Output Shape |
|------|-----------|-------------|--------------|
| 1 | HeteroData | - | Various per node type |
| 2 | EntityEncoder | [B, 1] ID | [B, 128] |
| 2 | FeatureEncoder | [B, D] features | [B, 128] |
| 2 | BallEncoder | [N, 18+IDs] | [N, 128] |
| 2 | QueryEncoder | - | [B, 128] |
| 3 | FiLM condition | [B, 6+7+2] | [B, 15] |
| 4 | HeteroConvBlock | x_dict | x_dict (same shapes) |
| 5 | Matchup MLP | [B, 256] | [B, 128] |
| 5 | NS Gate | [B, 128] | [B, 128] |
| 5 | Query Proj | [B, 128] | [B, 128] |
| 5 | Combiner | [B, 256] | [B, 128] |
| 6 | First Head | [B, 128] | [B, 7] |
| 6 | Second Head | [B, 135] | [B, 7] |
| 7 | Output | - | [B, 7] logits |

## Information Flow Visualization

```
                                    ┌─────────────────┐
                                    │   Raw Input     │
                                    │   (HeteroData)  │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
            ┌───────▼───────┐       ┌────────▼────────┐      ┌────────▼────────┐
            │ Entity IDs    │       │ Feature Vectors │      │ Ball Data       │
            │ [venue: 42]   │       │ [phase: 6 dims] │      │ [60 balls x 18] │
            └───────┬───────┘       └────────┬────────┘      └────────┬────────┘
                    │                        │                        │
            ┌───────▼───────┐       ┌────────▼────────┐      ┌────────▼────────┐
            │ EntityEncoder │       │ FeatureEncoder  │      │ BallEncoder     │
            │ Embed + MLP   │       │ MLP             │      │ Embed + MLP     │
            └───────┬───────┘       └────────┬────────┘      └────────┬────────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │         x_dict               │
                              │  All nodes: [N_type, 128]    │
                              └──────────────┬───────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
            ┌───────────────┐    ┌───────────────────┐    ┌───────────────┐
            │   Layer 1     │    │     Layer 2       │    │    Layer 3    │
            │ HeteroConv    │───▶│   HeteroConv      │───▶│  HeteroConv   │
            │ + Residual    │    │   + Residual      │    │  + Residual   │
            │ + LayerNorm   │    │   + LayerNorm     │    │  + LayerNorm  │
            │ + FiLM        │    │   + FiLM          │    │  + FiLM       │
            └───────────────┘    └───────────────────┘    └───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │     Updated x_dict           │
                              │  (aggregated information)    │
                              └──────────────┬───────────────┘
                                             │
                    ┌─────────────┬──────────┼──────────┬─────────────┐
                    │             │          │          │             │
                    ▼             ▼          ▼          ▼             ▼
            [striker_id]   [bowler_id] [nonstriker] [query]     [chase_state]
                    │             │          │          │             │
                    └──────┬──────┘          │          │             │
                           ▼                 │          │             │
                    ┌─────────────┐          │          │             │
                    │ Matchup MLP │          │          │             │
                    │ [256]→[128] │          │          │             │
                    └──────┬──────┘          │          │             │
                           │      ┌──────────┘          │             │
                           │      ▼                     │             │
                           │ ┌────────────┐             │             │
                           │ │ NS Gate    │             │             │
                           │ │ [128]→[128]│             │             │
                           │ └─────┬──────┘             │             │
                           │       │                    │             │
                           ▼       ▼                    │             │
                    ┌─────────────────────┐             │             │
                    │ Modulated Matchup   │             │             │
                    │ base * (1+0.1*gate) │             │             │
                    └──────────┬──────────┘             │             │
                               │                        │             │
                               │     ┌──────────────────┘             │
                               │     ▼                                │
                               │  ┌────────────┐                      │
                               │  │ Query Proj │                      │
                               │  │ [128]→[128]│                      │
                               │  └─────┬──────┘                      │
                               │        │                             │
                               └────┬───┘                             │
                                    ▼                                 │
                            ┌───────────────┐                         │
                            │   Combiner    │                         │
                            │ [256]→[128]   │                         │
                            └───────┬───────┘                         │
                                    │                                 │
                    ┌───────────────┴───────────────┐                 │
                    │                               │                 │
                    ▼                               ▼                 │
            ┌───────────────┐               ┌──────────────┐          │
            │ 1st Innings   │               │ 2nd Innings  │◀─────────┘
            │ Head          │               │ Head         │ (+ chase_state)
            │ [128]→[7]     │               │ [135]→[7]    │
            └───────┬───────┘               └──────┬───────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼ (select by is_chase)
                            ┌───────────────┐
                            │    Logits     │
                            │    [B, 7]     │
                            └───────────────┘
```

---

*Back to: [INDEX.md](./INDEX.md)*
