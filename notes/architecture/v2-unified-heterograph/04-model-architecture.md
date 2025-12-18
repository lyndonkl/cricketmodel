# Model Architecture

## Overview

The model is a **Heterogeneous Graph Neural Network** implemented using PyTorch Geometric. It consists of three main stages:

1. **Node Encoding**: Convert raw features to hidden representations
2. **Message Passing**: Multiple layers of HeteroConv
3. **Readout**: Extract query node representation → predict

```
Raw Features → Node Encoders → HeteroConv Layers → Query Readout → Prediction
```

---

## 1. Node Encoding Layer

Each node type has its own encoder to project features to `hidden_dim` (128).

### 1.1 Entity Encoders (Learned Embeddings)

For nodes representing entities (venue, teams):

```python
class EntityEncoder(nn.Module):
    def __init__(self, num_entities, embed_dim, hidden_dim):
        self.embedding = nn.Embedding(num_entities + 1, embed_dim)  # +1 for unknown
        self.projection = nn.Linear(embed_dim, hidden_dim)

    def forward(self, entity_ids):
        x = self.embedding(entity_ids)
        return self.projection(x)
```

| Node | num_entities | embed_dim | hidden_dim |
|------|--------------|-----------|------------|
| venue | ~50 | 32 | 128 |
| batting_team | ~20 | 32 | 128 |
| bowling_team | ~20 | 32 | 128 |

### 1.1.1 Hierarchical Player Encoder (Cold-Start Handling)

For player identity nodes (striker, non-striker, bowler), we use a hierarchical encoder that handles unknown players:

```python
class HierarchicalPlayerEncoder(nn.Module):
    def __init__(self, num_players, num_teams, num_roles,
                 player_embed_dim=64, team_embed_dim=32, role_embed_dim=16,
                 hidden_dim=128, dropout=0.1):
        self.player_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(num_teams + 1, team_embed_dim, padding_idx=0)
        self.role_embed = nn.Embedding(num_roles + 1, role_embed_dim, padding_idx=0)

        # Separate projections for known vs unknown players
        self.known_proj = nn.Linear(player_embed_dim, hidden_dim)
        self.fallback_proj = nn.Linear(team_embed_dim + role_embed_dim, hidden_dim)

    def forward(self, player_ids, team_ids, role_ids):
        player_emb = self.player_embed(player_ids)
        team_emb = self.team_embed(team_ids)
        role_emb = self.role_embed(role_ids)

        known_output = self.known_proj(player_emb)
        fallback_output = self.fallback_proj(torch.cat([team_emb, role_emb], dim=-1))

        # Use player embedding for known, team+role for unknown (player_id=0)
        is_unknown = (player_ids == 0).unsqueeze(-1)
        return torch.where(is_unknown, fallback_output, known_output)
```

**Why Hierarchical?** Unknown players (debutants, obscure players) all map to player_id=0. Without hierarchical fallback, they're indistinguishable. With team+role embeddings:
- "Unknown opener for India" differs from "Unknown finisher for Australia"
- Model can leverage team batting style and role position

**Role Categories:** unknown, opener, top_order, middle_order, finisher, bowler, allrounder, keeper

| Node | Embedding Dims | Hidden Dim |
|------|---------------|------------|
| striker_identity | 64 (player) + 32 (team) + 16 (role) | 128 |
| nonstriker_identity | 64 (player) + 32 (team) + 16 (role) | 128 |
| bowler_identity | 64 (player) + 32 (team) + 16 (role) | 128 |

### 1.2 Feature Encoders (MLP Projection)

For nodes with numeric features:

```python
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features):
        return self.encoder(features)
```

| Node | input_dim | hidden_dim |
|------|-----------|------------|
| score_state | 4 | 128 |
| chase_state | 3 | 128 |
| phase_state | 5 | 128 |
| time_pressure | 3 | 128 |
| wicket_buffer | 2 | 128 |
| striker_state | 7 | 128 |
| nonstriker_state | 7 | 128 |
| bowler_state | 6 | 128 |
| partnership | 4 | 128 |
| batting_momentum | 1 | 128 |
| bowling_momentum | 1 | 128 |
| pressure_index | 1 | 128 |
| dot_pressure | 2 | 128 |

### 1.3 Ball Encoder (Hybrid)

Balls have both embeddings (bowler, batsman) and features (17 dimensions including run-out attribution):

```python
class BallEncoder(nn.Module):
    def __init__(self, num_players, player_embed_dim=32, feature_dim=17, hidden_dim=128):
        self.bowler_embed = nn.Embedding(num_players + 1, player_embed_dim)
        self.batsman_embed = nn.Embedding(num_players + 1, player_embed_dim)
        self.feature_proj = nn.Linear(feature_dim, hidden_dim - 2 * player_embed_dim)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, ball_features, bowler_ids, batsman_ids):
        bowler_emb = self.bowler_embed(bowler_ids)      # [N, 32]
        batsman_emb = self.batsman_embed(batsman_ids)   # [N, 32]
        feat_proj = self.feature_proj(ball_features)    # [N, 64]
        combined = torch.cat([bowler_emb, batsman_emb, feat_proj], dim=-1)  # [N, 128]
        return self.final_proj(combined)
```

**Ball features (17 dimensions):** runs, is_wicket, over, ball_in_over, is_boundary, is_wide, is_noball, is_bye, is_legbye, wicket_bowled, wicket_caught, wicket_lbw, wicket_run_out, wicket_stumped, wicket_other, **striker_run_out**, **nonstriker_run_out**

### 1.4 Query Encoder

The query node uses a learned embedding:

```python
class QueryEncoder(nn.Module):
    def __init__(self, hidden_dim):
        self.query_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)

    def forward(self, batch_size):
        return self.query_embedding.expand(batch_size, -1)
```

---

## 2. Heterogeneous Message Passing

### 2.1 HeteroConv Layer

Each layer uses `HeteroConv` to apply different convolutions per edge type:

```python
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, TransformerConv

def build_hetero_conv(hidden_dim, num_heads=4, edge_types=None):
    """Build a single HeteroConv layer with edge-type-specific operators."""

    convs = {}

    # Hierarchical conditioning: use attention
    for edge_type in [('global', 'conditions', 'state'),
                      ('state', 'conditions', 'actor'),
                      ('actor', 'conditions', 'dynamics')]:
        convs[edge_type] = GATv2Conv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, add_self_loops=False
        )

    # Intra-layer: use attention for flexibility
    for edge_type in [('global', 'relates_to', 'global'),
                      ('state', 'relates_to', 'state'),
                      ('dynamics', 'relates_to', 'dynamics')]:
        convs[edge_type] = GATv2Conv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, add_self_loops=True
        )

    # Actor matchup: attention on the key edges
    convs[('actor', 'matchup', 'actor')] = GATv2Conv(
        hidden_dim, hidden_dim // num_heads,
        heads=num_heads, add_self_loops=False
    )

    # Temporal edges: attention with edge features for position
    convs[('ball', 'precedes', 'ball')] = TransformerConv(
        hidden_dim, hidden_dim // num_heads,
        heads=num_heads, edge_dim=1  # temporal distance as edge feature
    )

    # Same-actor edges: TransformerConv with temporal decay edge features
    convs[('ball', 'same_bowler', 'ball')] = TransformerConv(
        hidden_dim, hidden_dim // num_heads,
        heads=num_heads, edge_dim=1  # 4-over (24 ball) decay window
    )
    convs[('ball', 'same_batsman', 'ball')] = TransformerConv(
        hidden_dim, hidden_dim // num_heads,
        heads=num_heads, edge_dim=1  # 10-over (60 ball) decay window
    )

    # Same-over edges: within-over context (max 6 balls)
    convs[('ball', 'same_over', 'ball')] = GATv2Conv(
        hidden_dim, hidden_dim // num_heads,
        heads=num_heads, add_self_loops=False
    )

    # Cross-domain: ball to actor connections
    convs[('ball', 'bowled_by', 'actor')] = SAGEConv(hidden_dim, hidden_dim)
    convs[('ball', 'faced_by', 'actor')] = SAGEConv(hidden_dim, hidden_dim)

    # Dynamics reflects recent balls
    convs[('ball', 'informs', 'dynamics')] = SAGEConv(hidden_dim, hidden_dim)

    # Query attends to everything
    for target_type in ['global', 'state', 'actor', 'dynamics', 'ball']:
        convs[('query', 'attends_to', target_type)] = GATv2Conv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, add_self_loops=False
        )

    return HeteroConv(convs, aggr='sum')
```

### 2.2 Full HeteroGNN Model

```python
class CricketHeteroGNN(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_venues=50,
        num_teams=20,
        num_players=500,
        num_classes=7,
        dropout=0.1
    ):
        super().__init__()

        # Node encoders
        self.encoders = nn.ModuleDict({
            'venue': EntityEncoder(num_venues, 32, hidden_dim),
            'batting_team': EntityEncoder(num_teams, 32, hidden_dim),
            'bowling_team': EntityEncoder(num_teams, 32, hidden_dim),
            'score_state': FeatureEncoder(4, hidden_dim),
            'chase_state': FeatureEncoder(3, hidden_dim),
            'phase_state': FeatureEncoder(4, hidden_dim),
            'time_pressure': FeatureEncoder(3, hidden_dim),
            'wicket_buffer': FeatureEncoder(2, hidden_dim),
            'striker_identity': EntityEncoder(num_players, 64, hidden_dim),
            'striker_state': FeatureEncoder(6, hidden_dim),
            'bowler_identity': EntityEncoder(num_players, 64, hidden_dim),
            'bowler_state': FeatureEncoder(6, hidden_dim),
            'partnership': FeatureEncoder(4, hidden_dim),
            'batting_momentum': FeatureEncoder(1, hidden_dim),
            'bowling_momentum': FeatureEncoder(1, hidden_dim),
            'pressure_index': FeatureEncoder(1, hidden_dim),
            'dot_pressure': FeatureEncoder(2, hidden_dim),
            'ball': BallEncoder(num_players, 32, 5, hidden_dim),
            'query': QueryEncoder(hidden_dim),
        })

        # Message passing layers
        self.convs = nn.ModuleList([
            build_hetero_conv(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Layer norms per node type
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in ['global', 'state', 'actor', 'dynamics', 'ball', 'query']
            })
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        """
        Args:
            data: HeteroData object with all node features and edge indices

        Returns:
            logits: [batch_size, num_classes]
        """
        # 1. Encode all nodes
        x_dict = {}
        for node_type, encoder in self.encoders.items():
            if node_type in data.node_types:
                x_dict[node_type] = encoder(data[node_type].x)

        # 2. Message passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)

            # Apply norm and residual
            for node_type in x_dict:
                x_dict[node_type] = self.norms[i][node_type](x_dict[node_type])
                x_dict[node_type] = self.dropout(x_dict[node_type])

        # 3. Readout from query node
        query_repr = x_dict['query']  # [batch_size, hidden_dim]

        # 4. Predict
        logits = self.predictor(query_repr)

        return logits
```

---

## 3. Convolution Choices

### Why Different Convolutions for Different Edges?

| Edge Type | Convolution | Rationale |
|-----------|-------------|-----------|
| Hierarchical | GATv2Conv | Attention learns which context features matter |
| Intra-layer | GATv2Conv | Self-attention for permutation equivariance |
| Actor matchup | GATv2Conv | Learn matchup dynamics with attention |
| Temporal (precedes) | TransformerConv | Edge features for temporal distance |
| Same-bowler/batsman | TransformerConv | Temporal decay edge features (4/10-over windows) |
| Same-over | GATv2Conv | Within-over context patterns |
| Cross-domain | GATv2Conv | Attention-weighted recency |
| Query | GATv2Conv | Learn what to attend to for prediction |

### GATv2Conv vs GATConv

We use **GATv2Conv** (from "How Attentive are Graph Attention Networks?") because:
- Dynamic attention: attention depends on both source AND target
- More expressive than original GAT
- Better at learning complex attention patterns

### TransformerConv for Temporal

`TransformerConv` supports edge features, allowing us to encode temporal distance:
- Edge feature = (target_ball_idx - source_ball_idx) / max_distance
- Model can learn that closer balls are more relevant

---

## 4. Handling Variable Graph Sizes

### Per-Sample Graphs

Each prediction has a different graph:
- Different number of ball nodes (1 to 120)
- Different temporal edges
- Different same-bowler/same-batsman edges

### Batching with PyG

PyTorch Geometric handles this via `Batch.from_data_list()`:
- Combines multiple graphs into one disconnected graph
- Maintains batch assignment for each node
- Automatically handles variable sizes

```python
from torch_geometric.loader import DataLoader

# Each sample is a HeteroData object
dataset = CricketDataset(...)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch is a batched HeteroData
    logits = model(batch)  # [32, 7]
```

---

## 5. Model Configuration

### Default Configuration

```python
config = {
    'hidden_dim': 128,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'num_venues': 50,
    'num_teams': 20,
    'num_players': 500,
    'num_classes': 7,
}
```

### Parameter Count Estimate

| Component | Parameters |
|-----------|------------|
| Entity embeddings | ~50K |
| Feature encoders | ~100K |
| Ball encoder | ~50K |
| HeteroConv layers (×3) | ~500K |
| Predictor | ~20K |
| **Total** | **~720K** |

Similar to V1, but with more structured computation.

---

## 6. Training Considerations

### Loss Function

Same as V1: Cross-entropy with optional class weights for imbalanced outcomes.

```python
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Optimizer

AdamW with learning rate scheduling:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Regularization

- Dropout in message passing layers
- Layer normalization
- Weight decay in optimizer

---

## 7. Advanced Model Variants

The implementation includes several model classes with different capabilities:

### 7.1 Phase-Modulated Message Passing (FiLM)

Different match phases (powerplay, middle, death) have fundamentally different dynamics. FiLM (Feature-wise Linear Modulation) conditions message passing on the current phase:

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for phase conditioning."""
    def __init__(self, condition_dim, hidden_dim):
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma and beta
        )

    def forward(self, x, condition):
        film_params = self.film_generator(condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * x + beta  # Affine transformation
```

**PhaseModulatedConvBlock** wraps each HeteroConv layer with FiLM modulation:
- Phase features (5-dim) are expanded to gamma/beta parameters
- After each conv layer: `output = gamma * conv_output + beta`
- This allows the model to learn phase-specific message passing patterns

### 7.2 Innings-Conditional Prediction Heads

First innings (setting a target) and second innings (chasing) are fundamentally different prediction tasks:

```python
class CricketHeteroGNNInningsConditional(CricketHeteroGNN):
    def __init__(self, config):
        super().__init__(config)
        # Separate prediction heads
        self.first_innings_head = nn.Sequential(...)  # Standard prediction
        self.second_innings_head = nn.Sequential(...)  # Includes chase state injection

    def forward(self, data):
        x_dict = self.encode_and_propagate(data)

        if data.is_chase:
            # Inject chase state into prediction
            chase_features = data['chase_state'].x
            combined = torch.cat([matchup_repr, query_repr, chase_features], dim=-1)
            return self.second_innings_head(combined)
        else:
            return self.first_innings_head(torch.cat([matchup_repr, query_repr], dim=-1))
```

### 7.3 CricketHeteroGNNFull (Production Model)

Combines all enhancements:
- Hierarchical player embeddings (cold-start handling)
- Hybrid matchup + query readout
- Phase-modulated message passing (FiLM)
- Innings-conditional prediction heads

Use `CricketHeteroGNNFull` with all config flags enabled for production.

---

## 8. Interpretability

### Attention Weights

Each GATv2Conv layer produces attention weights that can be extracted:

```python
# After forward pass
for edge_type, conv in model.convs[0].convs.items():
    if hasattr(conv, 'attention_weights'):
        print(f"{edge_type}: {conv.attention_weights.shape}")
```

### Edge Importance

For any prediction, we can see:
- Which global nodes influenced state most
- Which historical balls the query attended to
- Which actor matchup edges had highest attention
- How phase modulation (gamma/beta) varied across game phases

This provides explainability that V1's separate streams lacked.
