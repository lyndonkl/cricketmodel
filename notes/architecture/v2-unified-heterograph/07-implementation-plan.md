# Implementation Plan

## Critical Design Decision: Node Type Granularity

### The Problem

In PyTorch Geometric's `HeteroData`, all nodes of the same type must have the same feature dimension. However, our design has:

- **State nodes**: score_state (4 features), chase_state (3), phase_state (4), time_pressure (3), wicket_buffer (2)
- **Actor nodes**: striker_id (entity ID), striker_state (6 features), bowler_id (entity ID), bowler_state (6), partnership (4)

These have **different dimensions** within the same semantic group!

### Solution: Fine-Grained Node Types

Use **19 separate node types**, one per semantic node:

```
Node Types (19 total):
├── Entity nodes (5 types, 1 node each):
│   ├── venue
│   ├── batting_team
│   ├── bowling_team
│   ├── striker_identity
│   └── bowler_identity
│
├── Feature nodes (12 types, 1 node each):
│   ├── score_state (4 features)
│   ├── chase_state (3 features)
│   ├── phase_state (4 features)
│   ├── time_pressure (3 features)
│   ├── wicket_buffer (2 features)
│   ├── striker_state (6 features)
│   ├── bowler_state (6 features)
│   ├── partnership (4 features)
│   ├── batting_momentum (1 feature)
│   ├── bowling_momentum (1 feature)
│   ├── pressure_index (1 feature)
│   └── dot_pressure (2 features)
│
├── ball (N nodes, 5 features + player IDs)
│
└── query (1 node)
```

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Fine-grained (19 types) | Clean semantics, type-safe | Many edge type combinations |
| Grouped (6 types) | Fewer types | Complex internal handling |
| Single context type | Simple | Loses structure |

**Decision: Use fine-grained types** for semantic clarity. We'll manage edge type complexity through helper functions.

---

## Implementation Phases

### Phase 1: Project Setup

**Files to create:**
```
src/
├── __init__.py
├── config.py                 # Configuration dataclass
├── data/
│   ├── __init__.py
│   ├── entity_mapper.py      # Entity name → ID mapping
│   ├── feature_utils.py      # Feature computation helpers
│   ├── edge_builder.py       # Edge construction helpers
│   ├── hetero_data_builder.py # HeteroData construction
│   └── dataset.py            # CricketDataset class
├── model/
│   ├── __init__.py
│   ├── encoders.py           # Node encoders
│   ├── conv_builder.py       # HeteroConv construction
│   └── hetero_gnn.py         # Main model
└── training/
    ├── __init__.py
    ├── trainer.py            # Training loop
    └── metrics.py            # Evaluation metrics
```

**Dependencies:**
```python
# pyproject.toml additions
torch >= 2.0
torch-geometric >= 2.4
torch-scatter
torch-sparse
```

---

### Phase 2: Data Layer (src/data/)

#### Step 2.1: Entity Mapper (`entity_mapper.py`)

```python
class EntityMapper:
    """Bidirectional mapping between entity names and integer IDs."""

    def __init__(self):
        self.venue_to_id = {}
        self.team_to_id = {}
        self.player_to_id = {}

    # Methods:
    # - get_venue_id(name) -> int
    # - get_team_id(name) -> int
    # - get_player_id(name) -> int
    # - save(path) / load(path) for persistence
```

**Key consideration:** Build mapper from ALL data before splitting to ensure consistent IDs.

#### Step 2.2: Feature Utils (`feature_utils.py`)

```python
def compute_score_state(deliveries, innings_num) -> List[float]:
    """Returns [runs/250, wickets/10, balls/120, innings]"""

def compute_chase_state(deliveries, target, innings_num) -> List[float]:
    """Returns [runs_needed/250, RRR/20, is_chase]"""

def compute_phase_state(ball_number) -> List[float]:
    """Returns [is_pp, is_middle, is_death, over_progress]"""

def compute_time_pressure(balls_bowled, target_if_chase) -> List[float]:
    """Returns [balls_left/120, urgency, is_final_over]"""

def compute_wicket_buffer(wickets) -> List[float]:
    """Returns [wickets_in_hand/10, is_tail]"""

def compute_striker_state(deliveries, striker_name) -> List[float]:
    """Returns [runs/100, balls/60, sr/200, dots_pct, is_set, bdry/10]"""

def compute_bowler_state(deliveries, bowler_name) -> List[float]:
    """Returns [balls/24, runs/50, wkts/5, econ/15, dots_pct, threat]"""

def compute_partnership(deliveries, striker, non_striker) -> List[float]:
    """Returns [runs/100, balls/60, rr/10, stability]"""

def compute_dynamics(deliveries) -> Dict[str, List[float]]:
    """Returns dict with batting_mom, bowling_mom, pressure, dot_pressure"""

def compute_ball_features(delivery) -> List[float]:
    """Returns [runs/6, is_wicket, over/20, ball_in_over/6, is_boundary]"""

def outcome_to_class(delivery) -> int:
    """Returns 0-6 for dot/1/2/3/4/6/wicket"""
```

#### Step 2.3: Edge Builder (`edge_builder.py`)

```python
# Constants defining the graph structure
NODE_TYPE_TO_LAYER = {
    'venue': 'global', 'batting_team': 'global', 'bowling_team': 'global',
    'score_state': 'state', 'chase_state': 'state', ...
}

HIERARCHICAL_EDGES = [
    # (source_type, target_type) pairs for global→state, state→actor, actor→dynamics
    ('venue', 'score_state'), ('venue', 'chase_state'), ...
    ('batting_team', 'score_state'), ...
]

INTRA_LAYER_EDGES = {
    'global': [('venue', 'batting_team'), ('venue', 'bowling_team'), ...],
    'state': [('score_state', 'chase_state'), ...],
    'actor': [
        ('striker_identity', 'striker_state'),
        ('bowler_identity', 'bowler_state'),
        ('striker_identity', 'bowler_identity'),  # THE MATCHUP
        ('striker_state', 'partnership'),
        ('bowler_state', 'partnership'),
    ],
    'dynamics': [('batting_momentum', 'bowling_momentum'), ...],
}

def build_hierarchical_edges() -> Dict[Tuple, torch.Tensor]:
    """Build all global→state→actor→dynamics conditioning edges."""

def build_intra_layer_edges() -> Dict[Tuple, torch.Tensor]:
    """Build within-layer interaction edges."""

def build_temporal_edges(num_balls, bowler_ids, batsman_ids) -> Dict[Tuple, torch.Tensor]:
    """Build precedes, same_bowler, same_batsman edges."""

def build_cross_domain_edges(num_balls) -> Dict[Tuple, torch.Tensor]:
    """Build ball→actor and ball→dynamics edges."""

def build_query_edges(num_balls) -> Dict[Tuple, torch.Tensor]:
    """Build query→everything edges."""
```

**Key insight:** Pre-define edge structures as constants since context graph structure is fixed.

#### Step 2.4: HeteroData Builder (`hetero_data_builder.py`)

```python
from torch_geometric.data import HeteroData

def create_hetero_data(
    match_data: Dict,
    innings_idx: int,
    ball_idx: int,
    entity_mapper: EntityMapper
) -> HeteroData:
    """
    Create a HeteroData object for predicting ball at ball_idx.

    This is the core function that constructs the unified graph.
    """
    data = HeteroData()

    # 1. Extract context
    innings = match_data['innings'][innings_idx]
    deliveries = innings['deliveries'][:ball_idx]  # History
    current_ball = innings['deliveries'][ball_idx]  # To predict

    # 2. Build entity nodes (venue, teams, players)
    data['venue'].x = torch.tensor([[entity_mapper.get_venue_id(...)]])
    data['batting_team'].x = torch.tensor([[entity_mapper.get_team_id(...)]])
    # ... etc

    # 3. Build feature nodes (state, actor, dynamics)
    data['score_state'].x = torch.tensor([compute_score_state(...)])
    data['chase_state'].x = torch.tensor([compute_chase_state(...)])
    # ... etc

    # 4. Build ball nodes
    if len(deliveries) > 0:
        ball_features = [compute_ball_features(d) for d in deliveries]
        data['ball'].x = torch.tensor(ball_features)
        data['ball'].bowler_ids = torch.tensor([...])
        data['ball'].batsman_ids = torch.tensor([...])

    # 5. Build query node
    data['query'].x = torch.zeros(1, 1)  # Placeholder

    # 6. Build all edges
    data = add_all_edges(data, len(deliveries), bowler_ids, batsman_ids)

    # 7. Add label
    data.y = torch.tensor([outcome_to_class(current_ball)])

    return data
```

#### Step 2.5: Dataset (`dataset.py`)

```python
from torch_geometric.data import InMemoryDataset

class CricketDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for cricket ball prediction.

    Each sample is a HeteroData object representing the match state
    before a specific ball, with the label being that ball's outcome.
    """

    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['matches.json', 'entity_mapper.pkl']

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        # 1. Load matches and entity mapper
        # 2. Split by match (not by ball!) to avoid leakage
        # 3. Create HeteroData for each ball in split
        # 4. Save using self.save()
```

**Critical:** Split by MATCH, not by ball, to avoid data leakage.

---

### Phase 3: Model Layer (src/model/)

#### Step 3.1: Encoders (`encoders.py`)

```python
class EntityEncoder(nn.Module):
    """Encodes entity IDs to hidden representations."""
    def __init__(self, num_entities, embed_dim, hidden_dim):
        self.embedding = nn.Embedding(num_entities + 1, embed_dim)  # +1 for unknown
        self.projection = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x):
        # x: [num_nodes, 1] of entity IDs
        emb = self.embedding(x.squeeze(-1))  # [num_nodes, embed_dim]
        return self.projection(emb)  # [num_nodes, hidden_dim]


class FeatureEncoder(nn.Module):
    """Encodes numeric features to hidden representations."""
    def __init__(self, input_dim, hidden_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class BallEncoder(nn.Module):
    """Encodes ball nodes with both features and player embeddings."""
    def __init__(self, num_players, player_embed_dim, feature_dim, hidden_dim):
        self.bowler_embed = nn.Embedding(num_players + 1, player_embed_dim)
        self.batsman_embed = nn.Embedding(num_players + 1, player_embed_dim)

        concat_dim = 2 * player_embed_dim + feature_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, bowler_ids, batsman_ids):
        bowler_emb = self.bowler_embed(bowler_ids)
        batsman_emb = self.batsman_embed(batsman_ids)
        combined = torch.cat([x, bowler_emb, batsman_emb], dim=-1)
        return self.projection(combined)


class QueryEncoder(nn.Module):
    """Provides learned query embedding."""
    def __init__(self, hidden_dim):
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self, batch_size):
        return self.embedding.expand(batch_size, -1)
```

#### Step 3.2: Conv Builder (`conv_builder.py`)

```python
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, TransformerConv

# Define all edge types
EDGE_TYPES = [
    # Hierarchical (global → state)
    ('venue', 'conditions', 'score_state'),
    ('venue', 'conditions', 'chase_state'),
    # ... many more

    # Intra-layer
    ('venue', 'relates_to', 'batting_team'),
    # ... etc

    # Actor matchup
    ('striker_identity', 'matchup', 'bowler_identity'),
    # ... etc

    # Temporal
    ('ball', 'precedes', 'ball'),
    ('ball', 'same_bowler', 'ball'),
    ('ball', 'same_batsman', 'ball'),

    # Cross-domain
    ('ball', 'bowled_by', 'striker_identity'),  # Note: bowled_by goes to current bowler
    ('ball', 'faced_by', 'striker_identity'),
    ('ball', 'informs', 'batting_momentum'),
    # ... etc

    # Query edges
    ('venue', 'attended_by', 'query'),  # Reversed: info flows TO query
    # ... etc
]

def build_hetero_conv(hidden_dim, num_heads=4):
    """Build a HeteroConv layer with appropriate convolutions per edge type."""
    convs = {}

    for edge_type in EDGE_TYPES:
        src, rel, dst = edge_type

        if rel in ['conditions', 'relates_to', 'matchup', 'attended_by']:
            # Use attention for these
            convs[edge_type] = GATv2Conv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, add_self_loops=False,
                concat=True  # Output is hidden_dim
            )
        elif rel == 'precedes':
            # Temporal with edge features
            convs[edge_type] = TransformerConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, concat=True
            )
        elif rel in ['same_bowler', 'same_batsman']:
            # Attention for actor aggregation
            convs[edge_type] = GATv2Conv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, add_self_loops=False,
                concat=True
            )
        else:
            # Simple aggregation for cross-domain
            convs[edge_type] = SAGEConv(hidden_dim, hidden_dim)

    return HeteroConv(convs, aggr='sum')
```

**Important:** GATv2Conv with `concat=True` and `heads=4` outputs `4 * (hidden_dim // 4) = hidden_dim`.

#### Step 3.3: Main Model (`hetero_gnn.py`)

```python
class CricketHeteroGNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.hidden_dim

        # === Node Encoders ===
        # Entity encoders
        self.venue_encoder = EntityEncoder(config.num_venues, 32, hidden_dim)
        self.team_encoder = EntityEncoder(config.num_teams, 32, hidden_dim)
        self.player_encoder = EntityEncoder(config.num_players, 64, hidden_dim)

        # Feature encoders (one per node type with features)
        self.feature_encoders = nn.ModuleDict({
            'score_state': FeatureEncoder(4, hidden_dim),
            'chase_state': FeatureEncoder(3, hidden_dim),
            'phase_state': FeatureEncoder(4, hidden_dim),
            'time_pressure': FeatureEncoder(3, hidden_dim),
            'wicket_buffer': FeatureEncoder(2, hidden_dim),
            'striker_state': FeatureEncoder(6, hidden_dim),
            'bowler_state': FeatureEncoder(6, hidden_dim),
            'partnership': FeatureEncoder(4, hidden_dim),
            'batting_momentum': FeatureEncoder(1, hidden_dim),
            'bowling_momentum': FeatureEncoder(1, hidden_dim),
            'pressure_index': FeatureEncoder(1, hidden_dim),
            'dot_pressure': FeatureEncoder(2, hidden_dim),
        })

        # Ball encoder
        self.ball_encoder = BallEncoder(
            config.num_players, 32, 5, hidden_dim
        )

        # Query encoder
        self.query_encoder = QueryEncoder(hidden_dim)

        # === Message Passing ===
        self.convs = nn.ModuleList([
            build_hetero_conv(hidden_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])

        # Layer norms per node type
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in ALL_NODE_TYPES
            })
            for _ in range(config.num_layers)
        ])

        # === Prediction Head ===
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.num_classes)
        )

        self.dropout = nn.Dropout(config.dropout)

    def encode_nodes(self, data):
        """Encode all node features to hidden_dim."""
        x_dict = {}

        # Entity nodes
        x_dict['venue'] = self.venue_encoder(data['venue'].x)
        x_dict['batting_team'] = self.team_encoder(data['batting_team'].x)
        x_dict['bowling_team'] = self.team_encoder(data['bowling_team'].x)
        x_dict['striker_identity'] = self.player_encoder(data['striker_identity'].x)
        x_dict['bowler_identity'] = self.player_encoder(data['bowler_identity'].x)

        # Feature nodes
        for node_type, encoder in self.feature_encoders.items():
            x_dict[node_type] = encoder(data[node_type].x)

        # Ball nodes
        if 'ball' in data.node_types and data['ball'].num_nodes > 0:
            x_dict['ball'] = self.ball_encoder(
                data['ball'].x,
                data['ball'].bowler_ids,
                data['ball'].batsman_ids
            )

        # Query node
        batch_size = data['venue'].x.size(0)  # Each sample has 1 venue node
        x_dict['query'] = self.query_encoder(batch_size)

        return x_dict

    def forward(self, data):
        # 1. Encode nodes
        x_dict = self.encode_nodes(data)

        # 2. Message passing
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x_dict_new = conv(x_dict, data.edge_index_dict)

            # Residual + norm + dropout
            for node_type in x_dict_new:
                if node_type in x_dict:
                    x_dict_new[node_type] = x_dict_new[node_type] + x_dict[node_type]
                x_dict_new[node_type] = self.norms[i][node_type](x_dict_new[node_type])
                x_dict_new[node_type] = self.dropout(x_dict_new[node_type])

            x_dict = x_dict_new

        # 3. Readout from query
        query_repr = x_dict['query']  # [batch_size, hidden_dim]

        # 4. Predict
        return self.predictor(query_repr)
```

---

### Phase 4: Training Layer (src/training/)

#### Step 4.1: Trainer (`trainer.py`)

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss with class weights
        self.criterion = nn.CrossEntropyLoss(weight=config.class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        # ... similar to train_epoch but without backward
```

---

### Phase 5: Testing & Validation

#### Step 5.1: Unit Tests

```python
# tests/test_hetero_data.py
def test_hetero_data_creation():
    """Test that HeteroData is correctly constructed."""
    data = create_hetero_data(sample_match, 0, 10, entity_mapper)

    # Check node types exist
    assert 'venue' in data.node_types
    assert 'ball' in data.node_types
    assert 'query' in data.node_types

    # Check shapes
    assert data['venue'].x.shape == (1, 1)
    assert data['score_state'].x.shape == (1, 4)

    # Check edges exist
    assert ('venue', 'conditions', 'score_state') in data.edge_types

# tests/test_model.py
def test_model_forward():
    """Test model can process a batch."""
    model = CricketHeteroGNN(config)
    batch = create_sample_batch()

    logits = model(batch)
    assert logits.shape == (batch_size, 7)

def test_model_backward():
    """Test gradients flow correctly."""
    model = CricketHeteroGNN(config)
    batch = create_sample_batch()

    logits = model(batch)
    loss = F.cross_entropy(logits, batch.y)
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

#### Step 5.2: Sanity Checks

1. **Overfit on single batch**: Model should achieve ~100% accuracy
2. **Random labels**: Model should achieve ~14% accuracy (1/7)
3. **Check attention weights**: Should be non-uniform after training

---

## Implementation Order

```
Week 1: Data Layer
├── Day 1-2: entity_mapper.py, feature_utils.py
├── Day 3-4: edge_builder.py, hetero_data_builder.py
└── Day 5: dataset.py, unit tests

Week 2: Model Layer
├── Day 1-2: encoders.py
├── Day 3-4: conv_builder.py, hetero_gnn.py
└── Day 5: Unit tests, sanity checks

Week 3: Training & Evaluation
├── Day 1-2: trainer.py, metrics.py
├── Day 3-4: train.py (main script)
└── Day 5: Full training run, debugging
```

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Too many edge types (explosion) | Pre-compute edge structures as constants |
| Memory issues with large graphs | Use mini-batching, monitor memory |
| Slow edge construction | Vectorize edge building, cache fixed edges |
| GATv2Conv dimension issues | Verify concat=True gives correct output dim |
| Batching heterogeneous graphs | Use PyG's automatic batching, test thoroughly |

---

## Open Questions

1. **Edge direction for query**: Should edges be `(node, attends, query)` or `(query, attends, node)`?
   - PyG convention: messages flow from source to target
   - For query to aggregate: edges should be `(source, rel, query)`

2. **Same-bowler/batsman clique size**: Full clique is O(n²) per actor. Consider:
   - K-nearest temporal neighbors instead of full clique
   - Or accept quadratic edges (still sparse globally)

3. **Ball nodes with no history**: First ball of innings has 0 history
   - Handle gracefully with empty ball node type
   - Or require minimum 1 ball of history (skip first ball)
