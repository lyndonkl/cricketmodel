# Node Encoders

## Overview

The encoder layer transforms raw node features into a common `hidden_dim` embedding space. This is essential because:
- Entity nodes have IDs that need embedding lookup
- Feature nodes have varying dimensions (1 to 18 features)
- Ball nodes combine numeric features with player embeddings
- Query node needs a learned initial representation

```
Raw Input                    Encoder                     Output
─────────────────────────────────────────────────────────────────
venue_id: 42         →  EntityEncoder        →  [hidden_dim]
phase_state: [6 floats] →  FeatureEncoder   →  [hidden_dim]
ball: [18 floats + IDs]  →  BallEncoder     →  [hidden_dim]
query: placeholder       →  QueryEncoder     →  [hidden_dim]
```

## EntityEncoder

**Purpose**: Convert entity IDs (venues, teams, players) to learned embeddings.

**Source**: `src/model/encoders.py:17-70`

```python
class EntityEncoder(nn.Module):
    def __init__(
        self,
        num_entities: int,      # Number of unique entities
        embed_dim: int,         # Embedding table dimension
        hidden_dim: int,        # Output dimension
        dropout: float = 0.1,
    ):
        # +1 for unknown entity (ID 0 is reserved)
        self.embedding = nn.Embedding(num_entities + 1, embed_dim, padding_idx=0)
        
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
```

**Architecture**:
```
ID (int) → Embedding Table → [embed_dim] → MLP → [hidden_dim]
```

**Key Design Choices**:

1. **Separate embed_dim from hidden_dim**: Embedding tables can be smaller (32 for venues) than the model's working dimension (128). This saves parameters while still projecting to full hidden_dim.

2. **Padding index 0**: ID 0 is reserved for unknown entities. The embedding at index 0 is zero-initialized and kept at zero.

3. **Two-layer MLP projection**: Adds non-linearity and capacity beyond simple linear projection.

**Usage by Node Type**:
| Node Type | num_entities | embed_dim | Typical Count |
|-----------|--------------|-----------|---------------|
| venue | num_venues | 32 | ~50-100 |
| batting_team | num_teams | 32 | ~30-50 |
| bowling_team | num_teams | 32 | ~30-50 |

## HierarchicalPlayerEncoder

**Purpose**: Handle the cold-start problem for player embeddings.

**The Problem**: Unknown players (new debuts, rare players) get ID=0, which maps to a zero embedding. This provides no useful information.

**The Solution**: When player_id=0, fall back to team + role embeddings.

**Source**: `src/model/encoders.py:73-200`

```python
class HierarchicalPlayerEncoder(nn.Module):
    """
    Hierarchy:
    - Level 1: Player embedding (specific individual)
    - Level 2: Team embedding (team-level characteristics)
    - Level 3: Role embedding (role-level prior: opener, finisher, bowler, etc.)
    """
    
    def __init__(
        self,
        num_players: int,
        num_teams: int,
        num_roles: int,           # 8 roles: unknown + 7 categories
        player_embed_dim: int,    # 64
        team_embed_dim: int,      # 16 (smaller for fallback)
        role_embed_dim: int,      # 16
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        # Primary: Player embedding
        self.player_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)
        
        # Fallback embeddings
        self.team_embed = nn.Embedding(num_teams + 1, team_embed_dim, padding_idx=0)
        self.role_embed = nn.Embedding(num_roles + 1, role_embed_dim, padding_idx=0)
        
        # Projection for known players
        self.player_projection = nn.Sequential(
            nn.Linear(player_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Projection for unknown players (team + role → hidden)
        self.fallback_projection = nn.Sequential(
            nn.Linear(team_embed_dim + role_embed_dim, hidden_dim),
            ...
        )
```

**Forward Pass**:
```python
def forward(self, player_ids, team_ids, role_ids):
    # Get all embeddings
    player_emb = self.player_embed(player_ids)    # [N, 64]
    team_emb = self.team_embed(team_ids)          # [N, 16]
    role_emb = self.role_embed(role_ids)          # [N, 16]
    
    # Project known players
    known_output = self.player_projection(player_emb)   # [N, hidden_dim]
    
    # Project fallback (team + role concatenation)
    fallback_input = torch.cat([team_emb, role_emb], dim=-1)  # [N, 32]
    fallback_output = self.fallback_projection(fallback_input)  # [N, hidden_dim]
    
    # Select based on whether player is known
    is_unknown = (player_ids == 0)
    output = torch.where(is_unknown, fallback_output, known_output)
    
    return output
```

**Role Categories**:
```python
PLAYER_ROLES = [
    'unknown',      # 0 - fallback for completely unknown
    'opener',       # 1 - opening batsman (aggressive, faces new ball)
    'top_order',    # 2 - positions 3-4 (anchors innings)
    'middle_order', # 3 - positions 5-6 (builds or accelerates)
    'finisher',     # 4 - positions 6-7 (explosive, death overs)
    'bowler',       # 5 - primarily a bowler (tailender batting)
    'allrounder',   # 6 - genuine all-rounder
    'keeper',       # 7 - wicket-keeper (often finisher role)
]
```

**Why This Design?**
1. **Graceful degradation**: Unknown player still gets meaningful representation
2. **Team context**: "Australian opener" is more informative than nothing
3. **Role prior**: Opener vs tailender have very different expected behavior
4. **No wasted capacity**: Known players use full player embedding

## FeatureEncoder

**Purpose**: Project numeric feature vectors to hidden_dim.

**Source**: `src/model/encoders.py:203-241`

```python
class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,         # Number of input features
        hidden_dim: int,        # Output dimension
        dropout: float = 0.1,
    ):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)  # [batch, input_dim] → [batch, hidden_dim]
```

**Usage by Node Type**:
| Node Type | input_dim | Features |
|-----------|-----------|----------|
| score_state | 5 | runs, wickets, balls, innings, is_womens |
| chase_state | 7 | runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem |
| phase_state | 6 | is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over |
| time_pressure | 3 | balls_remaining, urgency, is_final_over |
| wicket_buffer | 2 | wickets_in_hand, is_tail |
| striker_state | 8 | runs, balls, sr, dots_pct, is_set, boundaries, is_debut, balls_since |
| nonstriker_state | 8 | (same structure, Z2 symmetric) |
| bowler_state | 8 | balls, runs, wickets, economy, dots_pct, threat, is_pace, is_spin |
| partnership | 4 | runs, balls, run_rate, stability |
| batting_momentum | 1 | momentum score |
| bowling_momentum | 1 | momentum score |
| pressure_index | 1 | pressure score |
| dot_pressure | 5 | consecutive_dots, balls_since_boundary, balls_since_wicket, accumulated, trend |

**Design Notes**:
- **LayerNorm on output**: Ensures consistent scale across node types
- **GELU activation**: Smooth non-linearity, better gradient flow than ReLU
- **Two layers**: Adds capacity for feature interactions

## BallEncoder

**Purpose**: Encode ball nodes with both numeric features AND player embeddings.

**The Challenge**: Ball nodes contain:
- 18 numeric features (runs, wicket type, position, etc.)
- Bowler ID (who bowled this ball)
- Batsman ID (who faced this ball)

These need to be combined into a single representation.

**Source**: `src/model/encoders.py:244-321`

```python
class BallEncoder(nn.Module):
    def __init__(
        self,
        num_players: int,
        player_embed_dim: int,    # 64
        feature_dim: int,         # 18
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        # Player embeddings for historical balls
        self.bowler_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)
        self.batsman_embed = nn.Embedding(num_players + 1, player_embed_dim, padding_idx=0)
        
        # Projection: features + bowler_emb + batsman_emb → hidden_dim
        concat_dim = 2 * player_embed_dim + feature_dim  # 64 + 64 + 18 = 146
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
```

**Forward Pass**:
```python
def forward(self, x, bowler_ids, batsman_ids):
    bowler_emb = self.bowler_embed(bowler_ids)     # [N_balls, 64]
    batsman_emb = self.batsman_embed(batsman_ids)  # [N_balls, 64]
    
    combined = torch.cat([x, bowler_emb, batsman_emb], dim=-1)  # [N_balls, 146]
    return self.projection(combined)  # [N_balls, hidden_dim]
```

**Why Include Player Embeddings?**
The identity of who bowled/faced each ball is crucial information:
- "6 runs off Bumrah" is very different from "6 runs off a part-timer"
- "Dot to Kohli" vs "Dot to a tailender" have different implications

**Note**: These are SEPARATE embedding tables from the identity nodes. Ball nodes learn "this bowler in historical context" while identity nodes learn "this bowler as current actor".

## QueryEncoder

**Purpose**: Provide a learned initial embedding for the query node.

**Source**: `src/model/encoders.py:324-350`

```python
class QueryEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        # Learned query embedding (shared across all samples)
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
    
    def forward(self, batch_size: int):
        return self.embedding.expand(batch_size, -1)  # [batch, hidden_dim]
```

**Why a Learned Embedding?**
- Query node is an AGGREGATION point, not a data-bearing node
- Its initial state is learned to best receive information from context
- After message passing, it will contain aggregated graph context
- Think of it as a "question vector" that asks "what will happen next?"

**Initialization**: Small random values (0.02 std) to avoid dominating initial message passing.

## NodeEncoderDict

**Purpose**: Container that organizes all encoders and provides a single `encode_nodes()` method.

**Source**: `src/model/encoders.py:353-510`

```python
class NodeEncoderDict(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_venues: int,
        num_teams: int,
        num_players: int,
        num_roles: int = 8,
        venue_embed_dim: int = 32,
        team_embed_dim: int = 32,
        player_embed_dim: int = 64,
        role_embed_dim: int = 16,
        dropout: float = 0.1,
        use_hierarchical_player: bool = True,
    ):
        # Entity encoders
        self.venue_encoder = EntityEncoder(num_venues, venue_embed_dim, hidden_dim, dropout)
        self.team_encoder = EntityEncoder(num_teams, team_embed_dim, hidden_dim, dropout)
        
        # Player encoder (hierarchical or simple)
        if use_hierarchical_player:
            self.player_encoder = HierarchicalPlayerEncoder(...)
        else:
            self.player_encoder = EntityEncoder(num_players, player_embed_dim, hidden_dim, dropout)
        
        # Feature encoders (one per feature node type)
        self.feature_encoders = nn.ModuleDict({
            'score_state': FeatureEncoder(5, hidden_dim, dropout),
            'chase_state': FeatureEncoder(7, hidden_dim, dropout),
            'phase_state': FeatureEncoder(6, hidden_dim, dropout),
            'time_pressure': FeatureEncoder(3, hidden_dim, dropout),
            'wicket_buffer': FeatureEncoder(2, hidden_dim, dropout),
            'striker_state': FeatureEncoder(8, hidden_dim, dropout),
            'nonstriker_state': FeatureEncoder(8, hidden_dim, dropout),
            'bowler_state': FeatureEncoder(8, hidden_dim, dropout),
            'partnership': FeatureEncoder(4, hidden_dim, dropout),
            'batting_momentum': FeatureEncoder(1, hidden_dim, dropout),
            'bowling_momentum': FeatureEncoder(1, hidden_dim, dropout),
            'pressure_index': FeatureEncoder(1, hidden_dim, dropout),
            'dot_pressure': FeatureEncoder(5, hidden_dim, dropout),
        })
        
        # Ball encoder
        self.ball_encoder = BallEncoder(num_players, player_embed_dim, 18, hidden_dim, dropout)
        
        # Query encoder
        self.query_encoder = QueryEncoder(hidden_dim)
```

**The encode_nodes() Method**:
```python
def encode_nodes(self, data) -> dict:
    x_dict = {}
    
    # Entity nodes
    x_dict['venue'] = self.venue_encoder(data['venue'].x)
    x_dict['batting_team'] = self.team_encoder(data['batting_team'].x)
    x_dict['bowling_team'] = self.team_encoder(data['bowling_team'].x)
    
    # Player nodes (hierarchical: needs player_id, team_id, role_id)
    if self.use_hierarchical_player:
        x_dict['striker_identity'] = self.player_encoder(
            data['striker_identity'].x,        # player_id
            data['striker_identity'].team_id,   # team fallback
            data['striker_identity'].role_id,   # role fallback
        )
        # ... same for nonstriker_identity, bowler_identity
    
    # Feature nodes
    for node_type, encoder in self.feature_encoders.items():
        x_dict[node_type] = encoder(data[node_type].x)
    
    # Ball nodes (handle empty case for first-ball prediction)
    if 'ball' in data.node_types and data['ball'].num_nodes > 0:
        x_dict['ball'] = self.ball_encoder(
            data['ball'].x,
            data['ball'].bowler_ids,
            data['ball'].batsman_ids
        )
    else:
        x_dict['ball'] = torch.zeros((0, hidden_dim), device=device)
    
    # Query node
    num_queries = data['query'].num_nodes
    x_dict['query'] = self.query_encoder(num_queries)
    
    return x_dict
```

**Output**: Dictionary mapping node type → tensor of shape [num_nodes, hidden_dim]

## Summary

| Encoder | Input | Output | Key Feature |
|---------|-------|--------|-------------|
| EntityEncoder | ID (int) | [hidden_dim] | Embedding + projection |
| HierarchicalPlayerEncoder | player_id, team_id, role_id | [hidden_dim] | Cold-start fallback |
| FeatureEncoder | [input_dim] floats | [hidden_dim] | MLP with LayerNorm |
| BallEncoder | [18] floats + IDs | [hidden_dim] | Combined features + player embeddings |
| QueryEncoder | (none) | [hidden_dim] | Learned parameter |

---

*Next: [04-conv-layers.md](./04-conv-layers.md) - Per-edge convolution operators and why*
