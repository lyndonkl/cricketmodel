# Hierarchical Attention Architecture

## Design Motivation

Standard flat attention over all 17 nodes:
- Loses semantic hierarchy
- Hard to interpret (which level mattered?)
- Computationally less efficient

Hierarchical attention:
- Respects the semantic layers
- Enables multi-resolution interpretability
- Allows global context to condition ball-level predictions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL ATTENTION FLOW                       │
│                                                                      │
│  GLOBAL LAYER                                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                              │
│  │  Venue  │  │  Team   │  │  Match  │                              │
│  │  Node   │  │  Node   │  │ Import. │                              │
│  └────┬────┘  └────┬────┘  └────┬────┘                              │
│       └────────────┼────────────┘                                    │
│                    ▼                                                 │
│            ┌──────────────┐                                         │
│            │ Global Pool  │  h_global = attention([venue, team, importance])
│            └──────┬───────┘                                         │
│                   │                                                  │
│                   ▼ (broadcast to all lower layers)                 │
│  ════════════════════════════════════════════════════════════════   │
│                                                                      │
│  MATCH STATE LAYER                                                   │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                 │
│  │ Score │ │ Chase │ │ Phase │ │ Time  │ │Wicket │                 │
│  │ State │ │ State │ │ State │ │ Press │ │Buffer │                 │
│  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘                 │
│      └─────────┼─────────┼─────────┼─────────┘                      │
│                ▼                                                     │
│        ┌─────────────────┐                                          │
│        │ State Attention │  h_state = attention([score, chase, ...], h_global)
│        └────────┬────────┘                                          │
│                 │                                                    │
│                 ▼ (conditions actor layer)                          │
│  ════════════════════════════════════════════════════════════════   │
│                                                                      │
│  ACTOR LAYER                                                         │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────────┐         │
│  │Batsman │ │Batsman │ │Bowler  │ │Bowler  │ │Partnership│         │
│  │Identity│ │ State  │ │Identity│ │ State  │ │   Node    │         │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └─────┬─────┘         │
│       └──────────┼──────────┼──────────┼───────────┘                │
│                  ▼                                                   │
│          ┌──────────────┐                                           │
│          │Actor Attention│  h_actor = GAT([bat, bowl, partner], h_state)
│          └──────┬───────┘                                           │
│                 │                                                    │
│                 ▼ (conditions dynamics layer)                       │
│  ════════════════════════════════════════════════════════════════   │
│                                                                      │
│  DYNAMICS LAYER                                                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                        │
│  │Batting │ │Bowling │ │Pressure│ │DotBall │                        │
│  │Momentum│ │Momentum│ │ Index  │ │Pressure│                        │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘                        │
│       └──────────┼──────────┼──────────┘                            │
│                  ▼                                                   │
│         ┌───────────────┐                                           │
│         │Dynamics Attn  │  h_dynamics = attention([...], h_actor)   │
│         └───────┬───────┘                                           │
│                 │                                                    │
│                 ▼                                                    │
│  ════════════════════════════════════════════════════════════════   │
│                                                                      │
│  BALL REPRESENTATION                                                 │
│         ┌────────────────────────────────────────────┐              │
│         │ h_ball = concat(h_global, h_state,          │              │
│         │                 h_actor, h_dynamics)        │              │
│         └────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer-Wise Attention Mechanisms

### 1. Global Context Attention

The global layer aggregates match-level context:

```python
class GlobalContextAttention(nn.Module):
    """
    Aggregates venue, team, and match importance into global context.
    Uses multi-head attention to capture different aspects.
    """
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)
        self.global_query = nn.Parameter(torch.randn(1, 1, node_dim))

    def forward(self, venue, team, importance):
        # Stack global nodes: (batch, 3, dim)
        global_nodes = torch.stack([venue, team, importance], dim=1)

        # Query all global nodes with learned query
        query = self.global_query.expand(venue.size(0), -1, -1)
        h_global, attn_weights = self.attention(query, global_nodes, global_nodes)

        return h_global.squeeze(1), attn_weights  # (batch, dim), (batch, 1, 3)
```

**Interpretability Output**:
```json
{
  "global_attention": {
    "venue": 0.45,
    "team_context": 0.35,
    "match_importance": 0.20
  }
}
```

### 2. Match State Cross-Attention

Match state nodes attend to global context and each other:

```python
class MatchStateAttention(nn.Module):
    """
    Match state nodes (score, chase, phase, time, wicket) attend to:
    1. Each other (intra-layer)
    2. Global context (cross-layer conditioning)
    """
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        # Intra-layer self-attention
        self.self_attn = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)
        # Cross-attention to global context
        self.cross_attn = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)

    def forward(self, state_nodes, h_global):
        # state_nodes: (batch, 5, dim) - [score, chase, phase, time, wicket]

        # Self-attention within match state layer
        h_self, self_attn = self.self_attn(state_nodes, state_nodes, state_nodes)

        # Cross-attention: state nodes query global context
        h_global_expanded = h_global.unsqueeze(1)  # (batch, 1, dim)
        h_cross, cross_attn = self.cross_attn(h_self, h_global_expanded, h_global_expanded)

        # Pool to single representation (or keep all for downstream)
        h_state = h_cross.mean(dim=1)  # (batch, dim)

        return h_state, {'self_attn': self_attn, 'cross_attn': cross_attn}
```

**Interpretability Output**:
```json
{
  "match_state_attention": {
    "score_state": 0.15,
    "chase_state": 0.40,
    "phase_state": 0.20,
    "time_pressure": 0.15,
    "wicket_buffer": 0.10
  },
  "global_influence": 0.35
}
```

### 3. Actor Layer GAT

Actor nodes use Graph Attention to model relationships:

```python
class ActorLayerGAT(nn.Module):
    """
    Actor nodes connected via meaningful edges:
    - Batsman Identity ↔ Bowler Identity (matchup)
    - Batsman Identity ↔ Batsman State (player-state link)
    - Bowler Identity ↔ Bowler State (player-state link)
    - Both States ↔ Partnership (contribution)

    Conditioned on match state context.
    """
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.gat_layer = GATConv(node_dim, node_dim, heads=num_heads, concat=False)
        self.context_projection = nn.Linear(node_dim, node_dim)

    def forward(self, actor_nodes, edge_index, edge_attr, h_state):
        # actor_nodes: (batch * 5, dim) - flattened actor nodes
        # Condition on match state
        h_context = self.context_projection(h_state)
        actor_nodes = actor_nodes + h_context.repeat(5, 1)  # Broadcast context

        # GAT forward
        h_actors, (edge_index_out, attn_weights) = self.gat_layer(
            actor_nodes, edge_index, edge_attr, return_attention_weights=True
        )

        return h_actors, attn_weights
```

**Interpretability Output**:
```json
{
  "actor_attention": {
    "batsman_identity": 0.25,
    "batsman_state": 0.20,
    "bowler_identity": 0.20,
    "bowler_state": 0.15,
    "partnership": 0.20
  },
  "matchup_attention": 0.35,
  "state_conditioned": true
}
```

### 4. Dynamics Layer Attention

Dynamics nodes capture recent patterns, conditioned on actors:

```python
class DynamicsAttention(nn.Module):
    """
    Dynamics nodes (momentum, pressure) attend to each other
    and are conditioned on actor representations.
    """
    def __init__(self, node_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(node_dim, num_heads, batch_first=True)
        self.actor_projection = nn.Linear(node_dim, node_dim)

    def forward(self, dynamics_nodes, h_actor):
        # dynamics_nodes: (batch, 4, dim)
        # Condition on actor representation
        h_actor_proj = self.actor_projection(h_actor).unsqueeze(1)
        dynamics_conditioned = dynamics_nodes + h_actor_proj

        # Self-attention over dynamics
        h_dynamics, attn_weights = self.attention(
            dynamics_conditioned, dynamics_conditioned, dynamics_conditioned
        )

        return h_dynamics.mean(dim=1), attn_weights
```

**Interpretability Output**:
```json
{
  "dynamics_attention": {
    "batting_momentum": 0.30,
    "bowling_momentum": 0.15,
    "pressure_index": 0.40,
    "dot_ball_pressure": 0.15
  }
}
```

## Cross-Layer Information Flow

### Top-Down Conditioning

Each layer receives context from the layer above:

```
h_global → conditions → h_state
h_state  → conditions → h_actor
h_actor  → conditions → h_dynamics
```

This creates **contextual attention**: what matters in the dynamics layer depends on what's happening at the actor layer.

### Bottom-Up Aggregation

Final ball representation combines all layers:

```python
def aggregate_hierarchical(h_global, h_state, h_actor, h_dynamics):
    """
    Aggregate hierarchical representations into ball embedding.
    Option 1: Concatenation
    Option 2: Weighted sum with learned weights
    Option 3: Cross-attention fusion
    """
    # Option 1: Simple concatenation
    h_ball = torch.cat([h_global, h_state, h_actor, h_dynamics], dim=-1)

    # Option 2: Learned layer importance
    # layer_weights = softmax(self.layer_importance)
    # h_ball = sum(w * h for w, h in zip(layer_weights, [h_global, h_state, h_actor, h_dynamics]))

    return h_ball
```

## Attention Weight Extraction

### For LLM Interpretability

```python
class HierarchicalAttentionModel(nn.Module):
    def get_all_attention_weights(self, batch):
        """
        Extract all attention weights at every level for LLM interpretation.
        """
        # Forward pass (stores attention weights)
        output = self.forward(batch)

        return {
            # Layer-level importance
            'layer_importance': self.get_layer_importance(),

            # Within-layer attention
            'global': {
                'venue': self.global_attn_weights[0],
                'team': self.global_attn_weights[1],
                'importance': self.global_attn_weights[2],
            },
            'match_state': {
                'score': self.state_attn_weights[0],
                'chase': self.state_attn_weights[1],
                'phase': self.state_attn_weights[2],
                'time': self.state_attn_weights[3],
                'wicket': self.state_attn_weights[4],
            },
            'actor': {
                'batsman_identity': self.actor_attn_weights[0],
                'batsman_state': self.actor_attn_weights[1],
                'bowler_identity': self.actor_attn_weights[2],
                'bowler_state': self.actor_attn_weights[3],
                'partnership': self.actor_attn_weights[4],
                'matchup_edge': self.matchup_attention,
            },
            'dynamics': {
                'batting_momentum': self.dynamics_attn_weights[0],
                'bowling_momentum': self.dynamics_attn_weights[1],
                'pressure_index': self.dynamics_attn_weights[2],
                'dot_pressure': self.dynamics_attn_weights[3],
            },

            # Cross-layer conditioning strengths
            'cross_layer': {
                'global_to_state': self.global_conditioning_strength,
                'state_to_actor': self.state_conditioning_strength,
                'actor_to_dynamics': self.actor_conditioning_strength,
            }
        }
```

## Example: Full Attention Profile for One Ball

```json
{
  "ball_number": 47,
  "over": 7.5,
  "prediction": {"boundary": 0.35, "single": 0.28, "dot": 0.22, "wicket": 0.08, "other": 0.07},

  "layer_importance": {
    "global": 0.10,
    "match_state": 0.35,
    "actor": 0.30,
    "dynamics": 0.25
  },

  "attention_breakdown": {
    "global": {
      "venue": 0.40,
      "team_context": 0.35,
      "match_importance": 0.25
    },
    "match_state": {
      "score_state": 0.12,
      "chase_state": 0.45,
      "phase_state": 0.18,
      "time_pressure": 0.15,
      "wicket_buffer": 0.10
    },
    "actor": {
      "batsman_identity": 0.28,
      "batsman_state": 0.22,
      "bowler_identity": 0.20,
      "bowler_state": 0.12,
      "partnership": 0.18
    },
    "dynamics": {
      "batting_momentum": 0.35,
      "bowling_momentum": 0.12,
      "pressure_index": 0.38,
      "dot_ball_pressure": 0.15
    }
  }
}
```

**LLM Insight from this profile**:
> "The model heavily weights the match state layer (35%), particularly the chase equation (45% within that layer). Combined with high attention on pressure index (38% in dynamics) and batting momentum (35%), this suggests the chase is entering a critical phase. The batsman's identity attention (28%) indicates the model considers who is batting to be important for predicting whether they can accelerate."

## Next: Temporal Attention

See [04-temporal-attention.md](./04-temporal-attention.md) for how attention flows across balls in the sequence.
