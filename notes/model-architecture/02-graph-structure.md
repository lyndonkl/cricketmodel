# Graph Structure

## Overview

The CricketHeteroGNN operates on a heterogeneous graph with:
- **21 context node types** (fixed per sample)
- **N ball nodes** (variable, one per historical delivery)
- **1 query node** (for prediction aggregation)
- **~150 edge types** (connecting nodes with typed relationships)

## Node Types

### Layer 1: Global (3 nodes)

These represent fixed match-level context:

| Node Type | Features | Meaning |
|-----------|----------|---------|
| `venue` | ID (int) | Cricket ground (Melbourne, Wankhede, etc.) |
| `batting_team` | ID (int) | Team currently batting (Australia, India, etc.) |
| `bowling_team` | ID (int) | Team currently bowling |

**Why separate nodes?** Venue conditions both teams; teams condition their players. This hierarchical structure is captured naturally.

### Layer 2: State (5 nodes)

These represent current match state:

| Node Type | Dim | Features |
|-----------|-----|----------|
| `score_state` | 5 | runs/250, wickets/10, balls/120, innings_indicator, is_womens |
| `chase_state` | 7 | runs_needed/250, rrr/20, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem |
| `phase_state` | 6 | is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over |
| `time_pressure` | 3 | balls_remaining/120, urgency, is_final_over |
| `wicket_buffer` | 2 | wickets_in_hand/10, is_tail |

**Why separate state nodes?** Each captures a different aspect of the game. Score is about resources, chase is about target, phase is about tactical context.

### Layer 3: Actor (7 nodes)

These represent the key players and their current state:

| Node Type | Features | Meaning |
|-----------|----------|---------|
| `striker_identity` | ID + team_id + role_id | WHO is batting (for embedding lookup) |
| `striker_state` | 8 features | HOW the striker is performing this innings |
| `nonstriker_identity` | ID + team_id + role_id | Partner at non-striker end |
| `nonstriker_state` | 8 features | Non-striker's current performance |
| `bowler_identity` | ID + team_id + role_id | WHO is bowling |
| `bowler_state` | 8 features | HOW the bowler is performing this spell |
| `partnership` | 4 features | Current batting pair's partnership |

**Why separate identity and state?**
- **Identity**: Uses embeddings (learned player representations)
- **State**: Uses current innings performance (runs, strike rate, etc.)

The model can learn "Virat Kohli is generally dangerous" (identity) AND "Kohli is struggling today at 5(12)" (state).

### Layer 4: Dynamics (4 nodes)

These capture momentum and pressure:

| Node Type | Dim | Features |
|-----------|-----|----------|
| `batting_momentum` | 1 | Recent run rate vs expected (positive = scoring well) |
| `bowling_momentum` | 1 | Inverse of batting momentum |
| `pressure_index` | 1 | Combined pressure score |
| `dot_pressure` | 5 | consecutive_dots, balls_since_boundary, balls_since_wicket, pressure_accumulated, pressure_trend |

**Why dynamics nodes?** These capture feedback loops:
- R1: Confidence spiral (high momentum → more aggressive → more runs)
- B1: Required rate pressure (high RRR → risk-taking → boundaries OR wickets)

### Layer 5: Ball (N nodes)

Each historical delivery becomes a ball node:

| Feature | Dim | Description |
|---------|-----|-------------|
| runs/6 | 1 | Normalized runs scored |
| is_wicket | 1 | Binary wicket indicator |
| over/20 | 1 | Normalized over number |
| ball_in_over/6 | 1 | Position in over |
| is_boundary | 1 | Four or six |
| is_wide, is_noball, is_bye, is_legbye | 4 | Extra types |
| wicket_bowled, wicket_caught, wicket_lbw, wicket_run_out, wicket_stumped, wicket_other | 6 | Wicket type one-hot |
| striker_run_out, nonstriker_run_out | 2 | Run-out attribution |
| bowling_end | 1 | Which end of pitch (0 or 1) |

**Total: 18 numeric features** + bowler_id + batsman_id (for embedding lookup)

**Why ball nodes?** Historical context is crucial for prediction. The model learns patterns like:
- "3 dots in a row → pressure building"
- "Same matchup earlier → how did it go?"

### Layer 6: Query (1 node)

```python
class QueryEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        # Learned query embedding
        self.embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
```

The query node starts as a learned parameter and aggregates information from the entire graph via message passing.

**Why a query node?** It provides a natural aggregation point. After message passing, the query has "seen" all context through its incoming edges.

## Edge Types

### Hierarchical Edges (Global → State → Actor → Dynamics)

```python
# All-to-all connections between consecutive layers
HIERARCHICAL_EDGES = [
    # Global → State (15 edges: 3 global × 5 state)
    (venue, conditions, score_state),
    (venue, conditions, chase_state),
    (batting_team, conditions, phase_state),
    ...
    
    # State → Actor (35 edges: 5 state × 7 actor)
    (score_state, conditions, striker_identity),
    (chase_state, conditions, striker_state),
    ...
    
    # Actor → Dynamics (28 edges: 7 actor × 4 dynamics)
    (striker_identity, conditions, batting_momentum),
    (bowler_state, conditions, bowling_momentum),
    ...
]
```

**Purpose**: Top-down conditioning. Venue conditions team behavior, team conditions player behavior, player conditions momentum.

### Intra-Layer Edges

#### Global Layer (bidirectional)
```python
INTRA_LAYER_GLOBAL = [
    (venue, relates_to, batting_team),
    (venue, relates_to, bowling_team),
    (batting_team, relates_to, bowling_team),
]
```

#### State Layer (bidirectional)
```python
INTRA_LAYER_STATE = [
    (score_state, relates_to, chase_state),
    (score_state, relates_to, phase_state),
    (chase_state, relates_to, time_pressure),
    ...
]
```

#### Actor Layer (matchup relation)
```python
INTRA_LAYER_ACTOR = [
    # Identity to state
    (striker_identity, matchup, striker_state),
    (bowler_identity, matchup, bowler_state),
    
    # THE KEY MATCHUP
    (striker_identity, matchup, bowler_identity),
    
    # Partnership connections
    (striker_state, matchup, partnership),
    (nonstriker_state, matchup, partnership),
    ...
]
```

**Why `matchup` relation for actors?** The striker vs bowler interaction is fundamentally different from score vs phase interaction. Using a distinct relation allows a different convolution operator.

### Temporal Ball Edges

```python
# Multi-scale temporal edges
(ball_i, recent_precedes, ball_j)    # gap 1-6 balls (within-over)
(ball_i, medium_precedes, ball_j)    # gap 7-24 balls (spell window)
(ball_i, distant_precedes, ball_j)   # gap 25+ balls (sparse, every 6th)

# Set-based grouping (with temporal distance as edge attribute)
(ball_i, same_bowler, ball_j)        # Same bowler bowled both
(ball_i, same_batsman, ball_j)       # Same batsman faced both
(ball_i, same_matchup, ball_j)       # Same bowler-batsman pair (CAUSAL only)
(ball_i, same_over, ball_j)          # Within same over (CAUSAL only)
```

**Why multi-scale temporal?**
- **Recent** (6 balls): Current over context, immediate pressure
- **Medium** (24 balls): 4-over spell window, momentum patterns
- **Distant** (sparse): Historical context without quadratic explosion

**Why CAUSAL for some edges?**
`same_matchup` and `same_over` use causal edges (older → newer only) to prevent train-test distribution shift. During training, bidirectional edges would allow future-to-past information flow, but at inference only historical balls exist.

### Cross-Domain Edges

```python
# Ball → Player (filtered to CURRENT players for relevance)
(ball, faced_by, striker_identity)      # Balls actually faced by current striker
(ball, partnered_by, nonstriker_identity)  # Balls where current NS was involved
(ball, bowled_by, bowler_identity)      # Balls bowled by current bowler

# Ball → Dynamics (recent balls inform momentum)
(ball, informs, batting_momentum)
(ball, informs, bowling_momentum)
(ball, informs, pressure_index)
(ball, informs, dot_pressure)
```

**Why filtered cross-domain?** Only balls involving the CURRENT players are relevant. If Kohli is batting now, we only care about balls Kohli faced, not all balls.

### Query Edges

```python
# Everything → Query (for aggregation)
(venue, attends, query)
(batting_team, attends, query)
...
(ball_i, attends, query)

# Dynamics → Query (explicit momentum influence)
(batting_momentum, drives, query)
(bowling_momentum, drives, query)
(pressure_index, drives, query)
(dot_pressure, drives, query)
```

**Why `drives` in addition to `attends`?** Dynamics have a direct influence on prediction (momentum → risk-taking → outcomes). The separate `drives` relation allows a different convolution operator with stronger attention weights.

## Edge Attribute Summary

Some edge types have edge attributes for position/distance-aware attention:

| Edge Type | Edge Attribute | Meaning |
|-----------|---------------|---------|
| `recent_precedes` | temporal_distance/6 | How many balls apart (0-1) |
| `medium_precedes` | (distance-6)/18 | Normalized within window |
| `same_bowler` | temporal_distance/24 | Balls since same bowler |
| `same_batsman` | temporal_distance/60 | Balls since same batsman |
| `same_over` | target_ball_position/5 | Position in over (0-1) |

These attributes are passed to `TransformerConv` via `edge_dim=1`.

## Complete Edge Type List

The function `get_all_edge_types()` returns ~150 unique edge types:

```python
# Categories:
# 1. Hierarchical 'conditions' edges (~78)
# 2. Intra-layer 'relates_to' edges (~14)
# 3. Actor 'matchup' edges (~26)
# 4. Temporal ball edges (7 types)
# 5. Cross-domain edges (~7)
# 6. Query 'attends' edges (~22)
# 7. Dynamics 'drives' edges (4)
```

## Graph Statistics (Example)

For a sample at ball 60 of an innings:

| Metric | Value |
|--------|-------|
| Total nodes | 22 context + 60 balls + 1 query = 83 |
| Hierarchical edges | 78 |
| Intra-layer edges | ~40 |
| Temporal edges (recent) | ~250 |
| Temporal edges (medium) | ~600 |
| Same-bowler edges | varies by spell patterns |
| Cross-domain edges | varies by player history |
| Query edges | ~82 |
| **Total edges** | ~1500-2500 |

---

*Next: [03-encoders.md](./03-encoders.md) - How raw features become embeddings*
