# Graph Structure for Interpretable Attention

## Design Goal

Create a graph structure where **every node has semantic meaning** that an LLM can interpret. When we extract attention weights, we want statements like:

> "The model attended 35% to Chase State, 25% to Pressure Index, 20% to Batsman Momentum"

Not:

> "The model attended 35% to Node 3, 25% to Node 7"

## Why 4 Nodes Is Insufficient

The original 4-node design (Batsman, Bowler, Partner, Context) conflates multiple concepts:

| Original Node | Conflated Concepts |
|---------------|-------------------|
| Batsman | Identity (WHO) + State (HOW they're doing) + Form (RECENT performance) |
| Bowler | Identity (WHO) + State (spell figures) + Form (RECENT performance) |
| Context | Score + Chase + Phase + Pressure + Time (too many things!) |

When attention is high on "Context", we can't tell if it's the chase equation, the phase, or the time pressure that matters.

## Revised Graph Structure: 17 Semantic Nodes

### Layer 1: Global Context (3 nodes)

These nodes represent **match-level context** that persists throughout:

| Node | Features | Semantic Meaning |
|------|----------|------------------|
| **Venue** | `venue_embedding`, `avg_1st_score`, `avg_2nd_score`, `boundary_rate`, `pace_friendly` | "What does this ground allow?" |
| **Team Context** | `batting_team_embedding`, `bowling_team_embedding`, `batting_depth_remaining`, `bowling_resources` | "What are the team capabilities?" |
| **Match Importance** | `tournament_stage`, `event_type`, `is_knockout` | "How much does this match matter?" |

### Layer 2: Match State (5 nodes)

These nodes represent **current match situation**:

| Node | Features | Semantic Meaning |
|------|----------|------------------|
| **Score State** | `total_runs`, `run_rate`, `projected_score` | "What's the scoring situation?" |
| **Chase State** | `target`, `runs_required`, `required_run_rate`, `rrr_gap` | "What does the chase demand?" (2nd innings only) |
| **Phase State** | `is_powerplay`, `is_middle`, `is_death`, `phase_balls_remaining` | "What phase of innings are we in?" |
| **Time Pressure** | `balls_remaining`, `overs_remaining`, `normalized_progress` | "How much time is left?" |
| **Wicket Buffer** | `wickets_remaining`, `wickets_lost`, `tail_exposed` | "How many resources remain?" |

### Layer 3: Actor Nodes (5 nodes)

These nodes represent **players involved in this ball**:

| Node | Features | Semantic Meaning |
|------|----------|------------------|
| **Batsman Identity** | `player_embedding`, `career_sr`, `career_avg`, `batting_style` | "WHO is batting? What's their capability?" |
| **Batsman State** | `runs`, `balls`, `strike_rate`, `fours`, `sixes`, `dots` | "HOW is the batsman performing this innings?" |
| **Bowler Identity** | `player_embedding`, `career_economy`, `career_sr`, `bowling_style` | "WHO is bowling? What's their capability?" |
| **Bowler State** | `overs`, `runs_conceded`, `wickets`, `economy`, `overs_remaining` | "HOW is the bowler performing this spell?" |
| **Partnership** | `partnership_runs`, `partnership_balls`, `striker_share`, `both_set` | "What's the partnership dynamic?" |

### Layer 4: Dynamics Nodes (4 nodes)

These nodes represent **recent dynamics and computed metrics**:

| Node | Features | Semantic Meaning |
|------|----------|------------------|
| **Batting Momentum** | `last_6_runs`, `last_12_runs`, `boundaries_last_12`, `batsman_recent_sr` | "Is the batting team on top recently?" |
| **Bowling Momentum** | `wickets_last_5_overs`, `economy_last_2_overs`, `consecutive_dots` | "Is the bowling team on top recently?" |
| **Pressure Index** | `composite_pressure`, `rr_pressure`, `wicket_pressure`, `time_pressure` | "How much pressure is on the batting team?" |
| **Dot Ball Pressure** | `consecutive_dots_batsman`, `consecutive_dots_team`, `dots_this_over` | "Is scoring pressure building?" |

## Node Feature Vectors

### Full specification for each node:

```python
# See: ../data-analysis/04-derived-features-catalog.md for computation details

VENUE_NODE_FEATURES = [
    'venue_id',           # For embedding lookup
    'avg_first_innings',  # Historical par score
    'avg_second_innings', # Historical chase score
    'boundary_rate',      # Historical boundary %
    'pace_advantage',     # Pace vs spin advantage
]  # dim = 5 + embedding_dim

TEAM_CONTEXT_FEATURES = [
    'batting_team_id',           # For embedding lookup
    'bowling_team_id',           # For embedding lookup
    'batting_positions_left',    # Batsmen yet to bat
    'bowler_overs_available',    # Total bowling overs left across team
]  # dim = 4 + 2*embedding_dim

MATCH_IMPORTANCE_FEATURES = [
    'is_international',   # 1 if international, 0 if franchise
    'is_knockout',        # 1 if knockout stage
    'event_importance',   # Scaled importance score
]  # dim = 3

SCORE_STATE_FEATURES = [
    'total_runs',         # Current score
    'run_rate',           # Current run rate
    'projected_score',    # If continued at this rate
    'par_score_gap',      # Current vs expected at this stage
]  # dim = 4

CHASE_STATE_FEATURES = [
    'has_target',         # 1 if 2nd innings, 0 otherwise
    'target',             # Target runs (0 if 1st innings)
    'runs_required',      # Runs still needed
    'required_run_rate',  # RRR
    'rrr_gap',            # RRR - current RR
    'runs_per_ball_needed', # Granular chase pressure
]  # dim = 6

PHASE_STATE_FEATURES = [
    'is_powerplay',       # 1 if overs 1-6
    'is_middle',          # 1 if overs 7-15
    'is_death',           # 1 if overs 16-20
    'phase_progress',     # Progress within current phase [0,1]
    'balls_to_phase_end', # Balls until phase change
]  # dim = 5

TIME_PRESSURE_FEATURES = [
    'balls_remaining',    # Balls left in innings
    'overs_remaining',    # Overs left
    'normalized_progress',# balls_faced / 120
    'time_urgency',       # Non-linear time pressure
]  # dim = 4

WICKET_BUFFER_FEATURES = [
    'wickets_remaining',  # Wickets in hand
    'wickets_lost',       # Wickets fallen
    'tail_exposed',       # 1 if tail is in
    'last_recognized_batsman', # Quality of remaining batsmen
]  # dim = 4

BATSMAN_IDENTITY_FEATURES = [
    'player_id',          # For embedding lookup
    'career_strike_rate', # Historical SR
    'career_average',     # Historical average
    'sr_vs_pace',         # SR against pace
    'sr_vs_spin',         # SR against spin
    'boundary_percentage',# Career boundary %
]  # dim = 6 + embedding_dim

BATSMAN_STATE_FEATURES = [
    'runs',               # This innings runs
    'balls',              # This innings balls
    'strike_rate',        # This innings SR
    'fours',              # Boundaries hit
    'sixes',              # Sixes hit
    'dots',               # Dot balls faced
    'setness',            # Computed setness [0,1]
]  # dim = 7

BOWLER_IDENTITY_FEATURES = [
    'player_id',          # For embedding lookup
    'career_economy',     # Historical economy
    'career_strike_rate', # Balls per wicket
    'is_pace',            # Pace bowler flag
    'is_spin',            # Spin bowler flag
    'death_specialist',   # Good in death overs
]  # dim = 6 + embedding_dim

BOWLER_STATE_FEATURES = [
    'overs',              # Overs bowled this innings
    'runs_conceded',      # Runs given
    'wickets',            # Wickets taken
    'economy',            # Current economy
    'overs_remaining',    # Overs left in quota
    'dots_bowled',        # Dot balls bowled
]  # dim = 6

PARTNERSHIP_FEATURES = [
    'runs',               # Partnership runs
    'balls',              # Partnership balls
    'strike_rate',        # Partnership SR
    'striker_share',      # Striker's contribution %
    'both_batsmen_set',   # Both faced 15+ balls
    'rotation_rate',      # Singles per 6 balls
]  # dim = 6

BATTING_MOMENTUM_FEATURES = [
    'runs_last_6',        # Runs in last 6 balls
    'runs_last_12',       # Runs in last 12 balls
    'boundaries_last_12', # Boundaries in last 12
    'sr_last_12',         # Strike rate last 12
    'momentum_score',     # Composite momentum
]  # dim = 5

BOWLING_MOMENTUM_FEATURES = [
    'wickets_last_30',    # Wickets in last 5 overs
    'economy_last_12',    # Economy last 12 balls
    'dots_last_6',        # Dots in last 6 balls
    'bowling_momentum',   # Composite bowling momentum
]  # dim = 4

PRESSURE_INDEX_FEATURES = [
    'composite_pressure', # Overall pressure [0,1]
    'rr_pressure',        # Run rate component
    'wicket_pressure',    # Wicket loss component
    'time_pressure',      # Time component
    'new_batsman_pressure', # If batsman is new
]  # dim = 5

DOT_BALL_PRESSURE_FEATURES = [
    'consecutive_dots_batsman', # Dots faced in a row
    'consecutive_dots_team',    # Team dots in a row
    'dots_this_over',           # Dots in current over
    'scoring_pressure',         # Need to score urgency
]  # dim = 4
```

## Edge Structure

Edges connect nodes that have meaningful relationships:

### Intra-Layer Edges (Within each layer)

| Edge | Nodes Connected | Features | Semantic Meaning |
|------|-----------------|----------|------------------|
| **Batting Matchup** | Batsman Identity ↔ Bowler Identity | `h2h_sr`, `h2h_balls`, `h2h_dismissals` | "How does this matchup typically go?" |
| **State Correlation** | Batsman State ↔ Partnership | `striker_contribution` | "How dominant is striker in partnership?" |

### Inter-Layer Edges (Cross-hierarchy)

| Edge | From → To | Features | Semantic Meaning |
|------|-----------|----------|------------------|
| **Pressure Impact** | Pressure Index → Batsman State | `pressure_weight` | "How does pressure affect this batsman?" |
| **Chase Influence** | Chase State → Actor Layer | `chase_urgency` | "How does chase equation affect behavior?" |
| **Phase Constraint** | Phase State → All Actors | `phase_modulation` | "How does phase change expectations?" |
| **Venue Effect** | Venue → Score State | `venue_adjustment` | "How does venue affect par scores?" |

## Attention Visualization

With this structure, we can extract and display attention at multiple levels:

### Level 1: Which Layer Matters?

```
Global Context: 15%
Match State:    35%  ◄── Chase and time pressure dominating
Actor Layer:    30%
Dynamics Layer: 20%
```

### Level 2: Which Nodes Within Layer?

```
Match State Layer:
├── Score State:    8%
├── Chase State:    18% ◄── Chase equation critical
├── Phase State:    4%
├── Time Pressure:  3%
└── Wicket Buffer:  2%
```

### Level 3: Which Features Within Node?

```
Chase State Node:
├── required_run_rate: 0.42 ◄── Highest attention
├── runs_required:     0.28
├── rrr_gap:           0.20
└── runs_per_ball:     0.10
```

## Implementation Considerations

### Node Embedding Dimensions

| Layer | Nodes | Base Features | Embedding | Total Dim |
|-------|-------|---------------|-----------|-----------|
| Global | 3 | ~12 | 64 (venue) + 128 (teams) | ~204 |
| Match State | 5 | ~23 | 0 | ~23 |
| Actor | 5 | ~31 | 128 (players) | ~415 |
| Dynamics | 4 | ~18 | 0 | ~18 |
| **Total** | **17** | **~84** | **~320** | **~660** |

### Computational Complexity

With 17 nodes, full attention has O(17²) = 289 attention weights per ball. This is:
- **Manageable**: Still much smaller than Transformer sequence attention
- **Sparse**: Many edges can be pruned (e.g., Venue ↔ Dot Pressure has no direct edge)
- **Hierarchical**: Can use hierarchical attention to reduce complexity

## Next: Hierarchical Attention Design

See [03-hierarchical-attention.md](./03-hierarchical-attention.md) for how attention flows across these layers.
