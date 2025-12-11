# Cricket Ball Prediction: Derived Features Catalog

## Overview

This catalog documents all supplementary features that can be computed from the raw Cricsheet JSON data. Features are organized by:
- **Source**: What raw data is needed
- **Computation**: How to derive the feature
- **Rationale**: Why this feature impacts ball outcomes
- **Priority**: How important for prediction (P0/P1/P2/P3)

---

## Category 1: Match State Features

### 1.1 Score State

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `total_runs` | Sum of all runs in innings | Basic progress indicator | P0 |
| `total_wickets` | Count of wickets in innings | Resource constraint | P0 |
| `run_rate` | `total_runs / overs_completed` | Scoring pace | P0 |
| `balls_faced` | Count of legal deliveries | Progress through innings | P0 |
| `balls_remaining` | `120 - balls_faced` (T20) | Time buffer | P0 |
| `wickets_remaining` | `10 - total_wickets` | Resource buffer | P0 |

### 1.2 Progress State

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `over_number` | Current over (0-indexed) | Phase indicator | P0 |
| `ball_in_over` | Position within current over (1-6) | Bowler's delivery choice | P0 |
| `normalized_progress` | `balls_faced / 120` | Standardized progress [0,1] | P1 |
| `over_ball_decimal` | `over + (ball_in_over - 1) / 6` | Continuous progress | P1 |

### 1.3 Chase State (2nd Innings Only)

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `target_runs` | From `innings[1].target.runs` | What to chase | P0 |
| `runs_required` | `target_runs - total_runs` | Gap to target | P0 |
| `required_run_rate` | `runs_required / (balls_remaining / 6)` | Chase pressure | P0 |
| `rrr_gap` | `required_run_rate - run_rate` | Deficit indicator | P1 |
| `runs_per_ball_needed` | `runs_required / balls_remaining` | Granular pressure | P1 |
| `is_gettable` | `runs_required <= balls_remaining * 2` | Feasibility flag | P2 |

### 1.4 Phase Indicators

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `is_powerplay` | `over_number < 6` | Fielding restrictions active | P1 |
| `is_middle_overs` | `6 <= over_number < 16` | Consolidation phase | P1 |
| `is_death_overs` | `over_number >= 16` | Acceleration phase | P1 |
| `powerplay_balls_remaining` | `max(0, 36 - balls_faced)` | Opportunity window | P2 |
| `death_overs_balls` | `max(0, balls_faced - 96)` | Death pressure duration | P2 |

---

## Category 2: Batsman State Features

### 2.1 Current Innings Stats

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `batsman_runs` | Sum of `runs.batter` for this batsman | Individual score | P0 |
| `batsman_balls` | Count of deliveries faced by batsman | Confidence proxy | P0 |
| `batsman_strike_rate` | `batsman_runs / batsman_balls * 100` | Current form | P1 |
| `batsman_fours` | Count of boundaries (runs.batter == 4) | Aggression indicator | P1 |
| `batsman_sixes` | Count of sixes (runs.batter == 6) | Big-hitting mode | P1 |
| `batsman_dots` | Count of dot balls faced | Pressure absorbed | P2 |
| `batsman_boundary_rate` | `(fours + sixes) / batsman_balls` | Boundary efficiency | P1 |

### 2.2 Batsman Confidence/Setness

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `batsman_is_new` | `batsman_balls < 10` | Vulnerability flag | P1 |
| `batsman_is_set` | `batsman_balls >= 20` | Confidence flag | P1 |
| `batsman_setness` | See formula below | Confidence score [0,1] | P1 |

**Setness Computation**:
```python
def compute_setness(balls_faced):
    if balls_faced < 5:
        return 0.0
    elif balls_faced < 15:
        return 0.3 + 0.7 * ((balls_faced - 5) / 10)
    else:
        return 1.0
```

### 2.3 Recent Form (Rolling Window)

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `batsman_last_6_runs` | Runs in last 6 balls faced | Recent form | P1 |
| `batsman_last_6_sr` | SR in last 6 balls | Recent aggression | P2 |
| `batsman_consecutive_dots` | Consecutive dot balls faced | Pressure buildup | P1 |
| `batsman_consecutive_scoring` | Consecutive scoring shots | Momentum indicator | P2 |
| `batsman_last_boundary_ago` | Balls since last boundary | Boundary hunger | P2 |

---

## Category 3: Bowler State Features

### 3.1 Current Spell Stats

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `bowler_overs` | Overs bowled in this innings | Workload | P1 |
| `bowler_balls` | Balls bowled in this innings | Granular workload | P1 |
| `bowler_runs_conceded` | Runs given away | Economy | P1 |
| `bowler_wickets` | Wickets taken in this innings | Threat level | P1 |
| `bowler_economy` | `bowler_runs_conceded / bowler_overs` | Control | P1 |
| `bowler_strike_rate` | `bowler_balls / bowler_wickets` if wickets > 0 | Penetration | P2 |
| `bowler_extras` | Wides + no-balls bowled | Discipline | P2 |
| `bowler_dots_bowled` | Count of dot balls | Pressure created | P2 |

### 3.2 Bowling Resource Management

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `bowler_overs_remaining` | `4 - bowler_overs` (T20 max) | Resource availability | P1 |
| `bowler_is_finishing_over` | `bowler_balls % 6 > 0` | Mid-over flag | P2 |
| `bowler_balls_in_over` | `bowler_balls % 6` or count in current over | Over progress | P2 |

### 3.3 Recent Form (Rolling Window)

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `bowler_last_over_runs` | Runs in last 6 balls bowled | Recent economy | P1 |
| `bowler_last_over_wickets` | Wickets in last 6 balls | Recent penetration | P2 |
| `bowler_consecutive_dots` | Consecutive dot balls bowled | Building pressure | P2 |
| `bowler_last_boundary_ago` | Balls since giving away boundary | Control streak | P2 |

---

## Category 4: Partnership Features

### 4.1 Current Partnership Stats

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `partnership_runs` | Runs since last wicket | Partnership strength | P1 |
| `partnership_balls` | Balls since last wicket | Partnership duration | P1 |
| `partnership_sr` | `partnership_runs / partnership_balls * 100` | Partnership rate | P2 |
| `partnership_run_rate` | `partnership_runs / (partnership_balls / 6)` | Partnership RPO | P2 |

### 4.2 Partnership Dynamics

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `strike_rotation_rate` | Singles/doubles per ball | Partnership fluidity | P2 |
| `batsman_share` | `batsman_runs / partnership_runs` | Strike dominance | P2 |
| `partner_share` | `1 - batsman_share` | Partner contribution | P2 |
| `partnership_boundary_rate` | Boundaries in partnership / balls | Aggression | P2 |

### 4.3 Partner State

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `partner_runs` | Non-striker's runs | Partner form | P1 |
| `partner_balls` | Non-striker's balls faced | Partner confidence | P1 |
| `partner_strike_rate` | Partner's SR | Partner capability | P2 |
| `partner_is_new` | `partner_balls < 10` | Partner vulnerability | P2 |

---

## Category 5: Momentum Features

### 5.1 Scoring Momentum

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `last_6_balls_runs` | Runs in last 6 balls | Recent momentum | P1 |
| `last_12_balls_runs` | Runs in last 12 balls | Extended momentum | P2 |
| `last_over_runs` | Runs in previous over | Over momentum | P1 |
| `momentum_score` | See formula below | Composite momentum | P1 |

**Momentum Score Computation**:
```python
def compute_momentum(last_n_balls, n=12):
    runs = sum(b.runs.total for b in last_n_balls[-n:])
    boundaries = sum(1 for b in last_n_balls[-n:] if b.runs.batter >= 4)
    wickets = sum(1 for b in last_n_balls[-n:] if b.wickets)

    expected_runs = n * 1.3  # ~7.8 RPO baseline
    run_momentum = (runs - expected_runs) / expected_runs  # [-1, +inf)

    # Penalize for wickets
    momentum = run_momentum - (wickets * 0.5)

    return momentum
```

### 5.2 Wicket Momentum

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `balls_since_last_wicket` | Balls since last dismissal | Stability indicator | P1 |
| `wickets_last_5_overs` | Wickets in last 30 balls | Collapse indicator | P2 |
| `is_new_batsman` | Current batsman faced < 10 balls | Vulnerability window | P1 |

### 5.3 Boundary Momentum

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `boundaries_last_12_balls` | 4s + 6s in last 12 deliveries | Boundary streak | P1 |
| `balls_since_boundary` | Balls since last 4 or 6 | Boundary drought | P1 |
| `boundary_cluster` | Boundaries in last 6 / 6 | Boundary concentration | P2 |

---

## Category 6: Pressure Features

### 6.1 Composite Pressure Index

```python
def compute_pressure_index(match_state, innings):
    """
    Returns pressure score in [0, 1]
    Higher = more pressure on batting team
    """

    # 1. Required rate pressure (2nd innings only)
    if innings == 2 and match_state.target:
        rrr = match_state.required_run_rate
        crr = match_state.run_rate
        rr_pressure = min(1.0, max(0, (rrr - crr) / 6.0))
    else:
        # 1st innings: compare to par
        expected_at_stage = 160 * (match_state.balls_faced / 120)  # Assume 160 par
        deficit = expected_at_stage - match_state.total_runs
        rr_pressure = min(1.0, max(0, deficit / 40))

    # 2. Wicket pressure
    wicket_pressure = match_state.total_wickets / 10

    # 3. Dot ball pressure
    consecutive_dots = match_state.consecutive_dots
    dot_pressure = min(1.0, consecutive_dots / 6)

    # 4. Time pressure
    time_pressure = match_state.balls_faced / 120

    # 5. New batsman pressure
    new_bat_pressure = 0.3 if match_state.batsman_balls < 10 else 0.0

    # Weighted combination
    pressure = (
        0.30 * rr_pressure +
        0.25 * wicket_pressure +
        0.20 * dot_pressure +
        0.15 * time_pressure +
        0.10 * new_bat_pressure
    )

    return pressure
```

### 6.2 Specific Pressure Indicators

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `pressure_index` | See formula above | Composite measure | P1 |
| `dot_ball_pressure` | Consecutive dots / 6 | Building frustration | P1 |
| `chase_pressure` | RRR - CRR (2nd innings) | Chase difficulty | P1 |
| `wicket_pressure` | Wickets / 10 | Risk aversion | P1 |
| `time_pressure` | Balls faced / 120 | Urgency | P1 |

### 6.3 Contextual Pressure

| Feature | Computation | Rationale | Priority |
|---------|-------------|-----------|----------|
| `death_over_pressure` | 1 if death overs, else 0 multiplied by other factors | Phase-specific | P2 |
| `equation_difficulty` | `runs_required / balls_remaining` normalized | Chase granularity | P1 |
| `knockout_flag` | From `event.stage` | Tournament pressure | P3 |

---

## Category 7: Historical/Career Features (External Data Required)

### 7.1 Batsman Career Stats

| Feature | Source | Rationale | Priority |
|---------|--------|-----------|----------|
| `batsman_career_sr` | External DB | Baseline expectation | P2 |
| `batsman_career_avg` | External DB | Quality indicator | P2 |
| `batsman_sr_vs_pace` | External DB | Style matchup | P2 |
| `batsman_sr_vs_spin` | External DB | Style matchup | P2 |
| `batsman_sr_powerplay` | External DB | Phase specialization | P2 |
| `batsman_sr_death` | External DB | Phase specialization | P2 |
| `batsman_boundary_pct` | External DB | Boundary propensity | P2 |

### 7.2 Bowler Career Stats

| Feature | Source | Rationale | Priority |
|---------|--------|-----------|----------|
| `bowler_career_economy` | External DB | Baseline expectation | P2 |
| `bowler_career_sr` | External DB | Wicket-taking ability | P2 |
| `bowler_economy_powerplay` | External DB | Phase specialization | P2 |
| `bowler_economy_death` | External DB | Phase specialization | P2 |
| `bowler_extras_rate` | External DB | Discipline | P3 |

### 7.3 Head-to-Head Stats

| Feature | Source | Rationale | Priority |
|---------|--------|-----------|----------|
| `h2h_balls` | External DB | Sample size | P2 |
| `h2h_runs` | External DB | Matchup score | P2 |
| `h2h_dismissals` | External DB | Matchup wickets | P2 |
| `h2h_sr` | `h2h_runs / h2h_balls * 100` | Matchup effectiveness | P2 |
| `h2h_dismissal_rate` | `h2h_dismissals / h2h_balls` | Matchup threat | P2 |

### 7.4 Venue Stats

| Feature | Source | Rationale | Priority |
|---------|--------|-----------|----------|
| `venue_avg_1st_score` | Historical data | Par score baseline | P2 |
| `venue_avg_2nd_score` | Historical data | Chase baseline | P2 |
| `venue_boundary_rate` | Historical data | Ground size effect | P3 |
| `venue_pace_friendly` | Historical data | Pitch type | P3 |

---

## Category 8: Sequence Features (For Temporal Model)

### 8.1 Ball-Level Feature Vector

For each ball in the sequence, encode:

```python
def encode_ball(delivery, match_state):
    """
    Encode a single delivery for sequence model input
    """
    return {
        # Outcome (target for previous balls, feature for current)
        'runs_batter': delivery.runs.batter,
        'runs_extras': delivery.runs.extras,
        'runs_total': delivery.runs.total,
        'is_wicket': 1 if delivery.wickets else 0,
        'is_boundary': 1 if delivery.runs.batter >= 4 else 0,
        'is_six': 1 if delivery.runs.batter == 6 else 0,
        'is_dot': 1 if delivery.runs.total == 0 else 0,

        # Context at this ball
        'over': delivery.over,
        'ball_in_over': delivery.ball_position,
        'total_score': match_state.score_at_ball,
        'total_wickets': match_state.wickets_at_ball,

        # Player IDs (for embedding lookup)
        'batsman_id': delivery.batter,
        'bowler_id': delivery.bowler,
        'non_striker_id': delivery.non_striker,

        # Extras type (categorical)
        'is_wide': 1 if delivery.extras.get('wides') else 0,
        'is_noball': 1 if delivery.extras.get('noballs') else 0,
        'is_bye': 1 if delivery.extras.get('byes') else 0,
        'is_legbye': 1 if delivery.extras.get('legbyes') else 0,
    }
```

### 8.2 Sequence Aggregations

| Feature | Computation | Use Case | Priority |
|---------|-------------|----------|----------|
| `last_n_outcomes` | Sequence of (runs, is_wicket) | Pattern detection | P1 |
| `same_bowler_sequence` | Outcomes when current bowler bowled | Bowler pattern | P1 |
| `same_batsman_sequence` | Outcomes when current batsman faced | Batsman pattern | P1 |
| `over_so_far` | Outcomes in current over | Over pattern | P1 |

---

## Category 9: Graph Node Feature Vectors

### 9.1 Batsman Node Features

```python
def build_batsman_node(batsman_id, match_state, career_stats=None):
    return {
        # Identity
        'id': batsman_id,  # For embedding lookup

        # Current innings
        'runs': match_state.batsman_runs,
        'balls': match_state.batsman_balls,
        'strike_rate': match_state.batsman_sr,
        'fours': match_state.batsman_fours,
        'sixes': match_state.batsman_sixes,
        'dots': match_state.batsman_dots,

        # Derived
        'setness': compute_setness(match_state.batsman_balls),
        'boundary_rate': match_state.batsman_boundary_rate,
        'consecutive_dots': match_state.batsman_consecutive_dots,

        # Career (if available)
        'career_sr': career_stats.sr if career_stats else 0,
        'career_avg': career_stats.avg if career_stats else 0,
    }
```

### 9.2 Bowler Node Features

```python
def build_bowler_node(bowler_id, match_state, career_stats=None):
    return {
        # Identity
        'id': bowler_id,

        # Current spell
        'overs': match_state.bowler_overs,
        'runs': match_state.bowler_runs,
        'wickets': match_state.bowler_wickets,
        'economy': match_state.bowler_economy,
        'overs_remaining': 4 - match_state.bowler_overs,

        # Derived
        'consecutive_dots': match_state.bowler_consecutive_dots,
        'last_over_runs': match_state.bowler_last_over_runs,

        # Career (if available)
        'career_economy': career_stats.economy if career_stats else 0,
        'career_sr': career_stats.sr if career_stats else 0,
    }
```

### 9.3 Match State Node Features

```python
def build_context_node(match_state, innings):
    return {
        # Score
        'runs': match_state.total_runs,
        'wickets': match_state.total_wickets,
        'run_rate': match_state.run_rate,

        # Progress
        'over': match_state.current_over,
        'ball_in_over': match_state.ball_in_over,
        'normalized_progress': match_state.balls_faced / 120,

        # Phase
        'is_powerplay': match_state.is_powerplay,
        'is_death': match_state.is_death,

        # Chase (2nd innings)
        'target': match_state.target if innings == 2 else 0,
        'required_rate': match_state.rrr if innings == 2 else 0,
        'runs_required': match_state.runs_required if innings == 2 else 0,

        # Pressure
        'pressure_index': compute_pressure_index(match_state, innings),

        # Momentum
        'momentum': compute_momentum(match_state.last_12_balls),

        # Buffers
        'wickets_remaining': 10 - match_state.total_wickets,
        'balls_remaining': 120 - match_state.balls_faced,
    }
```

### 9.4 Partner Node Features

```python
def build_partner_node(partner_id, match_state):
    return {
        # Identity
        'id': partner_id,

        # Current innings
        'runs': match_state.partner_runs,
        'balls': match_state.partner_balls,
        'strike_rate': match_state.partner_sr,

        # Partnership
        'partnership_runs': match_state.partnership_runs,
        'partnership_balls': match_state.partnership_balls,
    }
```

---

## Summary: Feature Count by Category

| Category | Feature Count | Priority Distribution |
|----------|---------------|----------------------|
| Match State | 16 | 6 P0, 6 P1, 4 P2 |
| Batsman State | 18 | 2 P0, 10 P1, 6 P2 |
| Bowler State | 15 | 0 P0, 8 P1, 7 P2 |
| Partnership | 12 | 0 P0, 4 P1, 8 P2 |
| Momentum | 12 | 0 P0, 6 P1, 6 P2 |
| Pressure | 9 | 0 P0, 5 P1, 3 P2, 1 P3 |
| Historical | 16 | 0 P0, 0 P1, 13 P2, 3 P3 |
| Sequence | 12 | 0 P0, 4 P1, 8 P2 |

**Total: ~110 derived features**

### Implementation Priority

1. **Phase 1 (MVP)**: P0 features (14 features) - Absolutely required
2. **Phase 2 (Core)**: P1 features (~47 features) - Significant improvement
3. **Phase 3 (Enhanced)**: P2 features (~45 features) - Marginal gains
4. **Phase 4 (Polish)**: P3 features (~4 features) - Minor refinement

---

## Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW CRICSHEET JSON                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     BALL ITERATOR                            │
│  For each match:                                            │
│    For each innings:                                        │
│      For each over:                                         │
│        For each delivery:                                   │
│          - Extract raw fields                               │
│          - Update running state                             │
│          - Compute derived features                         │
│          - Build graph nodes                                │
│          - Append to sequence                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT PER BALL                           │
│                                                              │
│  {                                                          │
│    'ball_id': unique identifier,                            │
│    'match_id': match identifier,                            │
│    'innings': 1 or 2,                                       │
│    'target': outcome label (runs + wicket encoding),        │
│                                                              │
│    'graph': {                                               │
│      'batsman_node': {...features...},                      │
│      'bowler_node': {...features...},                       │
│      'partner_node': {...features...},                      │
│      'context_node': {...features...},                      │
│      'edges': {...edge features...},                        │
│    },                                                        │
│                                                              │
│    'sequence': [                                            │
│      {...ball_t-N features...},                             │
│      {...ball_t-N+1 features...},                           │
│      ...                                                     │
│      {...ball_t-1 features...},                             │
│    ],                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```
