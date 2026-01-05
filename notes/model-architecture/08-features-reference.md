# Features Reference

## Overview

This document provides a complete reference for all features in the CricketHeteroGNN model.

**Source**: `src/data/feature_utils.py`, `src/data/hetero_data_builder.py`

## Node Feature Dimensions

| Node Type | Dim | Input Type |
|-----------|-----|------------|
| venue | 1 | ID (int) |
| batting_team | 1 | ID (int) |
| bowling_team | 1 | ID (int) |
| striker_identity | 1 | ID + team_id + role_id |
| nonstriker_identity | 1 | ID + team_id + role_id |
| bowler_identity | 1 | ID + team_id + role_id |
| score_state | 5 | floats |
| chase_state | 7 | floats |
| phase_state | 6 | floats |
| time_pressure | 3 | floats |
| wicket_buffer | 2 | floats |
| striker_state | 8 | floats |
| nonstriker_state | 8 | floats |
| bowler_state | 8 | floats |
| partnership | 4 | floats |
| batting_momentum | 1 | float |
| bowling_momentum | 1 | float |
| pressure_index | 1 | float |
| dot_pressure | 5 | floats |
| ball | 18 | floats + IDs |
| query | 1 | (learned) |

## Entity Nodes

### venue

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | venue_id | `entity_mapper.get_venue_id(venue_name)` |

**Notes**: ID 0 = unknown venue

### batting_team / bowling_team

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | team_id | `entity_mapper.get_team_id(team_name)` |

**Notes**: ID 0 = unknown team

### striker_identity / nonstriker_identity / bowler_identity

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | player_id | `entity_mapper.get_player_id(player_name)` |

**Additional attributes** (for hierarchical embedding):
- `team_id`: Team the player belongs to
- `role_id`: Player role (0-7)

**Role Categories**:
| ID | Role | Description |
|----|------|-------------|
| 0 | unknown | Fallback |
| 1 | opener | Opening batsman |
| 2 | top_order | Positions 3-4 |
| 3 | middle_order | Positions 5-6 |
| 4 | finisher | Death overs specialist |
| 5 | bowler | Primarily a bowler |
| 6 | allrounder | Genuine all-rounder |
| 7 | keeper | Wicket-keeper |

## State Nodes

### score_state (5 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | runs_norm | [0, 1] | `min(runs / 250, 1.0)` |
| 1 | wickets_norm | [0, 1] | `wickets / 10` |
| 2 | balls_norm | [0, 1] | `min(balls / 120, 1.0)` |
| 3 | innings_indicator | {0, 1} | `1 if innings == 2 else 0` |
| 4 | is_womens_cricket | {0, 1} | `1 if gender == 'female' else 0` |

### chase_state (7 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | runs_needed_norm | [0, 1] | `min(runs_needed / 250, 1.0)` |
| 1 | rrr_norm | [0, 1] | `min(required_rate / 20, 1.0)` |
| 2 | is_chase | {0, 1} | `1 if innings == 2 and target exists` |
| 3 | rrr_par_norm | [0, 1] | `min(required_rate / 12, 1.0)` |
| 4 | chase_difficulty | [0, 1] | Categorical: 0 (comfortable) to 1 (improbable) |
| 5 | balls_remaining_norm | [0, 1] | `balls_remaining / 120` |
| 6 | wickets_remaining_norm | [0, 1] | `wickets_remaining / 10` |

**chase_difficulty categories**:
- 0.0: RRR < 6 (comfortable)
- 0.25: RRR 6-8 (gettable)
- 0.5: RRR 8-10 (challenging)
- 0.75: RRR 10-12 (difficult)
- 1.0: RRR > 12 (improbable)

### phase_state (6 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | is_powerplay | {0, 1} | `1 if over < 6` |
| 1 | is_middle | {0, 1} | `1 if 6 <= over < 16` |
| 2 | is_death | {0, 1} | `1 if over >= 16` |
| 3 | over_progress | [0, 1] | `ball_in_over / 6` |
| 4 | is_first_ball | {0, 1} | `1 if balls_bowled == 0` |
| 5 | is_super_over | {0, 1} | `1 if super_over` |

### time_pressure (3 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | balls_remaining_norm | [0, 1] | `balls_remaining / 120` |
| 1 | urgency | [0, 1] | `min(max(RRR - 8, 0) / 12, 1.0)` (2nd innings only) |
| 2 | is_final_over | {0, 1} | `1 if over >= 19` |

### wicket_buffer (2 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | wickets_in_hand_norm | [0, 1] | `(10 - wickets) / 10` |
| 1 | is_tail | {0, 1} | `1 if wickets_in_hand < 3` |

## Actor State Nodes

### striker_state (8 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | runs_norm | [0, 1] | `min(runs / 100, 1.0)` |
| 1 | balls_norm | [0, 1] | `min(balls_faced / 60, 1.0)` |
| 2 | sr_norm | [0, 1] | `min(strike_rate / 200, 1.0)` |
| 3 | dots_pct | [0, 1] | `dot_balls / balls_faced` |
| 4 | is_set | {0, 1} | `1 if balls_faced > 10` |
| 5 | boundaries_norm | [0, 1] | `min(boundaries / 10, 1.0)` |
| 6 | is_debut_ball | {0, 1} | `1 if balls_faced == 0` |
| 7 | balls_since_on_strike | [0, 1] | `min(balls_since_last_faced / 12, 1.0)` |

**balls_since_on_strike**: Captures "cold restart" - batsman hasn't faced for a while.

### nonstriker_state (8 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0-6 | (same as striker_state) | | |
| 7 | balls_since_as_nonstriker | [0, 1] | `min(balls_since_last_faced / 12, 1.0)` |

**Note**: Z2 symmetric with striker_state.

### bowler_state (8 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | balls_norm | [0, 1] | `min(balls_bowled / 24, 1.0)` (4 overs) |
| 1 | runs_norm | [0, 1] | `min(runs_conceded / 50, 1.0)` |
| 2 | wickets_norm | [0, 1] | `min(wickets / 5, 1.0)` |
| 3 | economy_norm | [0, 1] | `min(economy / 15, 1.0)` |
| 4 | dots_pct | [0, 1] | `dot_balls / balls_bowled` |
| 5 | threat | [0, 1] | Combined wicket-taking + economy |
| 6 | is_pace | {0, 1} | `1 if bowling_type == 'pace'` |
| 7 | is_spin | {0, 1} | `1 if bowling_type == 'spin'` |

**Bowling types**: pace, spin, unknown (both 0)

### partnership (4 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | runs_norm | [0, 1] | `min(partnership_runs / 100, 1.0)` |
| 1 | balls_norm | [0, 1] | `min(partnership_balls / 60, 1.0)` |
| 2 | run_rate_norm | [0, 1] | `min(run_rate / 10, 1.0)` |
| 3 | stability | [0, 1] | `min(partnership_balls / 30, 1.0)` |

## Dynamics Nodes

### batting_momentum (1 feature)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | momentum | [-1, 1] | `(recent_rr / expected_rr) - 1` clamped |

**Expected run rate by phase**:
- Powerplay: 7.5
- Middle: 8.0
- Death: 10.0

### bowling_momentum (1 feature)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | momentum | [-1, 1] | `-batting_momentum` |

### pressure_index (1 feature)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | pressure | [0, 1] | Weighted: RRR (40%) + wickets (30%) + time (30%) |

### dot_pressure (5 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | consecutive_dots | [0, 1] | `min(consecutive_dots / 6, 1.0)` |
| 1 | balls_since_boundary | [0, 1] | `min(balls_since_boundary / 12, 1.0)` |
| 2 | balls_since_wicket | [0, 1] | `min(balls_since_wicket / 30, 1.0)` |
| 3 | pressure_accumulated | [0, 1] | Exponentially-weighted recent pressure |
| 4 | pressure_trend | [-1, 1] | First half vs second half of recent window |

## Ball Nodes (18 features)

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | runs_norm | [0, 1] | `runs / 6` |
| 1 | is_wicket | {0, 1} | `1 if wickets in delivery` |
| 2 | over_norm | [0, 1] | `over / 20` |
| 3 | ball_in_over_norm | [0, 1] | `ball_in_over / 6` |
| 4 | is_boundary | {0, 1} | `1 if batter_runs in [4, 6]` |
| 5 | is_wide | {0, 1} | `1 if 'wides' in extras` |
| 6 | is_noball | {0, 1} | `1 if 'noballs' in extras` |
| 7 | is_bye | {0, 1} | `1 if 'byes' in extras` |
| 8 | is_legbye | {0, 1} | `1 if 'legbyes' in extras` |
| 9 | wicket_bowled | {0, 1} | Dismissal type one-hot |
| 10 | wicket_caught | {0, 1} | Dismissal type one-hot |
| 11 | wicket_lbw | {0, 1} | Dismissal type one-hot |
| 12 | wicket_run_out | {0, 1} | Dismissal type one-hot |
| 13 | wicket_stumped | {0, 1} | Dismissal type one-hot |
| 14 | wicket_other | {0, 1} | Other dismissal types |
| 15 | striker_run_out | {0, 1} | Run-out attribution |
| 16 | nonstriker_run_out | {0, 1} | Run-out attribution |
| 17 | bowling_end | {0, 1} | `over % 2` (which end of pitch) |

**Additional attributes** (not in .x):
- `bowler_ids`: ID of bowler for this ball
- `batsman_ids`: ID of batsman who faced
- `nonstriker_ids`: ID of non-striker

## Query Node

| Index | Feature | Range | Computation |
|-------|---------|-------|-------------|
| 0 | placeholder | 0.0 | Replaced by learned embedding |

**Note**: Query encoder uses `nn.Parameter`, the input `.x` is ignored.

## Output Classes

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Dot | 0 runs, no wicket |
| 1 | Single | 1 run |
| 2 | Two | 2 runs |
| 3 | Three | 3 runs |
| 4 | Four | Boundary (4) |
| 5 | Six | Boundary (6) |
| 6 | Wicket | Dismissal occurred |

**Priority**: Wicket takes precedence over runs.

## Conditioning Signal (FiLM)

When using `CricketHeteroGNNFull` with phase modulation:

```python
condition = torch.cat([
    phase_state,    # [batch, 6]
    chase_state,    # [batch, 7]
    wicket_buffer,  # [batch, 2]
], dim=-1)          # [batch, 15]
```

| Indices | Source | Features |
|---------|--------|----------|
| 0-5 | phase_state | is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over |
| 6-12 | chase_state | runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem |
| 13-14 | wicket_buffer | wickets_in_hand, is_tail |

## Feature Validation

```python
from src.data.hetero_data_builder import get_node_feature_dims, validate_hetero_data

# Get expected dimensions
dims = get_node_feature_dims()
# {'venue': 1, 'batting_team': 1, ..., 'ball': 18, ...}

# Validate a sample
is_valid = validate_hetero_data(sample)
# Raises AssertionError if invalid
```

---

*Next: [09-forward-pass-walkthrough.md](./09-forward-pass-walkthrough.md) - Step-by-step data flow through the model*
