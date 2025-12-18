# Node Types Specification

## Overview

The unified graph contains **6 node type categories** with a total of **19 context nodes + N ball nodes + 1 query node** per prediction.

```
Node Type Categories
├── global     (3 nodes)   - Match-level context
├── state      (5 nodes)   - Current match state
├── actor      (7 nodes)   - Current players and matchup (includes non-striker)
├── dynamics   (4 nodes)   - Ball-to-ball momentum
├── ball       (N nodes)   - Historical deliveries
└── query      (1 node)    - Prediction target
```

---

## 1. Global Nodes (3 nodes)

These nodes represent match-level context that remains constant throughout the innings.

### 1.1 venue

**What it represents:** The cricket ground where the match is being played.

| Property | Value |
|----------|-------|
| Node type | `global` |
| Node index | 0 |
| Raw input | venue_id (integer) |
| Encoding | Learned embedding |
| Embedding dim | 32 |
| Output dim | hidden_dim (128) |

**Intuition:** Encodes ground characteristics - pitch behavior, boundary sizes, typical scores, home advantage effects.

### 1.2 batting_team

**What it represents:** The team currently batting.

| Property | Value |
|----------|-------|
| Node type | `global` |
| Node index | 1 |
| Raw input | team_id (integer) |
| Encoding | Learned embedding |
| Embedding dim | 32 |
| Output dim | hidden_dim (128) |

**Intuition:** Encodes team's batting DNA - aggressive vs defensive, strong middle order, etc.

### 1.3 bowling_team

**What it represents:** The team currently bowling.

| Property | Value |
|----------|-------|
| Node type | `global` |
| Node index | 2 |
| Raw input | team_id (integer) |
| Encoding | Learned embedding |
| Embedding dim | 32 |
| Output dim | hidden_dim (128) |

**Intuition:** Encodes team's bowling style - pace-heavy, spin-heavy, death bowling specialists.

---

## 2. State Nodes (5 nodes)

These nodes represent the current match situation. Updated after each ball.

### 2.1 score_state

**What it represents:** Current innings score.

| Property | Value |
|----------|-------|
| Node type | `state` |
| Node index | 0 |
| Input dim | 4 |

**Features:**
| Feature | Normalization | Range |
|---------|---------------|-------|
| runs | / 250 | [0, 1] |
| wickets | / 10 | [0, 1] |
| balls | / 120 | [0, 1] |
| innings | {1, 2} | binary |

### 2.2 chase_state

**What it represents:** Chase equation (2nd innings only).

| Property | Value |
|----------|-------|
| Node type | `state` |
| Node index | 1 |
| Input dim | 3 |

**Features:**
| Feature | Normalization | Range |
|---------|---------------|-------|
| runs_needed | / 250 | [0, 1] |
| required_rate | / 20 | [0, 1] |
| is_chase | boolean | {0, 1} |

### 2.3 phase_state

**What it represents:** Match phase indicators.

| Property | Value |
|----------|-------|
| Node type | `state` |
| Node index | 2 |
| Input dim | 5 |

**Features:**
| Feature | Description | Range |
|---------|-------------|-------|
| is_powerplay | Overs 1-6 | {0, 1} |
| is_middle | Overs 7-15 | {0, 1} |
| is_death | Overs 16-20 | {0, 1} |
| over_progress | ball_in_over / 6 | [0, 1] |
| is_first_ball | Cold-start indicator (first ball of innings) | {0, 1} |

### 2.4 time_pressure

**What it represents:** Time/balls remaining pressure.

| Property | Value |
|----------|-------|
| Node type | `state` |
| Node index | 3 |
| Input dim | 3 |

**Features:**
| Feature | Normalization | Range |
|---------|---------------|-------|
| balls_remaining | / 120 | [0, 1] |
| urgency | computed | [0, 1] |
| is_final_over | boolean | {0, 1} |

### 2.5 wicket_buffer

**What it represents:** Wickets in hand cushion.

| Property | Value |
|----------|-------|
| Node type | `state` |
| Node index | 4 |
| Input dim | 2 |

**Features:**
| Feature | Normalization | Range |
|---------|---------------|-------|
| wickets_in_hand | / 10 | [0, 1] |
| is_tail | wickets_in_hand < 3 | {0, 1} |

---

## 3. Actor Nodes (7 nodes)

These nodes represent the current players involved in the ball, including both batsmen and the bowler.

### 3.1 striker_identity

**What it represents:** WHO is on strike.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 0 |
| Raw input | player_id, team_id, role_id (integers) |
| Encoding | Hierarchical embedding (player → team → role fallback) |
| Embedding dim | 64 (player) + 32 (team) + 16 (role) |
| Output dim | hidden_dim (128) |

**Hierarchical Cold-Start Handling:**
For unknown players (player_id=0), the model uses team and role embeddings as fallback:
- **Known player:** Full player embedding used directly
- **Unknown player:** Team embedding + role embedding blended

**Role Categories:** unknown, opener, top_order, middle_order, finisher, bowler, allrounder, keeper

**Intuition:** Player's inherent characteristics - batting style, strengths against pace/spin, aggression level. Unknown players are distinguished by their team context and batting role.

### 3.2 striker_state

**What it represents:** HOW the striker is performing today.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 1 |
| Input dim | 7 |

**Features:**
| Feature | Normalization | Description |
|---------|---------------|-------------|
| runs_scored | / 100 | Runs in this innings |
| balls_faced | / 60 | Balls faced |
| strike_rate | / 200 | Current SR |
| dots_faced | / balls_faced | Dot ball % |
| is_set | balls_faced > 10 | Settled indicator |
| boundaries | / 10 | 4s + 6s count |
| is_debut_ball | boolean | Cold-start: first ball in match for this player |

### 3.3 nonstriker_identity

**What it represents:** WHO is at the non-striker's end.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 2 |
| Raw input | player_id, team_id, role_id (integers) |
| Encoding | Hierarchical embedding (player → team → role fallback) |
| Embedding dim | 64 (player) + 32 (team) + 16 (role) |
| Output dim | hidden_dim (128) |

**Intuition:** Non-striker's identity matters for:
- Strike rotation patterns (quick singles vs boundaries)
- Run-out risk assessment
- Partnership asymmetry (who is dominant)
- Next-ball context (who faces if single taken)

### 3.4 nonstriker_state

**What it represents:** HOW the non-striker is performing today.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 3 |
| Input dim | 7 |

**Features:** (same as striker_state)
| Feature | Normalization | Description |
|---------|---------------|-------------|
| runs_scored | / 100 | Runs in this innings |
| balls_faced | / 60 | Balls faced |
| strike_rate | / 200 | Current SR |
| dots_faced | / balls_faced | Dot ball % |
| is_set | balls_faced > 10 | Settled indicator |
| boundaries | / 10 | 4s + 6s count |
| is_debut_ball | boolean | Cold-start: first ball in match for this player |

**Intuition:** A well-set non-striker might encourage strike rotation, while a struggling non-striker might discourage singles.

### 3.5 bowler_identity

**What it represents:** WHO is bowling.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 4 |
| Raw input | player_id, team_id, role_id (integers) |
| Encoding | Hierarchical embedding (player → team → role fallback) |
| Embedding dim | 64 (player) + 32 (team) + 16 (role) |
| Output dim | hidden_dim (128) |

**Intuition:** Bowler's characteristics - pace/spin, variations, death bowling ability.

### 3.6 bowler_state

**What it represents:** HOW the bowler is performing today.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 5 |
| Input dim | 6 |

**Features:**
| Feature | Normalization | Description |
|---------|---------------|-------------|
| balls_bowled | / 24 | Balls in spell |
| runs_conceded | / 50 | Runs given |
| wickets_taken | / 5 | Wickets in spell |
| economy | / 15 | Current economy |
| dots_bowled | / balls_bowled | Dot ball % |
| threat_level | computed | Wicket threat |

### 3.7 partnership

**What it represents:** Current batting partnership.

| Property | Value |
|----------|-------|
| Node type | `actor` |
| Node index | 6 |
| Input dim | 4 |

**Features:**
| Feature | Normalization | Description |
|---------|---------------|-------------|
| partnership_runs | / 100 | Runs together |
| partnership_balls | / 60 | Balls together |
| partnership_rate | / 10 | Run rate |
| stability | computed | Partnership solidity |

---

## 4. Dynamics Nodes (4 nodes)

These nodes capture ball-to-ball momentum and pressure.

### 4.1 batting_momentum

**What it represents:** Recent scoring momentum.

| Property | Value |
|----------|-------|
| Node type | `dynamics` |
| Node index | 0 |
| Input dim | 1 |

**Features:**
| Feature | Formula | Range |
|---------|---------|-------|
| momentum | (recent_RR / expected_RR) - 1 | [-1, 1] |

### 4.2 bowling_momentum

**What it represents:** Recent bowling control.

| Property | Value |
|----------|-------|
| Node type | `dynamics` |
| Node index | 1 |
| Input dim | 1 |

**Features:**
| Feature | Formula | Range |
|---------|---------|-------|
| momentum | -batting_momentum | [-1, 1] |

### 4.3 pressure_index

**What it represents:** Composite pressure score.

| Property | Value |
|----------|-------|
| Node type | `dynamics` |
| Node index | 2 |
| Input dim | 1 |

**Features:**
| Feature | Formula | Range |
|---------|---------|-------|
| pressure | f(RRR, wickets, phase) | [0, 1] |

### 4.4 dot_pressure

**What it represents:** Consecutive dot ball pressure.

| Property | Value |
|----------|-------|
| Node type | `dynamics` |
| Node index | 3 |
| Input dim | 2 |

**Features:**
| Feature | Normalization | Description |
|---------|---------------|-------------|
| consecutive_dots | / 6 | Dots in a row |
| balls_since_boundary | / 12 | Since last 4/6 |

---

## 5. Ball Nodes (N nodes)

Each historical delivery is a node. N = number of balls bowled so far in innings.

### ball

**What it represents:** A single delivery in the innings history.

| Property | Value |
|----------|-------|
| Node type | `ball` |
| Node index | 0 to N-1 |
| Input dim | 17 + embeddings |

**Features (17 dimensions):**

| Feature | Encoding | Description |
|---------|----------|-------------|
| runs | normalized / 6 | Runs scored |
| is_wicket | boolean | Wicket fell |
| over | / 20 | Which over |
| ball_in_over | / 6 | Ball number |
| is_boundary | boolean | 4 or 6 |
| is_wide | boolean | Wide ball |
| is_noball | boolean | No ball |
| is_bye | boolean | Bye runs |
| is_legbye | boolean | Leg bye runs |
| wicket_bowled | boolean | Bowled dismissal |
| wicket_caught | boolean | Caught dismissal |
| wicket_lbw | boolean | LBW dismissal |
| wicket_run_out | boolean | Run out dismissal |
| wicket_stumped | boolean | Stumped dismissal |
| wicket_other | boolean | Other dismissals |
| striker_run_out | boolean | Striker was run out (attribution) |
| nonstriker_run_out | boolean | Non-striker was run out (attribution) |

**Run-Out Attribution:** The last two features disambiguate WHO was run out when a run-out occurs. This matters for risk assessment - the striker being run out typically indicates aggressive running, while the non-striker being run out often indicates backing up issues or direct hits.

**Player embeddings:**
| Feature | Encoding | Description |
|---------|----------|-------------|
| bowler_id | embedding (64d) | Who bowled |
| batsman_id | embedding (64d) | Who faced |

**Total input dim:** 17 + 64 + 64 = 145 → projected to hidden_dim (128)

**Why extras matter:**
- Wides/no-balls indicate bowler control issues
- Byes suggest keeper/pitch conditions
- Leg byes can indicate batsman discomfort with deliveries

**Why wicket types matter:**
- Bowled/LBW: Bowler skill, good line and length
- Caught: Risk-taking behavior by batsman
- Run out: Partnership running decisions and risk (now with attribution)
- Stumped: Batsman error against spin bowling

---

## 6. Query Node (1 node)

The prediction target node that aggregates information.

### query

**What it represents:** "What happens on the next ball?"

| Property | Value |
|----------|-------|
| Node type | `query` |
| Node index | 0 |
| Input | Learned embedding OR zeros |
| Input dim | hidden_dim (128) |

**Role:**
- Has edges to all other nodes
- Aggregates information through message passing
- Final representation used for prediction

---

## Summary Table

| Node Type | Count | Input Dim | Role |
|-----------|-------|-----------|------|
| global | 3 | 32 (embed) | Match context |
| state | 5 | 2-5 each | Current situation (phase_state now 5) |
| actor | 7 | 4-64 each | Players & matchup (hierarchical embeddings) |
| dynamics | 4 | 1-2 each | Momentum |
| ball | N | 145 | History (17 features + 2×64 embeddings) |
| query | 1 | 128 | Prediction target |
| **Total** | **21 + N** | - | - |

For a typical innings at ball 60: 21 + 60 = 81 nodes per prediction.

### Ball Feature Breakdown

| Category | Features | Count |
|----------|----------|-------|
| Basic | runs, is_wicket, over, ball_in_over, is_boundary | 5 |
| Extras | is_wide, is_noball, is_bye, is_legbye | 4 |
| Wicket Types | bowled, caught, lbw, run_out, stumped, other | 6 |
| Run-Out Attribution | striker_run_out, nonstriker_run_out | 2 |
| **Total** | | **17** |

### Cold-Start Handling

The architecture includes several cold-start indicators:
- **phase_state.is_first_ball:** Indicates the very first ball of an innings
- **striker_state.is_debut_ball:** Player's first ball in this match
- **nonstriker_state.is_debut_ball:** Player's first ball in this match
- **Hierarchical player embeddings:** Unknown players use team+role fallback instead of zero vectors
