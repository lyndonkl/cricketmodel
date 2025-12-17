# Edge Types Specification

## Overview

The unified graph uses **typed edges** to encode different relationships. Each edge type can have its own convolution operator, allowing the model to learn relationship-specific message passing.

```
Edge Type Categories
├── Hierarchical (3 types)   - Top-down conditioning between layers
├── Intra-layer (4 types)    - Within-layer interactions
├── Temporal (4 types)       - Ball history structure (incl. same_matchup)
├── Cross-domain (4 types)   - Connecting balls to context (incl. partnered_by)
└── Query (1 type)           - Prediction aggregation
```

**Total: 16 edge types**

---

## 1. Hierarchical Edges (Top-Down Conditioning)

These edges encode the cricket hierarchy: global context shapes match state, which shapes actor performance, which shapes dynamics.

### 1.1 (global, conditions, state)

**Semantics:** Venue and team context influences interpretation of match state.

```
global nodes                    state nodes
┌─────────┐                    ┌─────────────┐
│ venue   │ ──────────────────►│ score_state │
├─────────┤                    ├─────────────┤
│ bat_team│ ──────────────────►│ chase_state │
├─────────┤                    ├─────────────┤
│bowl_team│ ──────────────────►│ phase_state │
└─────────┘        ...         ├─────────────┤
                               │time_pressure│
     (all-to-all)              ├─────────────┤
                               │wicket_buffer│
                               └─────────────┘
```

| Property | Value |
|----------|-------|
| Source type | global |
| Edge type | conditions |
| Target type | state |
| Connectivity | All-to-all (3 × 5 = 15 edges) |
| Direction | global → state |

**Intuition:** "This is the MCG (high-scoring ground), so interpret 150/3 differently than at Chennai (slow pitch)."

### 1.2 (state, conditions, actor)

**Semantics:** Match situation influences how we interpret player performance.

| Property | Value |
|----------|-------|
| Source type | state |
| Edge type | conditions |
| Target type | actor |
| Connectivity | All-to-all (5 × 7 = 35 edges) |
| Direction | state → actor |

**Intuition:** "Needing 80 off 30 balls (state) means Rohit's 20(12) is good, but needing 20 off 30 means it's conservative."

### 1.3 (actor, conditions, dynamics)

**Semantics:** Current players influence interpretation of momentum.

| Property | Value |
|----------|-------|
| Source type | actor |
| Edge type | conditions |
| Target type | dynamics |
| Connectivity | All-to-all (7 × 4 = 28 edges) |
| Direction | actor → dynamics |

**Intuition:** "Bumrah bowling (actor) means 3 dots is expected, not pressure. But 3 dots from a part-timer is different."

---

## 2. Intra-Layer Edges (Within-Layer Interactions)

These edges allow nodes within the same semantic layer to exchange information.

### 2.1 (global, relates_to, global)

**Semantics:** Venue and teams interact (e.g., home advantage).

```
venue ◄──────► batting_team
  │                │
  │                │
  ▼                ▼
bowling_team ◄─────┘
```

| Property | Value |
|----------|-------|
| Source type | global |
| Edge type | relates_to |
| Target type | global |
| Connectivity | Fully connected (6 edges, bidirectional) |
| Direction | Bidirectional |

**Intuition:** "India at home vs Australia - the venue-team combination matters."

### 2.2 (state, relates_to, state)

**Semantics:** Match state components interact.

| Property | Value |
|----------|-------|
| Source type | state |
| Edge type | relates_to |
| Target type | state |
| Connectivity | Fully connected (20 edges, bidirectional) |
| Direction | Bidirectional |

**Intuition:** "Chase target (chase_state) relates to required rate, which relates to time pressure."

### 2.3 (actor, matchup, actor)

**Semantics:** The specific batter-bowler interactions, including non-striker.

```
striker_id ◄──────► striker_state
     │                    │
     │    THE MATCHUP     │
     ▼                    ▼
bowler_id  ◄──────► bowler_state
     │                    │
     └──────► partnership ◄┘
                  ▲
nonstriker_id ◄───┼───► nonstriker_state
                  │
     (non-striker contributes to partnership)
```

| Property | Value |
|----------|-------|
| Source type | actor |
| Edge type | matchup |
| Target type | actor |
| Connectivity | Semantic (22 edges bidirectional) |
| Direction | Bidirectional |

**Specific edges:**
- striker_id ↔ striker_state (identity linked to performance)
- nonstriker_id ↔ nonstriker_state (identity linked to performance)
- bowler_id ↔ bowler_state (identity linked to performance)
- **striker_id ↔ bowler_id** (THE KEY MATCHUP)
- **nonstriker_id ↔ bowler_id** (secondary matchup - run-out risk)
- **striker_id ↔ nonstriker_id** (partnership chemistry)
- striker_state ↔ partnership
- nonstriker_state ↔ partnership
- bowler_state ↔ partnership
- striker_id ↔ partnership
- nonstriker_id ↔ partnership

**Intuition:** "Rohit vs Bumrah - their identities interact (historical matchup), and both batsmen contribute to the partnership dynamics. Non-striker's running ability affects single-taking."

### 2.4 (dynamics, relates_to, dynamics)

**Semantics:** Momentum components interact.

| Property | Value |
|----------|-------|
| Source type | dynamics |
| Edge type | relates_to |
| Target type | dynamics |
| Connectivity | Fully connected (12 edges, bidirectional) |
| Direction | Bidirectional |

**Intuition:** "Batting momentum relates to bowling momentum (zero-sum) and both contribute to pressure."

---

## 3. Temporal Edges (Ball History Structure)

These edges encode the structure of the ball-by-ball history. **This is where we gain efficiency over sequence attention.**

### 3.1 (ball, precedes, ball)

**Semantics:** Temporal/causal ordering of deliveries.

```
Ball 1 ──► Ball 2 ──► Ball 3 ──► ... ──► Ball N
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | precedes |
| Target type | ball |
| Connectivity | Sequential (N-1 edges) |
| Direction | past → future (respects causality) |

**Construction:** For balls i, j where i < j and j = i + 1, create edge (i, precedes, j).

**Intuition:** "What happened on the previous ball directly influences the next."

**Note:** We could extend to k-hop temporal edges (ball i connects to balls i+1, i+2, ..., i+k) if needed.

### 3.2 (ball, same_bowler, ball)

**Semantics:** Balls delivered by the same bowler.

```
Bumrah's balls:  B3 ◄──► B9 ◄──► B15 ◄──► B21 ◄──► ...
Starc's balls:   B1 ◄──► B7 ◄──► B13 ◄──► B19 ◄──► ...
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | same_bowler |
| Target type | ball |
| Connectivity | Clique within bowler (sparse overall) |
| Direction | Bidirectional |

**Construction:** For balls i, j where bowler(i) == bowler(j), create edge (i, same_bowler, j).

**Intuition:** "All of Bumrah's deliveries form a pattern - his spell narrative."

**Efficiency:** If bowler A bowled 20 balls, that's 20×19/2 = 190 edges for that bowler. With ~6 bowlers, total ~600 edges. Much less than 120×120 = 14,400 for full attention!

### 3.3 (ball, same_batsman, ball)

**Semantics:** Balls faced by the same batsman.

```
Rohit's balls:   B2 ◄──► B4 ◄──► B8 ◄──► B10 ◄──► ...
Kohli's balls:   B1 ◄──► B3 ◄──► B7 ◄──► B11 ◄──► ...
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | same_batsman |
| Target type | ball |
| Connectivity | Clique within batsman (sparse overall) |
| Direction | Bidirectional |

**Construction:** For balls i, j where batsman(i) == batsman(j), create edge (i, same_batsman, j).

**Intuition:** "All balls Rohit faced form his innings narrative - building, accelerating, getting out."

### 3.4 (ball, same_matchup, ball)

**Semantics:** Balls with the same bowler-batsman combination (the key predictor).

```
Bumrah → Rohit:  B3 ◄──► B21 ◄──► B45 ◄──► ...
Starc → Kohli:   B7 ◄──► B25 ◄──► B31 ◄──► ...
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | same_matchup |
| Target type | ball |
| Connectivity | Clique within matchup (very sparse) |
| Direction | Bidirectional |

**Construction:** For balls i, j where bowler(i) == bowler(j) AND batsman(i) == batsman(j), create edge (i, same_matchup, j).

**Intuition:** "How has this specific bowler-batsman matchup played out? This is THE key predictor for the next ball."

**Efficiency:** Same_matchup is the intersection of same_bowler and same_batsman, so it's very sparse but highly informative.

---

## 4. Cross-Domain Edges (Connecting Balls to Context)

These edges link the ball history to the current context nodes.

### 4.1 (ball, bowled_by, actor)

**Semantics:** Links each ball to the bowler who delivered it.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | bowled_by |
| Target type | actor (specifically bowler_identity) |
| Connectivity | Each ball → one bowler (N edges) |
| Direction | ball → actor |

**Note:** Only connects to `bowler_identity` node (actor index 2), not all actor nodes.

**Intuition:** "This ball was delivered by Bumrah, so it connects to Bumrah's identity node."

### 4.2 (ball, faced_by, actor)

**Semantics:** Links each ball to the batsman who faced it.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | faced_by |
| Target type | actor (specifically striker_identity) |
| Connectivity | Each ball → one batsman (N edges) |
| Direction | ball → actor |

**Note:** Only connects to `striker_identity` node (actor index 0), not all actor nodes.

**Intuition:** "This ball was faced by Rohit, so it connects to Rohit's identity node."

### 4.3 (ball, partnered_by, actor)

**Semantics:** Links each ball to the non-striker who was at the other end.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | partnered_by |
| Target type | actor (specifically nonstriker_identity) |
| Connectivity | Each ball → one non-striker (N edges) |
| Direction | ball → actor |

**Note:** Only connects to `nonstriker_identity` node (actor index 2), not all actor nodes.

**Intuition:** "Who was at the other end when this ball was bowled? This context matters for understanding run-taking patterns and partnership dynamics."

### 4.4 (dynamics, reflects, ball)

**Semantics:** Recent balls inform current momentum.

| Property | Value |
|----------|-------|
| Source type | dynamics |
| Edge type | reflects |
| Target type | ball (recent balls only) |
| Connectivity | Dynamics → last K balls (4 × K edges) |
| Direction | dynamics ← ball (ball informs dynamics) |

**Note:** Only connects to the most recent K balls (e.g., K=12 for last 2 overs).

**Intuition:** "Current momentum is derived from what happened in the last 2 overs."

---

## 5. Query Edges (Prediction Aggregation)

The query node needs to gather information from everywhere.

### 5.1 (query, attends_to, *)

**Semantics:** Query node attends to all other nodes.

| Property | Value |
|----------|-------|
| Source type | query |
| Edge type | attends_to |
| Target type | ALL (global, state, actor, dynamics, ball) |
| Connectivity | 1 → all (19 + N edges) |
| Direction | query ← all (information flows to query) |

**Intuition:** "To predict the next ball, consider everything: venue, match state, players (including non-striker), momentum, and all history."

**Alternative design:** Could have separate edge types per target:
- (query, attends_global, global)
- (query, attends_state, state)
- (query, attends_actor, actor)
- (query, attends_dynamics, dynamics)
- (query, attends_ball, ball)

This allows learning different attention patterns for each category.

---

## Edge Count Analysis

For a typical innings at ball 60:

| Edge Type | Formula | Count |
|-----------|---------|-------|
| Hierarchical conditioning | 15 + 35 + 28 | 78 |
| Intra-layer | 6 + 20 + 22 + 12 | 60 |
| Temporal precedes | N - 1 | 59 |
| Same bowler | ~6 bowlers × C(10,2) | ~270 |
| Same batsman | ~4 batsmen × C(15,2) | ~420 |
| Same matchup | ~12 matchups × C(5,2) | ~120 |
| Ball → actor (faced_by + bowled_by + partnered_by) | 3N | 180 |
| Dynamics ← ball | 4 × 12 | 48 |
| Query | 19 + N | 79 |
| **Total** | | **~1300** |

Compare to V1 Transformer: 60 × 60 = 3,600 attention pairs (temporal only!)

**Efficiency gain: ~2.7x fewer computations, with MORE information (full context graph including non-striker)**

---

## Summary Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              QUERY NODE                 │
                    │         (attends_to everything)         │
                    └────────────────┬────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     GLOBAL      │       │      STATE      │       │    DYNAMICS     │
│  (relates_to)   │       │  (relates_to)   │       │  (relates_to)   │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │ conditions              │ conditions              │ reflects
         ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │◄──────│ ACTOR (7 nodes) │──────►│                 │
│                 │       │    (matchup)    │       │                 │
│                 │       │ striker/nonstr/ │       │                 │
│                 │       │ bowler + states │       │   BALL NODES    │
│                 │       └────────┬────────┘       │                 │
│                 │                │                │  ┌──► same_bowler
│                 │    ┌───────────┼───────────┐    │  │
│                 │    │           │           │    │  ├──► same_batsman
│                 │ bowled_by   faced_by  partnered │  │
│                 │    │           │        _by│    │  ├──► same_matchup
│                 │    ▼           ▼           ▼    │  │
│                 │       ┌─────────────────┐       │  └──► precedes
│                 │       │   BALL NODES    │◄──────┤
│                 │       │   (precedes)    │       │
│                 │       └─────────────────┘       │
└─────────────────┘                                 └─────────────────┘
```
