# Edge Types Specification

## Overview

The unified graph uses **typed edges** to encode different relationships. Each edge type can have its own convolution operator, allowing the model to learn relationship-specific message passing.

```
Edge Type Categories
├── Hierarchical (3 types)   - Top-down conditioning between layers
├── Intra-layer (4 types)    - Within-layer interactions
├── Temporal (6 types)       - Ball history structure (multi-scale + matchup edges)
├── Cross-domain (4 types)   - Connecting balls to context (incl. partnered_by)
└── Query (2 types)          - Prediction aggregation (attends + drives)
```

**Total: 19 edge types**

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

### Multi-Scale Temporal Architecture

Cricket temporal patterns operate at multiple scales:
- **Recent (6 balls):** Within-over context, immediate pressure
- **Medium (7-18 balls):** 2-over momentum window
- **Distant (19+ balls):** Historical patterns, phase transitions

We use three separate edge types to capture these patterns, each with appropriate convolution operators.

### 3.1 (ball, recent_precedes, ball)

**Semantics:** Recent temporal context within the current over.

```
Ball N-5 ──► Ball N-4 ──► Ball N-3 ──► Ball N-2 ──► Ball N-1 ──► Ball N
         (d=0.17)    (d=0.33)     (d=0.50)     (d=0.67)     (d=0.83)
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | recent_precedes |
| Target type | ball |
| Connectivity | All pairs within 6-ball window |
| Direction | past → future (respects causality) |
| **Edge features** | `gap / 6.0` (normalized distance) |
| Convolution | TransformerConv with edge_dim=1 (fast decay) |

**Construction:** For balls i, j where j - i ≤ 6, create edge (i, recent_precedes, j).

**Intuition:** "The last 6 balls form the immediate context - what just happened affects what's next."

### 3.2 (ball, medium_precedes, ball)

**Semantics:** Medium-range momentum window (2-over context).

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | medium_precedes |
| Target type | ball |
| Connectivity | All pairs with gap 7-18 balls |
| Direction | past → future (respects causality) |
| **Edge features** | `(gap - 6) / 12.0` (normalized distance) |
| Convolution | TransformerConv with edge_dim=1 (medium decay) |

**Construction:** For balls i, j where 7 ≤ j - i ≤ 18, create edge (i, medium_precedes, j).

**Intuition:** "The 2-over momentum window captures scoring patterns and bowler effectiveness."

### 3.3 (ball, distant_precedes, ball)

**Semantics:** Historical patterns and phase transitions.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | distant_precedes |
| Target type | ball |
| Connectivity | Sparse (every 6 balls for gaps > 18) |
| Direction | past → future (respects causality) |
| **Edge features** | `(gap - 18) / max_distance` |
| Convolution | SAGEConv mean aggregation (slow decay) |

**Construction:** For balls i, j where j - i > 18 AND (j - i) % 6 == 0, create edge (i, distant_precedes, j).

**Intuition:** "Historical context is sparse but important for phase-level patterns."

### 3.4 (ball, same_bowler, ball)

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

### 3.5 (ball, same_batsman, ball)

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

### 3.6 (ball, same_matchup, ball) - CAUSAL

**Semantics:** Balls with the same bowler-batsman combination (the key predictor).

```
Bumrah → Rohit:  B3 ──► B21 ──► B45 ──► ...   (older → newer ONLY)
Starc → Kohli:   B7 ──► B25 ──► B31 ──► ...
```

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | same_matchup |
| Target type | ball |
| Connectivity | **CAUSAL** within matchup (older → newer) |
| Direction | **Unidirectional (older → newer)** |

**CRITICAL: CAUSAL EDGES ONLY**

Unlike same_bowler and same_batsman (bidirectional), same_matchup edges are **CAUSAL** (older → newer only). This is essential to prevent train-test distribution shift:

- During training, the graph contains future balls
- Bidirectional edges would allow "future information leakage"
- At inference, only historical balls exist
- Causal edges ensure model learns patterns valid at inference time

**Construction:** For balls i, j where bowler(i) == bowler(j) AND batsman(i) == batsman(j) AND i < j, create edge (i, same_matchup, j).

**Intuition:** "How has this specific bowler-batsman matchup played out SO FAR? Past matchup outcomes inform future predictions."

**Efficiency:** Same_matchup is the intersection of same_bowler and same_batsman, so it's very sparse but highly informative.

---

## 4. Cross-Domain Edges (Connecting Balls to Context)

These edges link the ball history to the current context nodes.

**CRITICAL: Correct Player Attribution (Z2 Symmetry Respect)**

Cross-domain edges connect ONLY to balls involving the CURRENT players. This respects the Z2 symmetry of striker/non-striker swapping:

- `faced_by`: Only balls that the CURRENT striker actually faced
- `partnered_by`: Only balls where CURRENT non-striker was involved (as striker or non-striker)
- `bowled_by`: Only balls that the CURRENT bowler actually bowled

This is semantically correct: if Kohli faced balls 1-20 then got out, and Sharma is now batting, balls 1-20 should NOT connect to striker_identity (which represents Sharma). Only balls Sharma actually faced will connect.

### 4.1 (ball, bowled_by, actor)

**Semantics:** Links balls to the CURRENT bowler's identity IF they actually bowled those balls.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | bowled_by |
| Target type | actor (specifically bowler_identity) |
| Connectivity | Only balls bowled by current bowler (variable, ≤ N edges) |
| Direction | ball → actor |
| Convolution | **GATv2Conv** (attention-weighted aggregation) |

**Construction:** For each historical ball i where `bowler(i) == current_bowler`, create edge (i, bowled_by, bowler_identity).

**Intuition:** "Show me only the balls from the current bowler's spell - how has THIS bowler been bowling today?"

### 4.2 (ball, faced_by, actor)

**Semantics:** Links balls to the CURRENT striker's identity IF they actually faced those balls.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | faced_by |
| Target type | actor (specifically striker_identity) |
| Connectivity | Only balls faced by current striker (variable, ≤ N edges) |
| Direction | ball → actor |
| Convolution | **GATv2Conv** (attention-weighted aggregation) |

**Construction:** For each historical ball i where `batsman(i) == current_striker`, create edge (i, faced_by, striker_identity).

**Intuition:** "Show me only the balls the current striker has faced - how have THEY been batting?"

### 4.3 (ball, partnered_by, actor)

**Semantics:** Links balls to the CURRENT non-striker's identity IF they were involved.

| Property | Value |
|----------|-------|
| Source type | ball |
| Edge type | partnered_by |
| Target type | actor (specifically nonstriker_identity) |
| Connectivity | Balls where current non-striker was involved (variable, ≤ N edges) |
| Direction | ball → actor |
| Convolution | **GATv2Conv** (attention-weighted aggregation) |

**Construction:** For each historical ball i where `nonstriker(i) == current_nonstriker` OR `batsman(i) == current_nonstriker`, create edge (i, partnered_by, nonstriker_identity).

**Intuition:** "What has the current non-striker's involvement been? Both when they faced (before swap) and when they partnered."

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

### 5.2 (dynamics, drives, query)

**Semantics:** Dynamics nodes directly drive prediction via the query node.

| Property | Value |
|----------|-------|
| Source type | dynamics |
| Edge type | drives |
| Target type | query |
| Connectivity | 4 → 1 (4 edges) |
| Direction | dynamics → query |
| Convolution | **GATv2Conv** (attention-weighted) |

**Specific edges:**
- batting_momentum → query
- bowling_momentum → query
- pressure → query
- dot_pressure → query

**Why a separate edge type (not just attends)?**

The `drives` edge type captures a distinct causal relationship: momentum and pressure **directly cause** outcome changes through feedback loops:

1. **R1 Confidence Spiral:** High batting momentum → batsman plays more shots → more runs → higher momentum
2. **B1 Required Rate Pressure:** High pressure → batsman takes risks → more boundaries OR wickets
3. **B2 Dot Ball Spiral:** High dot pressure → desperate shots → wickets → rebuilding → more dots

These feedback loops are the core of cricket dynamics. By having `drives` edges with learned attention weights, the model can:
- Weight batting vs bowling momentum appropriately for the situation
- Learn when pressure is decisive vs when momentum dominates
- Capture the tension between conservative play (dot pressure) and aggressive play (momentum)

**Intuition:** "Current momentum and pressure directly influence what happens next - this is the core feedback mechanism of cricket dynamics."

---

## Edge Count Analysis

For a typical innings at ball 60:

| Edge Type | Formula | Count |
|-----------|---------|-------|
| Hierarchical conditioning | 15 + 35 + 28 | 78 |
| Intra-layer | 6 + 20 + 22 + 12 | 60 |
| **Multi-scale temporal:** | | |
| - recent_precedes (≤6 balls) | ~6 × N (recent pairs) | ~360 |
| - medium_precedes (7-18 balls) | ~12 × N (medium pairs) | ~720 |
| - distant_precedes (>18, sparse) | ~N/6 (every 6 balls) | ~300 |
| Same bowler | ~6 bowlers × C(10,2) | ~270 |
| Same batsman | ~4 batsmen × C(15,2) | ~420 |
| Same matchup (CAUSAL) | ~12 matchups × ~5 edges each | ~60 |
| Ball → actor (faced_by + bowled_by + partnered_by) | 3N | 180 |
| Dynamics ← ball (informs) | 4 × 12 | 48 |
| Query attends | 19 + N | 79 |
| Dynamics drives query | 4 | 4 |
| **Total** | | **~2,580** |

Compare to V1 Transformer: 60 × 60 = 3,600 attention pairs (temporal only!)

**Key differences from simple precedes:**
- Multi-scale temporal captures cricket-relevant time scales
- Recent (6 balls) = within-over context with fast decay
- Medium (7-18 balls) = 2-over momentum window
- Distant (sparse) = historical patterns without quadratic blowup
- Same_matchup is now CAUSAL (older→newer only), preventing train-test distribution shift

**Efficiency gain: ~1.4x fewer computations, with MORE information (full context graph including non-striker, multi-scale temporal, causal matchup edges)**

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
