# Cricket Application: GAT for Relational Modeling

## The Key Question: Is Cricket a Graph Problem?

Cricket is **primarily temporal** (ball-by-ball sequence), but it has **relational structure** that could benefit from graph modeling:

```
TEMPORAL VIEW (sequence):
Ball 1 → Ball 2 → Ball 3 → ... → Ball 300

RELATIONAL VIEW (graph):
        ┌─────────────────────────────────────┐
        │           Match State               │
        │                                     │
   Batsman A ←──────→ Bowler X               │
        │                │                    │
        ↓                ↓                    │
   Batsman B        Fielders                 │
        │                                     │
        └─────────────────────────────────────┘
```

**The question**: Can modeling these relationships with GAT improve predictions beyond pure sequence modeling?

---

## Potential Graph Structures for Cricket

### Option 1: Player Interaction Graph

**Nodes**: Players (batsmen, bowlers, fielders)
**Edges**: Interaction relationships (matchups, partnerships, etc.)

```
                    PLAYER INTERACTION GRAPH

                         ┌─────────┐
                         │ Bowler  │
                         │    A    │
                         └────┬────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │ Batsman │     │ Batsman │     │ Keeper  │
        │    1    │◄───►│    2    │     │         │
        └─────────┘     └─────────┘     └─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │Fielder 1│     │Fielder 2│     │Fielder 3│
        └─────────┘     └─────────┘     └─────────┘

Node features:
- Current performance (runs, balls, wickets)
- Career stats
- Recent form
- Matchup history

Edge features (potential extension):
- Head-to-head record
- Dismissal history
- Run-scoring patterns
```

**What GAT would learn**:
- Which player relationships matter for this ball
- Attention on bowler-batsman edge reveals matchup importance
- Partnership strength influences prediction

### Option 2: Temporal Ball Graph

**Nodes**: Individual ball deliveries
**Edges**: Temporal connections and contextual relationships

```
                    TEMPORAL BALL GRAPH

    Ball 1 ──→ Ball 2 ──→ Ball 3 ──→ Ball 4 ──→ Ball 5 ──→ Ball 6
      │          │          │          │          │          │
      └──────────┴──────────┴──────────┴──────────┴──────────┘
                         │
                    (over boundary)
                         │
    Ball 7 ──→ Ball 8 ──→ Ball 9 ──→ Ball 10 ──→ Ball 11 ──→ Ball 12
      │                                                        │
      └────────────────────────────────────────────────────────┘
                         (same bowler's previous over)

Edges:
- Sequential: Ball t → Ball t+1
- Same bowler: All balls by same bowler
- Same batsman: All balls faced by same batsman
- Over boundaries: First/last balls of overs
```

**What GAT would learn**:
- Which previous balls are most relevant
- Bowler's pattern repetition (attend to same bowler's recent balls)
- Batsman's momentum (attend to recent balls faced)

### Option 3: Match State Graph

**Nodes**: Different aspects of match state
**Edges**: Dependencies between aspects

```
                    MATCH STATE GRAPH

    ┌──────────────┐         ┌──────────────┐
    │    Score     │◄───────►│   Required   │
    │   (142/4)    │         │  Rate (9.2)  │
    └──────┬───────┘         └──────┬───────┘
           │                        │
           ▼                        ▼
    ┌──────────────┐         ┌──────────────┐
    │   Batting    │◄───────►│   Bowling    │
    │    Team      │         │    Team      │
    └──────┬───────┘         └──────┬───────┘
           │                        │
           ▼                        ▼
    ┌──────────────┐         ┌──────────────┐
    │   Current    │◄───────►│   Current    │
    │   Batsman    │         │   Bowler     │
    └──────────────┘         └──────────────┘
           │                        │
           └────────────────────────┘
                      │
               ┌──────────────┐
               │   Matchup    │
               │   Context    │
               └──────────────┘

Node features:
- Score state: runs, wickets, overs
- Required rate: target, balls remaining
- Team state: batting order, bowling resources
- Player state: current form, career vs opponent
- Matchup: head-to-head statistics
```

**What GAT would learn**:
- Under different situations, which state aspects matter most
- High pressure: attend more to required rate
- New batsman: attend more to matchup history

---

## GAT's Potential Value for Cricket

### 1. Learned Importance Weighting

Not all relationships equally matter at all times:

```
Situation A: First over of innings
- Batsman settling in
- Pitch behavior unknown
- GAT attention: High on pitch state, bowler form
                 Low on partnership, required rate

Situation B: Final over chase
- Every ball crucial
- Target pressure
- GAT attention: High on required rate, batsman form
                 Low on bowler's overall stats
```

**GAT can learn these context-dependent attention patterns**.

### 2. Interpretability

Attention weights reveal model reasoning:

```
For prediction of ball 47.5:

Attention to nodes:
- Current batsman state:     α = 0.35  ← "Batsman form matters most"
- Bowler recent performance: α = 0.25  ← "Bowler's current spell relevant"
- Required rate:             α = 0.20  ← "Chase pressure significant"
- Partnership state:         α = 0.12  ← "Partnership building"
- Pitch conditions:          α = 0.08  ← "Pitch relatively stable now"

This interpretability helps:
- Debug model decisions
- Validate domain knowledge
- Explain predictions to stakeholders
```

### 3. Inductive Capability

GAT can generalize to unseen players:

```
Training: Learn attention patterns on players A, B, C
Testing: Apply to new player D (not in training)

Because attention is computed from FEATURES:
- Player D's features → Transformed → Attention computed
- No memorization of specific players required
- Model generalizes based on feature similarity
```

---

## Challenges for Cricket GAT

### 1. Cricket is Primarily Sequential

GAT designed for **relational** data, not **temporal** data:

```
Natural GAT input:             Cricket reality:
- Social network               - Ball-by-ball sequence
- Molecule structure           - Time-ordered events
- Citation graph               - Causal dependencies

Cricket has temporal causality:
Ball t CANNOT attend to Ball t+1 (future!)
GAT assumes bidirectional relationships
```

**Solution**: Combine GAT with temporal model:
```
Ball sequence → LSTM/Transformer → Temporal features
                      ↓
Player graph → GAT → Relational features
                      ↓
              Combine → Prediction
```

### 2. Graph Structure Must Be Defined

Cricket doesn't have a natural graph structure - we must design it:

```
Design questions:
- What are the nodes? (Players? Balls? States?)
- What are the edges? (All pairs? Meaningful connections?)
- How are edges weighted? (Fixed? Dynamic?)
- Should structure change during match?

No single "correct" answer - requires experimentation
```

### 3. Dynamic Relationships

Cricket relationships change during match:

```
Over 1: Batsman A vs Bowler X (fresh matchup)
Over 10: Batsman A vs Bowler X (10 balls of data now!)

The "edge" between A and X should incorporate:
- Pre-match history
- Current match data (updating!)
- Recent ball outcomes

Static GAT: Edges fixed at start
Dynamic GAT needed: Edge features update during match
```

### 4. Scale Considerations

```
Players per match: ~22
Balls per ODI: ~600
Potential ball-level graph: 600 nodes, many edges

GAT scales as O(|E|) - manageable
But temporal attention over 600 balls might be better with Transformer
```

---

## Proposed Hybrid Architecture

### Best of Both Worlds: Temporal + Relational

```
┌─────────────────────────────────────────────────────────────┐
│                    CRICKET PREDICTION MODEL                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 TEMPORAL BRANCH                       │  │
│  │                                                       │  │
│  │  Ball sequence → [Transformer/LSTM] → h_temporal     │  │
│  │                                                       │  │
│  │  Captures: Ball-by-ball patterns, momentum           │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│                     ┌──────────────┐                       │
│                     │   COMBINE    │                       │
│                     └──────────────┘                       │
│                            ▲                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                RELATIONAL BRANCH                      │  │
│  │                                                       │  │
│  │  Player graph → [GAT] → h_relational                 │  │
│  │                                                       │  │
│  │  Captures: Player matchups, team dynamics            │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│                     ┌──────────────┐                       │
│                     │  PREDICTION  │                       │
│                     └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

**Temporal Branch (Transformer/LSTM)**:
```
Input: Recent ball sequence [b_{t-k}, ..., b_{t-1}]
Output: h_temporal ∈ R^d

Captures:
- Recent momentum (4s, 6s, dots)
- Over patterns
- Phase of innings
- Sequential dependencies
```

**Relational Branch (GAT)**:
```
Input: Player interaction graph G = (V, E)
       Node features: current match stats + career stats
Output: h_relational ∈ R^d

Captures:
- Batsman-bowler matchup strength
- Partnership synergy
- Team balance
- Tactical relationships
```

**Combination**:
```
h_combined = MLP([h_temporal || h_relational])
prediction = softmax(W · h_combined)

Or more sophisticated fusion:
h_combined = CrossAttention(h_temporal, h_relational)
```

---

## Concrete Example: Death Over Prediction

### Setup

```
Match situation:
- Over 48, Ball 4 (T20 chase)
- Score: 142/4, Target: 168
- Batsman: Player A (45 off 32, set)
- Bowler: Player X (death specialist)
- Required: 26 off 10 balls
```

### Temporal Branch Input

```
Recent ball sequence (last 12 balls):

[Ball 43.1: 1 run, Ball 43.2: dot, Ball 43.3: 4 runs, ...]

Transformer encodes:
- Recent scoring rate
- Dot ball pressure building
- Boundary frequency
- Bowler's recent execution

h_temporal = Transformer(ball_sequence)
```

### Relational Branch Input

```
Player graph at this moment:

Nodes:
├── Batsman A: {runs: 45, balls: 32, SR: 140, vs_pace: 0.65}
├── Batsman B: {runs: 8, balls: 6, SR: 133, vs_pace: 0.52}
├── Bowler X:  {overs: 3.3, runs: 28, wickets: 1, death_econ: 8.5}
└── Match:     {req_rate: 15.6, wickets_left: 6, pressure: high}

Edges:
├── A ↔ X: {balls_faced: 8, runs: 12, dismissals: 0}
├── A ↔ B: {partnership: 35 off 24}
└── A ↔ Match: {context link}

GAT computes attention:
- A-X matchup attention: α = 0.4 (key for this ball!)
- A-Match attention: α = 0.35 (pressure context)
- A-B attention: α = 0.15 (partnership)
- Other: α = 0.1

h_relational = GAT(player_graph)
```

### Combined Prediction

```
h_combined = MLP([h_temporal || h_relational])

Temporal says: "Recent dots → batsman might attack"
Relational says: "A has scored well vs X → can take risk"

Combined prediction:
P(0) = 0.15   (dot ball)
P(1) = 0.30   (single)
P(2) = 0.12   (two runs)
P(4) = 0.28   (boundary)  ← Elevated due to both signals
P(6) = 0.08   (six)
P(W) = 0.07   (wicket)
```

---

## Comparison: When Does GAT Help?

| Scenario | Pure Temporal | GAT Hybrid | Winner |
|----------|---------------|------------|--------|
| Routine middle overs | Good | Similar | Tie |
| Key matchup (Kohli vs Anderson) | Misses context | Captures | GAT |
| New batsman facing | Limited history | Career data in graph | GAT |
| Momentum shift | Captures well | Also captures | Tie |
| Unusual field placement | Can learn | Graph can encode | GAT |
| Simple dot ball prediction | Sufficient | Overhead not worth it | Temporal |

**GAT adds most value when**:
- Player-specific relationships matter
- Historical matchup data is predictive
- Context beyond recent balls is important

---

## Implementation Considerations

### Graph Construction

```python
def build_match_graph(match_state):
    """Build player interaction graph for current moment"""

    nodes = []
    edges = []

    # Add player nodes
    for batsman in match_state.current_batsmen:
        nodes.append(create_batsman_node(batsman, match_state))

    for bowler in match_state.active_bowlers:
        nodes.append(create_bowler_node(bowler, match_state))

    # Add match context node
    nodes.append(create_context_node(match_state))

    # Add edges
    for batsman in match_state.current_batsmen:
        for bowler in match_state.active_bowlers:
            edge = create_matchup_edge(batsman, bowler, match_state)
            edges.append(edge)

    # Add partnership edge
    edges.append(create_partnership_edge(
        match_state.current_batsmen[0],
        match_state.current_batsmen[1]
    ))

    return Graph(nodes, edges)
```

### Dynamic Updates

```python
def update_graph_after_ball(graph, ball_outcome):
    """Update graph after each ball"""

    # Update batsman node
    batsman_node = graph.get_node(ball_outcome.batsman)
    batsman_node.runs += ball_outcome.runs
    batsman_node.balls += 1

    # Update bowler node
    bowler_node = graph.get_node(ball_outcome.bowler)
    bowler_node.runs_conceded += ball_outcome.runs
    bowler_node.balls_bowled += 1

    # Update matchup edge
    edge = graph.get_edge(ball_outcome.batsman, ball_outcome.bowler)
    edge.update(ball_outcome)

    return graph
```

---

## Summary: GAT for Cricket

### Strengths

| Strength | Cricket Application |
|----------|---------------------|
| Learned importance | Which matchups matter in this situation |
| Interpretability | Explain predictions via attention |
| Inductive | Generalize to new players |
| Flexible structure | Model various relationships |

### Limitations

| Limitation | Cricket Impact |
|------------|----------------|
| Not naturally temporal | Need hybrid architecture |
| Requires graph design | Additional engineering |
| Static structure | Cricket relationships evolve |
| Overhead | May not improve simple predictions |

### Recommendation

**GAT is best used as a COMPONENT, not the whole model**:

```
Primary model: Temporal (Transformer/LSTM) for ball sequence
Enhancement:   GAT for player relationships
Fusion:        Combine for final prediction

This captures:
- Sequential patterns (temporal branch)
- Relational context (GAT branch)
- Interaction effects (fusion layer)
```

### When to Prioritize GAT for Cricket

1. **Matchup-heavy analysis**: Predicting specific batsman vs bowler outcomes
2. **Team composition effects**: How player combinations affect outcomes
3. **Interpretability required**: Need to explain predictions
4. **Limited sequential data**: New tournament, few recent balls

### When to Skip GAT

1. **Simple aggregate predictions**: Match winner, total runs
2. **Strong temporal signal**: Recent form dominates
3. **Minimal player interaction effects**: Data shows matchups don't matter much
4. **Computational constraints**: GAT adds overhead
