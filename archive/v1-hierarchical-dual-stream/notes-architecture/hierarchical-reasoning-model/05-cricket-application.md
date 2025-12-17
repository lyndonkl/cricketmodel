# Cricket Application: HRM for Ball-by-Ball Prediction

## Why HRM Might Be Relevant for Cricket

Cricket ball-by-ball prediction has characteristics that align well with HRM's strengths:

1. **Variable Complexity**: Some balls are predictable (dot ball to a set batsman), others require deep reasoning (final over of a chase)
2. **Hierarchical Structure**: Strategy (innings-level) influences tactics (ball-level)
3. **Constraint Satisfaction**: Match state must be consistent (runs, wickets, over count)
4. **Long-Range Dependencies**: Current ball depends on events from many overs ago
5. **Limited Training Data**: Only ~100-500 matches per format per season

---

## Mapping HRM to Cricket

### The Conceptual Alignment

| HRM Concept | Cricket Analog |
|-------------|----------------|
| H-module (slow, strategic) | Match strategy, innings phase, target calculation |
| L-module (fast, tactical) | Ball-level execution, shot selection, field placement |
| Hierarchical convergence | Strategy guides tactics; tactics inform strategy updates |
| ACT (adaptive halting) | Think longer for crucial balls (final over), less for routine balls |
| One-step gradient | Learn from recent outcomes without full match replay |

### Concrete Example: A Chase Scenario

**Situation**: 15 runs needed from 12 balls, 2 wickets in hand

**Standard Transformer approach**:
- Fixed-depth processing of the current state
- Same computation whether it's ball 1 of the chase or the final ball
- Might "pattern match" to similar situations in training

**HRM approach**:

```
H-module state: "Chase scenario, tight finish, need 1.25 RPB"
                "Batsman A is set (45 balls), B is new (3 balls)"
                "Bowler is their death specialist"

L-module processing (given H-context):
  Cycle 1: Analyze field placement constraints
           → If bowler bowls yorker, options are: defend, jam for single, or risk
           → If bowler bowls short, options are: pull, duck, or leave

  Cycle 2: Evaluate risk/reward
           → Attacking shot: 30% boundary, 40% single, 20% dot, 10% wicket
           → Defensive: 60% dot, 35% single, 5% wicket

  Cycle 3: Consider match context
           → Losing a wicket now would be catastrophic
           → But 15 from 12 requires at least some boundaries

H-module update: "Prioritize keeping wicket, target 1 from this ball minimum"

Output: P(run outcome | state) with strategy-informed adjustments
```

---

## Data Efficiency: The Key Advantage

### The Cricket Data Problem

High-quality ball-by-ball data is limited:
- ~100-300 international matches per year per format
- ~300 balls per ODI match × 200 matches = ~60,000 balls/year
- But each ball is highly context-dependent!

For comparison:
- LLMs train on billions of tokens
- HRM achieved 40% on ARC-AGI with ~1000 examples

### Why HRM Might Help

**HRM's data efficiency comes from**:
1. **Latent reasoning**: Doesn't need CoT traces - learns from input-output pairs
2. **Direct supervision**: Learn from final outcomes, not intermediate steps
3. **Hierarchical inductive bias**: Built-in structure matches the problem

**For cricket**:
- You have ball-by-ball outcomes (not reasoning traces)
- Strategic structure is inherent (innings phases, target chasing)
- Limited examples could suffice with the right inductive bias

---

## Proposed Architecture for Cricket

### Input Representation

```
Input x for ball n:

Match state (global):
├── Innings: 1st or 2nd
├── Runs: current score
├── Wickets: current wickets lost
├── Overs: current over.ball (e.g., 45.3)
├── Target: (if 2nd innings)
└── Required rate: (if chasing)

Recent history (local):
├── Last 6 balls: [outcome, bowler_type, shot_type, ...]
├── Last over: run_rate, boundaries, dots
└── Current partnership: runs, balls

Batsman state:
├── Current runs, balls faced
├── Recent strike rate
└── Career stats vs this bowler type

Bowler state:
├── Current over: runs conceded
├── Spell figures
└── Career stats vs this batsman type

Match context:
├── Pitch conditions (estimated)
├── Weather
└── Venue historical stats
```

### Module Design

**H-module (Strategic)**:
```
Input: z_H^{k-1} (previous strategic state)
       z_L^{kT} (tactical assessment from L)

Output: z_H^k encoding:
- Innings phase (powerplay, middle, death)
- Risk tolerance (batting conservatively vs aggressively)
- Target adjustment (ahead/behind par)
- Key matchup to exploit/avoid
```

**L-module (Tactical)**:
```
Input: z_L^{i-1} (previous tactical state)
       z_H^k (current strategy from H)
       x̃ (encoded ball context)

Output: z_L^i encoding:
- Ball-level probabilities (dot, single, boundary, wicket)
- Shot type distribution
- Field placement influence
```

### Output Head

```
Output: ŷ = P(outcome | state, strategy)

Outcome space:
├── Runs: 0, 1, 2, 3, 4, 5, 6
├── Extras: wide, no-ball, bye, leg-bye
├── Wicket: bowled, caught, LBW, run-out, etc.
└── Combined: e.g., "1 + caught (attempting 2nd)"
```

---

## Why Adaptive Computation Matters for Cricket

### Routine Balls vs Crucial Balls

**Routine ball** (1st over of innings, settled batsman):
- Context is relatively stable
- Historical patterns dominate
- Little "reasoning" needed
- ACT should halt early: 1-2 segments

**Crucial ball** (final ball, 2 to win, last wicket):
- Every factor matters
- Need to consider: batsman's pressure handling, bowler's death bowling stats, field placement optimization, historical choke patterns
- Deep "reasoning" needed
- ACT should continue: 6-8 segments

### The Computational Budget Analogy

```
Standard model: Every ball gets 100ms of "thinking"

HRM with ACT:
- Ball 1.1 (opening ball): 20ms (routine)
- Ball 35.4 (middle overs, set batsman): 30ms
- Ball 49.5 (death over, need 8 from 2): 200ms
- Ball 49.6 (final ball, 3 to win): 500ms

Total compute is similar, but distributed according to importance!
```

---

## Hierarchical Convergence in Cricket Context

### How Strategy Guides Tactics

**Cycle 1** (H says: "Protect wicket phase"):
```
L-module reasoning:
- Weight defensive outcomes higher
- Reduce boundary attempt probability
- Emphasize singles
- Converges to: "Defensive approach, targeting 1-2 runs"
```

**Cycle 2** (H receives L's assessment, updates):
```
H-module: "L found defensive approach yields ~1.2 runs/ball"
         "Required rate is 1.5 runs/ball"
         "Must be slightly more aggressive"

New H-state: "Calculated aggression phase"
```

**Cycle 3** (H says: "Calculated aggression"):
```
L-module reasoning:
- Allow some boundary attempts
- But only on "hittable" deliveries
- Maintain wicket protection on good balls
- Converges to: "Selective aggression, targeting 1.5 runs"
```

### Why This Is Better Than Single-Pass

A standard model might:
- See "need 1.5 RPB" and predict aggressive play
- Ignore that "2 wickets in hand" demands caution
- Miss the balance required

HRM's hierarchy:
- H-module holds the balance (need runs BUT can't lose wicket)
- L-module explores options within that constraint
- Iteration finds the optimal trade-off

---

## Comparison: Cricket Prediction Approaches

| Approach | Strengths | Weaknesses | Data Needs |
|----------|-----------|------------|------------|
| Statistical model (GLM) | Interpretable, stable | No long-range deps | Low |
| Standard Transformer | Learns complex patterns | Fixed depth, data hungry | High |
| Transformer + CoT | Can "reason" | Slow, needs CoT data | Very High |
| LSTM/RNN | Sequential structure | Premature convergence | Medium |
| **HRM** | Adaptive depth, strategic | Unproven on cricket | Low |

### The HRM Advantage for Cricket

1. **Data efficiency**: ~1000 examples might suffice (HRM's demonstrated strength)
2. **Hierarchical structure**: Matches the strategy/tactics split in cricket
3. **Adaptive compute**: More thinking for crucial balls
4. **No CoT needed**: Learn directly from outcomes, not reasoning traces

---

## Potential Challenges

### 1. Sequence Length

HRM as presented uses fixed-length grids (30×30 for ARC/Maze, 9×9 for Sudoku).

Cricket matches have:
- ~300 balls per ODI (variable based on innings)
- Complex structure (overs, partnerships, innings)

**Solution**: Hierarchical representation
```
Instead of 300-position sequence:
- Current over (6 balls) as L-level input
- Over summaries as H-level context
- Match state as global conditioning
```

### 2. Continuous Outcomes

HRM was tested on discrete tasks (Sudoku cells, maze paths).

Cricket has:
- Discrete: run outcomes (0,1,2,3,4,6)
- Continuous: win probability, expected runs
- Mixed: discrete events with continuous probabilities

**Solution**: Mixed output heads
```
Discrete head: P(runs = 0, 1, 2, 3, 4, 6, wicket)
Continuous head: E[runs], Var[runs]
```

### 3. Partial Observability

Unlike Sudoku (fully visible grid), cricket has:
- Unknown bowler intentions
- Hidden pitch deterioration
- Unobserved fatigue levels

**Solution**: Learned embeddings for latent factors
```
H-module could maintain beliefs about:
- Pitch behavior (estimated from observations)
- Bowler fatigue (inferred from recent performance)
- Batsman confidence (inferred from shot selection)
```

---

## Experimental Design Suggestion

### Phase 1: Validate on Simplified Cricket

Create a "Cricket-Lite" task:
- Single over (6 balls)
- Fixed batsman/bowler
- Predict run sequence
- ~1000 training examples

**Success criterion**: Outperform Transformer baseline on same data

### Phase 2: Extend to Match-Level

- Full ODI/T20 matches
- Multiple batsmen/bowlers
- Predict ball-by-ball outcomes
- ~500-1000 match histories

**Success criterion**: Competitive with larger Transformer on same data

### Phase 3: Strategic Tasks

Test on cricket "reasoning" tasks:
- Optimal batting order given situation
- Field placement selection
- Bowling change decisions

These are more like ARC/Sudoku - discrete choices requiring multi-step reasoning.

---

## Summary: HRM for Cricket

| HRM Property | Cricket Relevance |
|--------------|-------------------|
| Hierarchical modules | Strategy (H) → Tactics (L) maps naturally |
| Adaptive computation | Crucial balls need more "thinking" |
| Data efficiency | Limited match data availability |
| Latent reasoning | Don't need CoT annotations |
| Dimensionality hierarchy | Strategic flexibility vs tactical execution |

**The key insight**: Cricket prediction isn't just pattern matching - it requires something like "reasoning" about strategy, constraints, and trade-offs. HRM's architecture is designed exactly for this kind of task.

**Open question**: Can HRM's success on discrete reasoning (Sudoku, ARC) transfer to the partially observable, stochastic domain of cricket? The architectural fit suggests it's worth exploring.
