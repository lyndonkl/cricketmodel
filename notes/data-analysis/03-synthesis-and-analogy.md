# Cricket Ball Prediction: Synthesis & Analogy Analysis

## Part 1: Synthesizing Findings from Decomposition & Systems Analysis

### 1.1 Cross-Analysis Themes

**Theme 1: Cricket is BOTH Relational AND Temporal**

From decomposition analysis:
- Raw data is fundamentally sequential (ball-by-ball)
- But each ball involves relationships (batsman-bowler-match state)

From systems analysis:
- Feedback loops operate across time (momentum builds over balls)
- But leverage points are often relational (matchup, pressure interactions)

**Synthesis**: The proposed hybrid architecture (GAT for relationships + Temporal model for sequence) is well-justified. Neither alone captures both dimensions.

---

**Theme 2: Context Changes Everything**

From decomposition analysis:
- Same raw features have different meaning in different contexts
- Phase (powerplay/middle/death) changes interpretation of run rate, boundaries, etc.

From systems analysis:
- Feedback loop STRENGTH varies by context
- R1 (confidence spiral) is weaker for new batsmen
- B1 (required rate pressure) is only active in 2nd innings

**Synthesis**: Model architecture needs CONTEXT-AWARE attention. The GAT attention mechanism should learn to weight features differently based on context (phase, innings, pressure level). This argues for:
- Context embeddings as inputs to attention computation
- Phase-specific attention heads (or conditional attention)

---

**Theme 3: Derived Features Outweigh Raw Features**

From decomposition analysis:
- Tier 3 features (derived) often more predictive than Tier 1 raw features
- "Pressure index", "momentum vector", "setness" capture dynamics

From systems analysis:
- High-leverage features are mostly COMPUTED from raw data
- Consecutive dots, partnership balls, RRR gap - none are directly in JSON

**Synthesis**: Feature engineering is critical preprocessing. The raw JSON provides atoms; we must construct molecules (derived features) before feeding to model.

---

**Theme 4: Wickets Are the Structural Constraint**

From decomposition analysis:
- Wickets fundamentally change batting lineup (triggers new batsman)
- Score can always increase; wickets only increase (until 10)

From systems analysis:
- B2 (wicket conservation) is a BALANCING loop that limits aggression
- Wickets interact with every other stock (runs, balls, pressure)

**Synthesis**: Wickets should be modeled as a PRIMARY state variable. The model should have explicit "wickets in hand" representation that modulates attention weights. With 8 wickets, attend more to aggressive patterns. With 2 wickets, attend to survival patterns.

---

### 1.2 Conflict Resolution

**Conflict: Batsman Identity vs Situation**

- Decomposition says: Batsman ID is the #1 predictor (who is batting)
- Systems says: Situation (pressure, required rate) drives behavior

**Resolution via Meta-Framework**: Both are correct at different levels of abstraction.
- **Batsman ID** determines the CAPABILITY distribution (what the batsman CAN do)
- **Situation** determines the INTENT distribution (what the batsman TRIES to do)
- Final outcome = Capability × Intent

**Model Implication**: Use batsman embedding to represent capability. Use situation features to modulate capability into intent. Final prediction combines both.

---

**Conflict: Static vs Dynamic Features**

- Decomposition emphasizes: Per-match static features (venue, teams, toss)
- Systems emphasizes: Within-match dynamic features (momentum, pressure)

**Resolution via Scope Distinction**:
- Static features set the BASELINE distribution (venue par score, team strength)
- Dynamic features create DEVIATIONS from baseline (momentum shifts)

**Model Implication**: Encode static features as global context (background node in graph). Let dynamic features update ball-by-ball (sequence model captures this).

---

## Part 2: Analogical Mapping to Architecture

### 2.1 Analogy: Cricket Match as a Theater Production

**Source Domain**: Theater production
**Target Domain**: Cricket ball prediction

| Theater | Cricket | Why It Maps |
|---------|---------|-------------|
| Script | Match rules (overs, target) | Defines the constraints and goal |
| Director | Captain | Makes strategic decisions |
| Lead Actor | Current batsman | Primary focus of attention |
| Supporting Actor | Non-striker | Affects lead's performance |
| Antagonist | Bowler | Opposing force creating tension |
| Scene | Over | Unit of action |
| Act | Innings phase | Larger structural unit |
| Audience energy | Momentum | External force affecting performers |
| Dramatic tension | Pressure | Drives behavior changes |
| Climax | Death overs | Peak intensity |

**What Transfers**:
- **Attention on lead actor**: Just as audience attention is 70% on lead, model attention should be heavily on current batsman
- **Supporting cast matters at transitions**: Non-striker becomes crucial when lead changes (wicket)
- **Tension builds structurally**: Like a play has rising action, cricket has structured tension (powerplay → middle → death)
- **Antagonist-protagonist duel**: Batsman-bowler matchup is the SCENE-LEVEL conflict

**Limitations**:
- Theater is scripted; cricket is stochastic
- Theater audience doesn't affect outcome; cricket pressure does
- No "take two" in cricket; single realization

**Architectural Insight from Analogy**:
```
Graph structure should reflect dramatic structure:
- Protagonist node (batsman) = central attention
- Antagonist node (bowler) = attention weighted by "conflict intensity"
- Supporting node (non-striker) = attention weighted by "transition probability"
- Context node (match state) = "stage directions" that constrain action
```

---

### 2.2 Analogy: Cricket Ball Prediction as Chess Engine

**Source Domain**: Chess position evaluation
**Target Domain**: Cricket ball outcome prediction

| Chess | Cricket | Structural Mapping |
|-------|---------|-------------------|
| Board position | Match state | Current snapshot |
| Piece values | Player quality | Static assessment of strength |
| Positional advantage | Situational advantage | Dynamic assessment |
| King safety | Wickets in hand | Critical resource to protect |
| Time control | Balls remaining | Resource constraint |
| Initiative | Momentum | Psychological edge |
| Tactics | Matchup exploitation | Short-term optimization |
| Strategy | Phase planning | Long-term optimization |

**What Transfers**:
- **Two-level evaluation**: Chess engines evaluate positions at tactical (next few moves) AND strategic (long-term) levels. Cricket model should evaluate at ball-level AND innings-level.
- **Move probability from evaluation**: Engine converts position eval to move probabilities. Cricket converts state eval to outcome probabilities.
- **Lookahead matters**: Chess engines simulate future moves. Cricket model should implicitly represent "what happens if wicket falls here" (sequence modeling does this).

**Limitations**:
- Chess is perfect information; cricket has hidden state (bowler's intent)
- Chess is two-player deterministic; cricket has stochastic outcomes
- Chess positions are finite; cricket match states are continuous

**Architectural Insight from Analogy**:
```
Like chess engines use:
- Position encoding → Cricket: Match state encoding
- Piece-square tables → Cricket: Player-context embeddings
- Two-level search (tactical/strategic) → Cricket: Two-level hierarchy (GAT for ball, Transformer/HRM for sequence)

This supports the hybrid architecture!
```

---

### 2.3 Analogy: Match State as a Markov Decision Process (MDP)

**Source Domain**: Reinforcement learning MDP
**Target Domain**: Cricket ball prediction

| MDP Component | Cricket Equivalent |
|---------------|-------------------|
| State s | (score, wickets, balls, batsman, bowler, context) |
| Action a | Ball outcome (0,1,2,3,4,6,W,extras) |
| Transition P(s'|s,a) | How state changes after outcome |
| Reward R | Progress toward winning |
| Policy π(a|s) | What outcome distribution to predict |
| Value V(s) | Win probability from state |

**What Transfers**:
- **Markov property (partial)**: Next ball depends primarily on current state, not full history. But SOME history matters (momentum), so it's a POMDP.
- **Factored states**: State has COMPONENTS (batsman state, bowler state, match state). This maps to NODES in a graph.
- **Compositional value**: V(s) decomposes into component values that interact.

**Architectural Insight from Analogy**:
```
Graph = Factored state representation
- Batsman node = batsman component of state
- Bowler node = bowler component of state
- Match node = shared context component

GAT attention = Learned state-component interaction
- Which components matter most for predicting outcome

Temporal model = Capturing non-Markovian effects (momentum, patterns)
```

---

## Part 3: Architecture Mapping for Cricket Data

### 3.1 Data → Graph Structure Mapping

Based on synthesis, here's how Cricsheet data maps to GAT graph structure:

```
CRICSHEET JSON                      GAT GRAPH
─────────────────                   ─────────────────

info.players[team][]       ──→      Player Embeddings
                                    (pre-trained or learned)

innings[].overs[].deliveries[]
├── batter              ──→      BATSMAN NODE
│   └── Features:                  ├── ID embedding
│       - Current runs              ├── Innings: runs, balls
│       - Current balls             ├── Career stats (external)
│       - Career stats              └── Form (computed)
│
├── bowler              ──→      BOWLER NODE
│   └── Features:                  ├── ID embedding
│       - This spell overs          ├── Spell: overs, runs, wickets
│       - This spell runs           ├── Career stats (external)
│       - Career stats              └── Form (computed)
│
├── non_striker         ──→      PARTNER NODE
│   └── Features:                  ├── ID embedding
│       - Current runs              └── Partnership contribution
│       - Current balls
│
├── Match state         ──→      CONTEXT NODE
│   └── Derived:                   ├── Score, wickets
│       - Score                     ├── Over.ball progress
│       - Wickets                   ├── Pressure index
│       - Over.ball                 ├── Phase flags
│       - Target (2nd innings)      └── RRR (if chasing)
│
└── Sequence            ──→      SEQUENCE INPUT
    └── Last N deliveries          └── For Transformer/HRM
        - Outcomes
        - Ball features
```

### 3.2 Edge Structure

```
EDGES IN GAT GRAPH:

1. BATSMAN ←→ BOWLER (Matchup Edge)
   - Feature: Head-to-head history (SR, dismissals)
   - Attention: Should be HIGH when matchup is significant
   - Example: Kohli vs Anderson = high attention
   - Example: Tail-ender vs spinner = low attention (skill dominates)

2. BATSMAN ←→ CONTEXT (Pressure Edge)
   - Feature: Required rate, balls remaining
   - Attention: Should be HIGH in high-pressure situations
   - Example: 20 needed off 6 = very high attention
   - Example: 40 needed off 60, 8 wickets in hand = low attention

3. BATSMAN ←→ PARTNER (Partnership Edge)
   - Feature: Partnership balls, runs, strike rotation
   - Attention: Should be HIGH when partnership is established
   - Example: 50-run partnership = high attention
   - Example: New batsman just in = low attention

4. BOWLER ←→ CONTEXT (Tactical Edge)
   - Feature: Bowler's remaining overs, current economy
   - Attention: Should be HIGH in death overs
   - Example: Death specialist with last over = high attention
```

### 3.3 Temporal Integration

```
HYBRID ARCHITECTURE:

Per-Ball Input:
                    ┌─────────────┐
Ball state graph ──→│     GAT     │──→ h_relational (captures matchups,
(4 nodes, 4 edges)  │  (2 layers) │     partnerships, pressure)
                    └─────────────┘

Sequence Input:
                    ┌─────────────────┐
Last N balls    ──→│ Transformer/HRM │──→ h_temporal (captures momentum,
(N × d features)    │                 │     patterns, streaks)
                    └─────────────────┘

Combination:
                    ┌───────────────────┐
[h_relational,  ──→│   Fusion Layer    │──→ P(outcome | state)
 h_temporal]        │ (MLP or Cross-Attn)│
                    └───────────────────┘
```

---

## Part 4: HRM vs Transformer for Temporal Branch

### 4.1 Interpretability Analysis

**Question**: Does HRM provide interpretability advantages over Transformer?

#### HRM Interpretability

| Component | What Can Be Inspected | Interpretability Level |
|-----------|----------------------|------------------------|
| L-module outputs | High-level reasoning steps | MEDIUM - Can see what L produces, but meaning is opaque |
| H-module iterations | Low-level refinement | LOW - Many iterations, hard to trace single step |
| Halting decisions | When model stops refining | MEDIUM - Can see WHEN it stops, not WHY |
| z_L, z_H hidden states | Internal representations | LOW - High-dimensional, not human-readable |

**HRM Interpretability Summary**:
- Can observe: Number of iterations, halt timing, rough "effort" allocation
- Cannot observe: Why specific features were weighted, what relationships were learned
- Cricket insight: Could see "model thought harder about this ball" but not "model focused on matchup"

#### Transformer Interpretability

| Component | What Can Be Inspected | Interpretability Level |
|-----------|----------------------|------------------------|
| Self-attention weights | Which previous balls attend to which | HIGH - Direct visualization of ball-to-ball attention |
| Layer-wise attention | How attention patterns change by layer | MEDIUM - Can compare layers |
| Attention heads | Specialized attention patterns | HIGH - Different heads learn different patterns |
| Position-wise outputs | Per-position representations | LOW - High-dimensional |

**Transformer Interpretability Summary**:
- Can observe: "For predicting ball 47, model attended to balls 45, 42, 36"
- Can observe: "Head 1 focuses on recent balls, Head 4 focuses on same-bowler balls"
- Cricket insight: "Model attended to the last boundary" or "Model attended to bowler's previous over"

### 4.2 Interpretability Verdict

**Transformer provides better interpretability for cricket** because:

1. **Attention is meaningful for sequences**: Cricket is temporal, attention over time has natural interpretation
2. **Discrete attention targets**: Each ball is an interpretable unit (unlike HRM's continuous refinement)
3. **Head specialization**: Can discover "this head tracks momentum" vs "this head tracks bowler patterns"
4. **Direct visualization**: Can create "attention maps" showing ball-to-ball relationships

**HRM interpretability is structural, not content-based**:
- HRM tells you HOW MUCH the model thought (iterations)
- Transformer tells you WHAT the model attended to (specific balls)
- For cricket, the WHAT is more actionable

### 4.3 Architecture Recommendation

**Recommended: GAT + Transformer (not HRM)**

```
FINAL ARCHITECTURE:

┌──────────────────────────────────────────────────────────────┐
│                    BALL PREDICTION MODEL                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   INPUT LAYER                            │ │
│  │                                                          │ │
│  │  Static Context: venue_emb, team_emb, innings_flag       │ │
│  │  Current State: score, wickets, over.ball, pressure      │ │
│  │  Player States: batsman_emb+stats, bowler_emb+stats      │ │
│  │  Sequence: last_N_balls (outcome, batsman, bowler)       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                             │                                 │
│              ┌──────────────┴──────────────┐                 │
│              ▼                              ▼                 │
│  ┌─────────────────────┐      ┌─────────────────────────┐   │
│  │   RELATIONAL BRANCH │      │    TEMPORAL BRANCH       │   │
│  │                     │      │                          │   │
│  │  Ball State Graph   │      │  Ball Sequence           │   │
│  │  ├── Batsman Node   │      │  [b_{t-N}, ..., b_{t-1}] │   │
│  │  ├── Bowler Node    │      │                          │   │
│  │  ├── Partner Node   │      │  Transformer Encoder     │   │
│  │  └── Context Node   │      │  - Positional encoding   │   │
│  │                     │      │  - Multi-head attention  │   │
│  │  GAT (2 layers)     │      │  - 4-6 heads             │   │
│  │  - 4 attention heads│      │                          │   │
│  │  - Attend over edges│      │  Output: h_temporal      │   │
│  │                     │      │  (momentum, patterns)    │   │
│  │  Output: h_relational│     │                          │   │
│  │  (matchup, pressure) │     │                          │   │
│  └─────────────────────┘      └─────────────────────────┘   │
│              │                              │                 │
│              └──────────────┬───────────────┘                │
│                             ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    FUSION LAYER                          │ │
│  │                                                          │ │
│  │  Option A: Concatenate + MLP                             │ │
│  │  h_combined = MLP([h_relational || h_temporal])          │ │
│  │                                                          │ │
│  │  Option B: Cross-Attention                               │ │
│  │  h_combined = CrossAttn(h_relational, h_temporal)        │ │
│  │                                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                             │                                 │
│                             ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    OUTPUT LAYER                          │ │
│  │                                                          │ │
│  │  P(outcome) = softmax(W · h_combined + b)                │ │
│  │                                                          │ │
│  │  Outcomes: [0, 1, 2, 3, 4, 6, Wicket, Wide, NB, Bye]    │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 4.4 Interpretability Features in This Architecture

| Component | What's Interpretable | How To Use |
|-----------|---------------------|------------|
| GAT attention (batsman→bowler) | Matchup importance | "Model weighted matchup history at 0.4" |
| GAT attention (batsman→context) | Pressure sensitivity | "Model heavily attended to required rate" |
| Transformer attention | Temporal dependencies | "Model looked at balls 3, 7, 12 (same bowler's previous deliveries)" |
| Head specialization | Learned patterns | "Head 2 tracks boundaries, Head 5 tracks wickets" |
| Feature attributions | Which inputs mattered | Use SHAP/attention for feature importance |

---

## Part 5: Summary and Recommendations

### 5.1 Key Synthesis Points

1. **Cricket is inherently graph + sequence**: Use GAT for relational structure, Transformer for temporal patterns
2. **Context modulates everything**: Phase, innings, pressure should be explicit inputs that affect attention
3. **Derived features are essential**: Pressure index, momentum, partnership state > raw data
4. **Wickets are the structural constraint**: Model should treat wickets as primary state dimension
5. **Transformer > HRM for interpretability**: Attention over discrete balls is more inspectable than iterative refinement

### 5.2 Feature Priority (Final Synthesis)

| Priority | Feature Category | Examples |
|----------|-----------------|----------|
| P0 (Critical) | Player identity | batsman_id, bowler_id, partner_id |
| P0 (Critical) | Match progress | score, wickets, over.ball, innings |
| P0 (Critical) | Target state | target_runs, required_rate (2nd innings) |
| P1 (High) | Derived dynamics | pressure_index, momentum, consecutive_dots |
| P1 (High) | Player current state | batsman_runs, batsman_balls, bowler_spell |
| P2 (Medium) | Historical stats | career_sr, career_economy, matchup_history |
| P2 (Medium) | Context | venue, powerplay_flag, phase |
| P3 (Low) | Static metadata | team_type, event, season |

### 5.3 Data Pipeline Recommendation

```
RAW DATA                    PROCESSING                      MODEL INPUT
─────────                   ──────────                      ───────────

Cricsheet JSON              Feature Engineering             Graph + Sequence
                                  │
┌─────────────┐                   │
│ Match JSON  │──→ Parse ──────►  │
└─────────────┘                   │
       │                          │
       ▼                          ▼
┌─────────────┐            ┌─────────────┐
│  Deliveries │            │ Ball-level  │
│  (sequence) │──────────► │  features   │──────► Sequence tensor
└─────────────┘            └─────────────┘        (batch × seq_len × d)
       │                          │
       ▼                          ▼
┌─────────────┐            ┌─────────────┐
│ Player stats│            │ Node        │
│  (external) │──────────► │  features   │──────► Graph tensor
└─────────────┘            └─────────────┘        (batch × nodes × d)
       │                          │
       ▼                          ▼
┌─────────────┐            ┌─────────────┐
│  Derived    │            │  Edge       │
│  features   │◄────────── │  features   │──────► Adjacency + edge features
└─────────────┘            └─────────────┘
  (pressure,
   momentum,
   partnership)
```

### 5.4 Next: Derived Features Catalog

The next document catalogs all derived features that should be computed from raw Cricsheet data.
