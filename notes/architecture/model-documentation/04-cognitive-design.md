# Cricket Prediction Model: Cognitive Design Guide

## Purpose

This guide applies cognitive design principles to help stakeholders understand, interpret, and trust the cricket prediction model. It provides visualization recommendations, mental models, and interpretability strategies aligned with how humans naturally process information.

---

## Quick Validation (3-Question Check)

### 1. Attention: Is it obvious what to look at first?

**For model interpretation:**
- Layer importance should be immediately visible (which layer matters most?)
- Outcome probabilities should have clear visual hierarchy
- Attention weights should highlight "hot" vs "cold" nodes

### 2. Memory: Is the user required to remember anything that could be shown?

**For model interpretation:**
- Show current state alongside predictions (don't make users remember score)
- Display attention weights visually, not just numerically
- Provide contextual baselines ("35% dot ball vs 40% average")

### 3. Clarity: Can someone unfamiliar understand in 5 seconds?

**For model interpretation:**
- Lead with the prediction, then show evidence
- Use cricket terminology users already know
- Avoid ML jargon in stakeholder interfaces

---

## Cognitive Design Pyramid Applied to Model Interpretation

### Layer 1: Perceptual Efficiency

**Goal:** Make key information instantly visible

**Recommendations:**

1. **Outcome Probability Bar Chart**
   ```
   Dot    ████████████████████  35%
   Single ██████████████        28%
   Four   ██████                12%
   Two    █████                 10%
   Six    ███                    5%
   Wicket ███                    5%
   Three  ██                     5%
   ```
   - Horizontal bars sorted by probability
   - Color coding: Green (run-scoring) vs Red (wicket) vs Gray (dot)
   - Immediate pattern recognition: "Dots most likely"

2. **Layer Importance Donut**
   ```
        Global (22%)
           ┌───┐
          /     \
    Actor │     │ State
    (30%) │     │ (28%)
          \     /
           └───┘
       Dynamics (20%)
   ```
   - Spatial grouping shows relative importance
   - Largest slice = current focus

3. **Actor Graph Visualization**
   ```
   [Striker] ══════ [Bowler]
       │     0.35      │
       │               │
   [Striker   [Bowler
    State]     State]
       \       /
        \     /
      [Partnership]
   ```
   - Edge thickness = attention weight
   - Highlight matchup edge when important
   - Color nodes by "positive/negative" influence

### Layer 2: Cognitive Coherence

**Goal:** Make relationships and causality clear

**Recommendations:**

1. **Hierarchical Information Flow**
   ```
   GLOBAL CONTEXT
   ├── Venue: MCG (high-scoring)
   ├── Batting Team: India (aggressive)
   └── Bowling Team: Australia (pace attack)
           │
           ▼ conditions
   MATCH STATE
   ├── Score: 52/2 chasing 156
   ├── Phase: Middle overs
   └── Pressure: Moderate (0.45)
           │
           ▼ influences
   ACTOR MATCHUP
   ├── Rohit: Set (24 off 18)
   ├── Bumrah: Threatening (1-12)
   └── Partnership: Stable (28 off 22)
           │
           ▼ drives
   DYNAMICS
   ├── Momentum: Slight batting (+0.3)
   └── Dot pressure: Building (2 dots)
           │
           ▼
   PREDICTION: 35% dot, 28% single, 12% four
   ```
   - Top-down flow matches mental model
   - "Conditions → influences → drives → prediction"
   - Chunking: 4 layers, 3-5 items each (working memory limit)

2. **Temporal History Timeline**
   ```
   Ball:  -6   -5   -4   -3   -2   -1   NOW
          │    │    │    │    │    │
          1    0    0    4    0    0    ?
          ▪    ▪    ▪    █    ▪    ▪
                              ↑
                         Recent dots
                         building pressure
   ```
   - Left-to-right temporal ordering (natural)
   - Highlight patterns model is attending to
   - Annotate significant events (boundary at -3)

3. **Causal Reasoning Display**
   ```
   WHY 35% DOT?
   ┌────────────────────────────────────┐
   │ ✓ Bumrah's economy: 6.0 (excellent)│ → +15%
   │ ✓ 2 consecutive dots              │ → +8%
   │ ✓ Bumrah bowled 3 dots to Rohit   │ → +5%
   │ ✗ Rohit is set (SR 133)           │ → -12%
   │ ✗ Venue high-scoring              │ → -6%
   └────────────────────────────────────┘
   Net: Base 25% + 15% + 8% + 5% - 12% - 6% ≈ 35%
   ```
   - Shows factors FOR and AGAINST
   - Approximate attribution (not exact, but interpretable)
   - Users can follow the reasoning

### Layer 3: Emotional Engagement

**Goal:** Connect technical outputs to cricket drama

**Recommendations:**

1. **Tension Meter**
   ```
   Match Tension
   ░░░░░░░░████████░░░░  (Moderate)
   Low                High

   Sources:
   ├── Chase pressure: ███░░ (RRR 8.3)
   ├── Matchup:        ████░ (Rohit vs Bumrah)
   └── Dot pressure:   ██░░░ (2 consecutive)
   ```
   - Aggregates pressure signals into one emotional read
   - Maps to commentary language ("pressure building")

2. **Narrative Prompts**
   - Display alongside predictions:
   - "The equation demands 8.3 an over..."
   - "Bumrah senses the opportunity..."
   - "Rohit looking to break free..."
   - Generated from attention patterns

3. **Historical Comparison**
   ```
   Similar situations:
   ├── 2023 WC Final: Kohli 54*, faced Cummins → 4
   ├── 2022 T20 WC: Pandya vs Stokes → Wicket
   └── Model accuracy in similar: 67%
   ```
   - Grounds predictions in memorable examples
   - Builds trust through track record

### Layer 4: Behavioral Alignment

**Goal:** Guide users to appropriate actions and interpretations

**Recommendations:**

1. **Confidence Calibration**
   ```
   Model Confidence: MODERATE

   Top outcome: Dot (35%)
   Gap to second: 7% (Single at 28%)

   Interpretation:
   "Model slightly favors dot, but single nearly as likely.
    Don't bet the house on this one."
   ```
   - Translate probabilities to decision guidance
   - Warn when predictions are uncertain

2. **Alert Thresholds**
   ```
   ⚠️ WICKET ALERT
   Wicket probability: 12% (normally 5%)

   Triggers:
   ├── Consecutive dots > 4
   ├── New batter < 5 balls
   └── Pressure index > 0.7
   ```
   - Proactive alerts for significant events
   - Explain why the alert triggered

3. **Decision Support for Commentary**
   ```
   SUGGESTED TALKING POINTS:
   1. The Rohit-Bumrah duel (high matchup attention)
   2. Two dots building pressure (dynamics focus)
   3. Set batter vs. in-form bowler tension

   AVOID:
   - Venue discussion (low attention weight)
   - Chase equation (moderate, not dominant)
   ```
   - Direct guidance for LLM or human commentators
   - Based on attention weights

---

## Visualization Specifications

### 1. Outcome Probability Display

**Chart Type:** Horizontal bar chart
**Sorting:** Descending by probability
**Colors:**
- Dot: Gray (#888)
- Singles/Twos/Threes: Green gradient (#4CAF50 to #8BC34A)
- Four/Six: Gold (#FFD700)
- Wicket: Red (#F44336)

**Annotations:**
- Baseline mark for each outcome (historical average)
- Confidence interval where available

### 2. Attention Heat Map

**For temporal attention:**
```
          Ball -24  ...  Ball -6  ...  Ball -1
Recency      ░░░         ▒▒▒         ███
Same-Bowler  ░░░         ███         ███
Same-Batter  ▒▒▒         ░░░         ███
Learned      ░▒▒         ▒▒▒         ▒▒▒
```
**Colors:** White (0.0) → Yellow (0.5) → Red (1.0)
**Purpose:** Show what history the model is "looking at"

### 3. Layer Importance Over Time

**Chart Type:** Stacked area chart across match
```
100% ┬────────────────────────────────────
     │ Dynamics
 75% │ ▄▄▄▄▄▄████████▄▄▄▄
     │ Actor
 50% │ ████████▄▄▄▄▄▄████
     │ State
 25% │ ▄▄▄▄▄▄▄▄▄▄▄▄████████
     │ Global
  0% └────────────────────────────────────
        Ball 1    Ball 60   Ball 120
```
**Purpose:** Show how model focus shifts through match phases

### 4. Actor Graph Network

**Layout:** Fixed positions (striker left, bowler right, partnership bottom)
**Edge encoding:**
- Width = attention weight
- Color = positive (green) vs negative (red) influence
**Node encoding:**
- Size = importance in current prediction
- Border = identity vs state (solid vs dashed)

---

## Mental Models for Stakeholders

### For Cricket Experts

"The model watches cricket like a seasoned analyst:
1. First it checks the venue and teams—the 'setting'
2. Then reads the scorecard and match situation
3. Focuses on who's batting and bowling right now
4. Finally feels the momentum and pressure

The attention weights tell you what the model thinks matters most *right now*."

### For ML Engineers

"It's a dual-stream architecture:
1. HierarchicalGAT: 17-node graph with 4-layer top-down conditioning
2. TemporalTransformer: 24-ball sequence with specialized attention heads
3. Late fusion: Concatenate + MLP

The attention weights are fully extractable for interpretability."

### For Product Managers

"We can explain every prediction:
- Show which factors matter most (layer importance)
- Show which historical balls influenced it (temporal attention)
- Show the batter-bowler matchup dynamics (actor attention)

This enables trust-building features and LLM commentary generation."

---

## Cognitive Pitfalls to Avoid

### 1. Information Overload
**Bad:** Showing all 17 node embeddings simultaneously
**Good:** Show layer-level summaries, drill down on demand

### 2. False Precision
**Bad:** "34.7% probability of dot ball"
**Good:** "About 1 in 3 chance of dot ball"

### 3. Attention = Importance Confusion
**Bad:** "High attention on bowler means bowler is performing well"
**Good:** "High attention on bowler means the bowler is a key factor in this prediction (good or bad)"

### 4. Outcome Probability Misinterpretation
**Bad:** User expects highest probability outcome to always happen
**Good:** Frame as "most likely, but remember: 35% means it WON'T happen 65% of the time"

---

## Summary

Applying cognitive design to the cricket prediction model means:

1. **Perceptual efficiency:** Clear visual hierarchy of probabilities and attention
2. **Cognitive coherence:** Top-down hierarchy matches cricket reasoning
3. **Emotional engagement:** Connect technical signals to match drama
4. **Behavioral alignment:** Guide interpretation, warn about uncertainty

The goal is to make the model's reasoning as transparent as a TV commentator explaining their thought process.
