# Cricket Data Analysis: Feature Engineering Notes

## Overview

This folder contains comprehensive analysis of Cricsheet T20 data for ball-by-ball prediction modeling. The analysis uses **five complementary skill frameworks** to ensure thorough coverage:

1. **Decomposition & Reconstruction** - Breaking down data components and their relationships
2. **Systems Thinking & Leverage** - Identifying feedback loops and high-impact features
3. **Synthesis & Analogy** - Mapping data to architecture and explaining via analogies
4. **Negative Contrastive Framing** - Defining good features by showing what they're NOT
5. **Layered Reasoning** - Structuring the problem at strategic/tactical/operational levels

## Data Source

**Cricsheet JSON Format** (v1.0.0, v1.1.0)
- Documentation: https://cricsheet.org/format/json/
- Dataset: 3,046 male T20 matches
- Format: Ball-by-ball delivery data with player identifiers

## Notes Structure

| File | Skill Framework | Key Insights |
|------|-----------------|--------------|
| [01-decomposition-reconstruction.md](./01-decomposition-reconstruction.md) | Decomposition & Reconstruction | Batsman ID is #1 predictor; 3-tier feature priority system |
| [02-systems-thinking-leverage.md](./02-systems-thinking-leverage.md) | Systems Thinking & Leverage | 4 reinforcing loops, 4 balancing loops; pressure index as high-leverage feature |
| [03-synthesis-and-analogy.md](./03-synthesis-and-analogy.md) | Synthesis & Analogy | Cricket as theater/chess/MDP; Transformer > HRM for interpretability |
| [04-derived-features-catalog.md](./04-derived-features-catalog.md) | (Consolidated Output) | 110+ derived features with code; organized by P0-P3 priority |
| [05-negative-contrastive-framing.md](./05-negative-contrastive-framing.md) | Negative Contrastive Framing | 5 anti-goals, 8 near-miss features, 5 failure patterns |
| [06-layered-reasoning.md](./06-layered-reasoning.md) | Layered Reasoning | 30K/3K/300ft architecture with consistency checks and implementation code |

---

## Key Conclusions

### 1. Cricket is Graph + Sequence

Cricket has BOTH relational structure (batsman-bowler matchup, partnership) AND temporal structure (ball-by-ball sequence, momentum). Neither alone captures both dimensions.

**Architecture Recommendation**: GAT for relational + Transformer for temporal

### 2. Derived Features Outweigh Raw Features

Raw JSON fields are necessary but insufficient. High-impact features like:
- Pressure index (composite)
- Momentum score (rolling window)
- Batsman setness (confidence)
- Required run rate gap (chase pressure)

...must be computed from raw data.

### 3. Context Modulates Everything

The same feature means different things in different contexts:
- Dot ball in powerplay: Bad for batting team
- Dot ball in death overs: Very bad
- Dot ball at 190/2 off 18: Acceptable

Model needs context-aware attention.

### 4. Transformer > HRM for Interpretability

**HRM** provides: Number of iterations (how hard model thought)
**Transformer** provides: Which balls attended to (what model focused on)

For cricket, knowing WHAT (specific balls, patterns) is more actionable than HOW MUCH.

### 5. Feature Priority Tiers

| Tier | Examples | Count |
|------|----------|-------|
| P0 (Critical) | Batsman ID, bowler ID, score, wickets, over.ball, target | 14 |
| P1 (High) | Pressure index, momentum, setness, partnership, consecutive dots | 47 |
| P2 (Medium) | Career stats, head-to-head, venue effects | 45 |
| P3 (Low) | Event metadata, officials | 4 |

### 6. Feature Quality Criteria (from Negative Contrastive)

Good features must be:
- ✅ Available before ball is bowled (no leakage)
- ✅ Variable across situations (not constant)
- ✅ Causally or correlationally linked to outcome
- ✅ Non-redundant (adds information)
- ✅ Generalizable (uses embeddings, not memorization)
- ✅ Properly normalized and contextualized

### 7. Strategic Principles (from Layered Reasoning)

| Principle | Constraint |
|-----------|------------|
| P1: Temporal Integrity | All features computed from t-1 and earlier |
| P2: Interpretability | Architecture supports attention visualization |
| P3: Generalization | Use embeddings, not one-hot for players |
| P4: Domain Fidelity | Architecture reflects cricket's relational + temporal nature |
| P5: Realistic Inputs | Only use data available before ball |

---

## Proposed Architecture

```
┌────────────────────────────────────────────────────────┐
│                 BALL PREDICTION MODEL                   │
│                                                         │
│  ┌─────────────────┐       ┌──────────────────────┐   │
│  │ RELATIONAL      │       │ TEMPORAL             │   │
│  │ BRANCH          │       │ BRANCH               │   │
│  │                 │       │                      │   │
│  │ Ball State Graph│       │ Last N Balls         │   │
│  │ - Batsman node  │       │ - Sequence encoding  │   │
│  │ - Bowler node   │       │                      │   │
│  │ - Partner node  │       │ Transformer Encoder  │   │
│  │ - Context node  │       │ - Multi-head attn    │   │
│  │                 │       │ - Positional enc     │   │
│  │ GAT (2 layers)  │       │                      │   │
│  │ - 4 attn heads  │       │ h_temporal           │   │
│  │                 │       │                      │   │
│  │ h_relational    │       │                      │   │
│  └────────┬────────┘       └──────────┬───────────┘   │
│           │                           │               │
│           └───────────┬───────────────┘               │
│                       ▼                               │
│              ┌────────────────┐                       │
│              │ FUSION LAYER   │                       │
│              │ MLP or X-Attn  │                       │
│              └────────┬───────┘                       │
│                       ▼                               │
│              ┌────────────────┐                       │
│              │ OUTPUT         │                       │
│              │ P(outcome)     │                       │
│              │ [0,1,2,3,4,6,  │                       │
│              │  W,wd,nb,bye]  │                       │
│              └────────────────┘                       │
└────────────────────────────────────────────────────────┘
```

---

## Skill Framework Summary

### Decomposition & Reconstruction (01)
- Broke down Cricsheet data into 3 levels: match, innings, delivery
- Identified ~40 raw features across categories
- Created bottleneck analysis: Batsman ID → Bowler ID → Phase → Pressure → Matchup
- Recommended graph node structure

### Systems Thinking & Leverage (02)
- Identified 4 reinforcing loops: Confidence spiral, Collapse trigger, Momentum cascade, Bowling dominance
- Identified 4 balancing loops: Required rate pressure, Wicket conservation, Bowler quota, Powerplay expiry
- Mapped to Meadows' 12 leverage points: High-leverage = feedback loops (level 6-7), Low-leverage = parameters (level 12)
- Created pressure index formula capturing multiple feedback loops

### Synthesis & Analogy (03)
- Synthesized decomposition + systems findings into unified architecture
- Used analogies: Cricket as theater (attention structure), Chess engine (two-level evaluation), MDP (factored state)
- Resolved conflicts: Batsman ID (capability) vs Situation (intent) - both matter at different levels
- Made HRM vs Transformer decision: Transformer for content-level interpretability

### Negative Contrastive Framing (05)
- Defined 5 anti-goals: Future leakage, no variance, proxy for target, no generalization, spurious correlation
- Identified 8 near-miss features that seem good but fail (raw counts, career stats without context, single-dimension pressure)
- Created 5 failure pattern taxonomies: Temporal leakage, Identity memorization, Context collapse, Aggregation artifacts, Survivorship bias
- Built 7-point feature inclusion checklist

### Layered Reasoning (06)
- 30K (Strategic): Core mission, 5 invariant principles, success criteria
- 3K (Tactical): Architecture decisions (GAT + Transformer), graph design (4 nodes), sequence design (24 balls)
- 300 ft (Operational): Python implementation code for data pipeline, feature extraction, model architecture
- Consistency checks: Upward, downward, and lateral consistency validated

---

## Next Steps

1. **Data Pipeline**: Build Cricsheet JSON parser that computes all P0 and P1 features
2. **External Data**: Source career statistics for head-to-head and historical features
3. **Model Implementation**: Implement GAT + Transformer hybrid architecture (code in 06-layered-reasoning.md)
4. **Baseline**: Compare against simpler models (LSTM-only, MLP-only) to validate hybrid value
5. **Interpretability**: Build attention visualization tools for both GAT and Transformer components
6. **Validation**: Verify against negative-contrastive checklist before training
