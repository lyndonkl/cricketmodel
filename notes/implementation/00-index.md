# Cricket Ball Prediction: Implementation Guide

## Purpose

This folder synthesizes findings from [data analysis](../data-analysis/00-index.md) and [architecture studies](../architecture/) into a concrete implementation plan. The architecture is designed for **LLM-consumable interpretability**: attention patterns that an LLM can observe and translate into real-time match insights.

## Design Philosophy

The model must produce attention patterns that answer questions like:
- "Why did the model predict a boundary here?"
- "What factors suggest a wicket is likely?"
- "How is the chase pressure affecting predictions?"

To achieve this, we need:
1. **Semantically meaningful graph nodes** - Not just "batsman node" but separate nodes for identity, state, and momentum
2. **Hierarchical attention** - Global context → match state → actors → dynamics
3. **Temporal attention** - Which past balls are relevant to this prediction

## Document Structure

| Document | Purpose |
|----------|---------|
| [01-available-data.md](./01-available-data.md) | What data we can use from Cricsheet and its limitations |
| [02-graph-structure.md](./02-graph-structure.md) | Detailed node design for interpretable within-ball attention |
| [03-hierarchical-attention.md](./03-hierarchical-attention.md) | Multi-level attention from global to ball-level |
| [04-temporal-attention.md](./04-temporal-attention.md) | Cross-ball attention design |
| [05-derived-vs-learned.md](./05-derived-vs-learned.md) | When to compute features explicitly vs let attention learn |
| [06-llm-interpretability.md](./06-llm-interpretability.md) | How an LLM consumes attention patterns for insights |
| [07-live-data-contract.md](./07-live-data-contract.md) | Input/output specification for live predictions |
| [08-cold-start-embeddings.md](./08-cold-start-embeddings.md) | Handling unseen players with embedding generation |

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL ATTENTION ARCHITECTURE                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     GLOBAL CONTEXT LAYER                             │  │
│  │  ┌─────────┐  ┌─────────────┐  ┌────────────────┐                   │  │
│  │  │  Venue  │  │    Team     │  │ Match Context  │                   │  │
│  │  │  Node   │  │   Node      │  │     Node       │                   │  │
│  │  └────┬────┘  └──────┬──────┘  └───────┬────────┘                   │  │
│  └───────┼──────────────┼─────────────────┼────────────────────────────┘  │
│          └──────────────┼─────────────────┘                               │
│                         ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     MATCH STATE LAYER                                │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │ Score   │  │ Chase   │  │ Phase   │  │ Time    │  │ Wicket  │   │  │
│  │  │ State   │  │ State   │  │ State   │  │Pressure │  │ Buffer  │   │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │  │
│  └───────┼────────────┼────────────┼────────────┼────────────┼────────┘  │
│          └────────────┴────────────┼────────────┴────────────┘            │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     ACTOR LAYER                                      │  │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │  │
│  │  │ Batsman   │    │ Batsman   │    │  Bowler   │    │  Bowler   │  │  │
│  │  │ Identity  │◄──►│  State    │◄──►│ Identity  │◄──►│  State    │  │  │
│  │  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘  │  │
│  │        │                │                │                │         │  │
│  │        │         ┌──────┴──────┐         │                │         │  │
│  │        │         │ Partnership │         │                │         │  │
│  │        └────────►│    Node     │◄────────┘                │         │  │
│  │                  └─────────────┘                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     DYNAMICS LAYER                                   │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │  │
│  │  │ Batting   │  │ Bowling   │  │ Pressure  │  │  Dot Ball │        │  │
│  │  │ Momentum  │  │ Momentum  │  │  Index    │  │  Pressure │        │  │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     TEMPORAL LAYER                                   │  │
│  │                                                                      │  │
│  │  Ball(t-24) ─► Ball(t-23) ─► ... ─► Ball(t-2) ─► Ball(t-1) ─► ?    │  │
│  │       │            │                    │            │               │  │
│  │       └────────────┴────────────────────┴────────────┘               │  │
│  │                    Transformer Attention                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     OUTPUT LAYER                                     │  │
│  │                                                                      │  │
│  │              P(outcome) = [0, 1, 2, 3, 4, 6, W, wd, nb, bye]        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Within-ball structure** | 17-node hierarchical graph | Each node semantically meaningful for LLM interpretation |
| **Cross-ball structure** | Transformer attention | Shows which past balls matter; interpretable heads |
| **Global context** | Persistent nodes attended by each ball | Venue, team, match importance persist and influence |
| **Derived features** | Compute explicitly AND let attention weight | Hybrid approach - see [05-derived-vs-learned.md](./05-derived-vs-learned.md) |
| **HRM vs Transformer** | Transformer | Content-level interpretability (WHAT) vs HRM's effort-level (HOW MUCH) |

## For LLM Insight Generation

The attention weights will be structured so an LLM can generate insights like:

**Example attention pattern**:
```json
{
  "ball": 47,
  "prediction": "boundary (4)",
  "confidence": 0.42,
  "within_ball_attention": {
    "chase_state": 0.28,
    "pressure_index": 0.22,
    "batsman_momentum": 0.18,
    "batsman_identity": 0.15,
    "phase_state": 0.12,
    "other": 0.05
  },
  "temporal_attention": {
    "same_bowler_balls": [43, 37, 31],
    "recent_balls": [46, 45, 44],
    "attention_weights": [0.15, 0.12, 0.10, 0.08, 0.07, 0.06]
  }
}
```

**LLM-generated insight**:
> "The model predicts a boundary with 42% confidence, heavily weighted by the chase equation (28%) and current pressure (22%). The batsman has positive momentum (18%) and the model looked at this bowler's previous deliveries (balls 43, 37, 31) where 2 boundaries were scored. The death overs phase (12%) typically sees more aggressive batting."

## Related Documentation

- [Data Analysis Notes](../data-analysis/00-index.md) - Feature engineering and systems analysis
- [GAT Architecture](../architecture/graph-attention-networks/00-index.md) - Graph attention mechanism
- [Transformer Architecture](../architecture/attention-is-all-you-need/00-index.md) - Temporal attention
- [HRM Analysis](../architecture/hierarchical-reasoning-model/00-index.md) - Why we chose Transformer over HRM
