# Cricket Prediction Model: Decomposition & Reconstruction Analysis

## System Definition

**System:** Cricket ball outcome prediction model
**Goal:** Predict the outcome of the next ball (7 classes: dot, 1, 2, 3, 4, 6, wicket) given match context
**Boundaries:** Single ball prediction only; does not simulate full match sequences
**Success Criteria:** Accurate probability distribution over outcomes; interpretable attention weights for commentary

---

## Component Decomposition

### Level 1: Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CricketPredictor                               │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Embedding  │    │ Hierarchical │    │   Temporal   │          │
│  │   Manager    │───►│     GAT      │    │ Transformer  │          │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘          │
│                             │                   │                   │
│                             └─────────┬─────────┘                   │
│                                       ▼                             │
│                              ┌──────────────┐                       │
│                              │    Fusion    │                       │
│                              │    Layer     │                       │
│                              └──────┬───────┘                       │
│                                     ▼                               │
│                              ┌──────────────┐                       │
│                              │   Output     │                       │
│                              │    Head      │                       │
│                              └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Components:**
1. **EmbeddingManager** - Converts entity IDs to dense vectors
2. **HierarchicalGAT** - Processes current ball context with layered attention
3. **TemporalTransformer** - Processes ball history with specialized attention heads
4. **Fusion Layer** - Combines GAT and Transformer outputs
5. **Output Head** - Projects to 7-class probability distribution

---

### Level 2: Embedding Manager Decomposition

```
EmbeddingManager
├── PlayerEmbeddingTable (num_players, 64-dim)
│   ├── Learned nn.Embedding
│   └── PlayerEmbedding (feature-based generator)
│       ├── Batsman encoder: [SR, avg, position, roles, exp] → 64d
│       └── Bowler encoder: [econ, SR, pace, role, exp] → 64d
│
├── VenueEmbeddingTable (num_venues, 32-dim)
│   ├── Learned nn.Embedding
│   └── VenueEmbedding (feature-based generator)
│       └── Encoder: [avg_scores, boundary%, wickets, pace_share] → 32d
│
└── TeamEmbedding (num_teams, 32-dim)
    └── Learned nn.Embedding only
```

**Design Choice Analysis:**

| Component | Learned | Feature-Based | Why |
|-----------|---------|---------------|-----|
| Players | ✓ | ✓ (optional) | Handles cold-start; ~thousands of players |
| Venues | ✓ | ✓ (optional) | Regional patterns; fewer venues than players |
| Teams | ✓ | ✗ | Teams are always known; few teams |

**Data Flow:**
```
player_idx: int → Embedding → 64d vector
venue_idx: int → Embedding → 32d vector
team_idx: int → Embedding → 32d vector
```

---

### Level 3: Hierarchical GAT Decomposition

```
HierarchicalGAT (17 nodes, 4 layers)
│
├── Layer 1: GlobalContextAttention
│   ├── Inputs: [venue, batting_team, bowling_team] (3 nodes)
│   ├── Mechanism: MultiheadAttention with learned query
│   └── Output: h_global (128d)
│
├── Layer 2: MatchStateAttention (conditioned on h_global)
│   ├── Inputs: [score, chase, phase, time_pressure, wicket_buffer] (5 nodes)
│   ├── Mechanism: Self-attention + Cross-attention to h_global
│   └── Output: h_state (128d)
│
├── Layer 3: ActorLayerGAT (conditioned on h_state)
│   ├── Inputs: [striker_id, striker_state, bowler_id, bowler_state, partnership] (5 nodes)
│   ├── Mechanism: GATv2Conv with fixed edge structure
│   │   ├── striker_id ↔ striker_state
│   │   ├── bowler_id ↔ bowler_state
│   │   ├── striker ↔ bowler (matchup edge)
│   │   └── states ↔ partnership
│   └── Output: h_actor (128d)
│
├── Layer 4: DynamicsAttention (conditioned on h_actor)
│   ├── Inputs: [batting_momentum, bowling_momentum, pressure, dot_pressure] (4 nodes)
│   ├── Mechanism: Self-attention
│   └── Output: h_dynamics (128d)
│
└── Fusion
    ├── Weighted sum: softmax(learned_weights) · [h_global, h_state, h_actor, h_dynamics]
    ├── Concatenation: [h_global || h_state || h_actor || h_dynamics] → 512d
    └── MLP: 512d → 256d → 128d
```

**17 Node Feature Sources:**

| Layer | Node | Input Dim | Source |
|-------|------|-----------|--------|
| Global | venue | 32 | VenueEmbedding |
| Global | batting_team | 32 | TeamEmbedding |
| Global | bowling_team | 32 | TeamEmbedding |
| State | score_state | 4 | [score/200, wickets/10, balls/120, innings] |
| State | chase_state | 3 | [runs_needed/200, RRR/15, is_chase] |
| State | phase_state | 4 | [powerplay, middle, death, progress] |
| State | time_pressure | 3 | [balls_remaining, inverse, is_death] |
| State | wicket_buffer | 2 | [wickets_left, is_danger] |
| Actor | striker_identity | 64 | PlayerEmbedding |
| Actor | striker_state | 6 | [runs, balls, SR, dots, setness, boundaries] |
| Actor | bowler_identity | 64 | PlayerEmbedding |
| Actor | bowler_state | 6 | [balls, runs, wickets, econ, dots, threat] |
| Actor | partnership | 4 | [runs, balls, RR, stability] |
| Dynamics | batting_momentum | 1 | recent_runs / 48 |
| Dynamics | bowling_momentum | 1 | -batting_momentum |
| Dynamics | pressure_index | 1 | wicket + chase + stage pressure |
| Dynamics | dot_pressure | 2 | [consecutive_dots, balls_since_boundary] |

---

### Level 4: Temporal Transformer Decomposition

```
TemporalTransformer
│
├── BallEmbedding
│   ├── Inputs per ball: [runs/6, is_wicket, progress/120, batter_idx, bowler_idx]
│   ├── player_embed: idx → 32d (shared for batter/bowler)
│   ├── feature_proj: [runs, wickets, overs] → 64d
│   └── Output: 128d per ball
│
├── PositionalEncoding
│   └── Sinusoidal encoding (standard transformer)
│
├── SpecializedMultiHeadAttention (4 heads)
│   ├── Head 0: Recency bias (recent balls weighted higher)
│   ├── Head 1: Same-bowler bias (spell coherence)
│   ├── Head 2: Same-batsman bias (form tracking)
│   └── Head 3+: Free to learn any pattern
│
├── Transformer Layers (×2)
│   ├── SpecializedMHA + LayerNorm + Residual
│   └── FFN (128 → 512 → 128) + LayerNorm + Residual
│
├── Query Token
│   └── Learned parameter that aggregates sequence
│
└── Output Projection
    └── Linear(128, 128)
```

**Specialized Head Biases:**

```python
# Head 0: More attention to recent balls
scores[:, 0] += recency_strength * (position / seq_len)

# Head 1: More attention to same bowler's balls
scores[:, 1] += same_bowler_strength * same_bowler_mask

# Head 2: More attention to same batsman's balls
scores[:, 2] += same_batsman_strength * same_batsman_mask
```

---

### Level 5: Fusion and Output

```
Fusion Layer
├── Input: [gat_out (128d), temporal_out (128d)] → 256d
├── Layer 1: Linear(256, 128) + ReLU + Dropout
├── Layer 2: Linear(128, 64) + ReLU
└── Output: 64d

Output Head
├── Input: 64d
├── Linear(64, 7)
└── Softmax → probabilities
```

**Outcome Classes:**
```
0: dot ball (0 runs)
1: single (1 run)
2: two (2 runs)
3: three (3 runs)
4: four (boundary)
5: six
6: wicket
```

---

## Dependency Analysis

### Information Flow

```
Dataset
   │
   ▼
┌─────────────────────┐
│ Entity Indices      │──────────────────────────┐
│ (batter, bowler,    │                          │
│  venue, teams)      │                          │
└─────────────────────┘                          │
   │                                             │
   ▼                                             │
┌─────────────────────┐                          │
│ EmbeddingManager    │                          │
│ → Dense vectors     │                          │
└─────────────────────┘                          │
   │                                             │
   ├──────────────────────────┐                  │
   │                          │                  │
   ▼                          ▼                  ▼
┌──────────────┐       ┌──────────────┐   ┌──────────────┐
│ Node Feature │       │   History    │   │  Raw State   │
│ Construction │       │   Tensors    │   │   Tensors    │
│ (17 nodes)   │       │ (24 balls)   │   │              │
└──────────────┘       └──────────────┘   └──────────────┘
   │                          │                  │
   ▼                          ▼                  │
┌──────────────┐       ┌──────────────┐          │
│ Hierarchical │       │   Temporal   │          │
│     GAT      │       │ Transformer  │          │
│   (128d)     │       │   (128d)     │          │
└──────────────┘       └──────────────┘          │
   │                          │                  │
   └──────────┬───────────────┘                  │
              │                                  │
              ▼                                  │
       ┌──────────────┐                          │
       │    Fusion    │◄─────────────────────────┘
       │   (64d)      │  (state/chase tensors used
       └──────────────┘   in node construction)
              │
              ▼
       ┌──────────────┐
       │   Output     │
       │  (7 probs)   │
       └──────────────┘
```

### Critical Path

**Latency bottleneck:** ActorLayerGAT processes batch items sequentially (line 191-197 in hierarchical.py)

```python
for i in range(batch_size):
    x = actor_nodes[i]
    out, (_, attn) = self.gat(x, self.edge_index, ...)
```

**Memory bottleneck:** Embedding tables scale with unique entities
- Players: ~10,000+ unique → 10,000 × 64 = 640KB
- Venues: ~100 unique → 100 × 32 = 3.2KB
- Teams: ~50 unique → 50 × 32 = 1.6KB

---

## Reconstruction: Optimization Opportunities

### 1. Bottleneck: Sequential GAT Processing

**Current:** Loop over batch for GAT
**Recommendation:** Batch GAT using edge_index repetition

```python
# Instead of looping:
batched_x = actor_nodes.view(-1, hidden_dim)  # [batch*5, hidden]
batched_edge_index = self.edge_index.repeat(1, batch_size) + \
                     torch.arange(batch_size)[:, None] * 5
out = self.gat(batched_x, batched_edge_index)
```

### 2. Parallelization: GAT and Transformer

**Current:** Run in parallel (good)
**Status:** Already optimal - both branches computed independently

### 3. Potential Simplification

**Question:** Is 4-layer hierarchy necessary?
**Experiment:** Compare layer importance weights after training
- If one layer consistently low → consider removing
- Use `return_attention=True` to analyze

### 4. Cold-Start Enhancement

**Current:** Feature-based + learned embedding fusion
**Gap:** Stats not passed during training (only indices used)
**Recommendation:** Include stats in training for true hybrid embeddings

---

## Key Metrics for Monitoring

| Metric | What It Measures | Where to Extract |
|--------|------------------|------------------|
| `layer_importance` | Relative contribution of 4 GAT layers | `gat_attention["layer_importance"]` |
| `global_attention` | Venue vs team importance | `gat_attention["global"]` |
| `head_attention` | Recency/bowler/batsman patterns | `temporal_attention["head_attention"]` |
| `aggregate_attention` | Which history balls matter most | `temporal_attention["aggregate"]` |

---

## Summary

The CricketPredictor is a **dual-stream architecture**:

1. **Spatial stream (HierarchicalGAT):** Models current ball context through 4 layers of attention (Global → State → Actor → Dynamics) with information flowing top-down.

2. **Temporal stream (TemporalTransformer):** Models ball history through specialized attention heads that encode cricket-specific patterns (recency, same-bowler spells, same-batsman form).

3. **Fusion:** Late fusion combines both streams before final classification.

**Key architectural insight:** The model processes "current state" and "history" in parallel rather than using history to enrich per-ball representations (which would be more expensive but potentially more powerful).
