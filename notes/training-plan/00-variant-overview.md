# Model Variant Overview

This document explains the 6 CricketHeteroGNN model variants, their inheritance structure, and why each exists.

---

## Inheritance Hierarchy

```
nn.Module
├── CricketHeteroGNN (Base)
│   ├── CricketHeteroGNNWithPooling
│   └── CricketHeteroGNNHybrid
│       ├── CricketHeteroGNNPhaseModulated
│       └── CricketHeteroGNNInningsConditional
│
└── CricketHeteroGNNFull (Standalone)
```

**Key insight**: There are two independent roots - the Base hierarchy and Full as standalone.

---

## Why CricketHeteroGNNFull is Standalone

Full does NOT inherit from any other variant. This is intentional:

### 1. Diamond Problem Avoidance
PhaseModulated and InningsConditional both extend Hybrid. If Full tried to combine both via multiple inheritance:

```python
# This would create ambiguity:
class Full(PhaseModulated, InningsConditional):  # MRO conflict!
    pass
```

### 2. Different Conditioning Dimensions
- **PhaseModulated**: 6-dim condition vector (phase one-hot)
- **Full**: 15-dim condition vector (phase + chase_context + wickets_fallen)

### 3. Config-Based Ablation
Full uses boolean flags for clean ablation studies:

```python
class CricketHeteroGNNFull:
    def __init__(
        self,
        use_film=True,           # Toggle FiLM modulation
        use_hierarchical=True,   # Toggle hierarchical embeddings
        use_innings_heads=True,  # Toggle dual innings heads
        ...
    ):
```

### 4. Production Deployment
A single, configurable class is easier to deploy than choosing between 6 variants.

---

## Component Matrix

| Component | Base | Pooling | Hybrid | Phase | Innings | Full |
|-----------|:----:|:-------:|:------:|:-----:|:-------:|:----:|
| HeteroConvBlock | ✓ | ✓ | ✓ | - | ✓ | - |
| PhaseModulatedConvBlock | - | - | - | ✓ | - | ✓ |
| `ball_attention` | - | ✓ | - | - | - | - |
| `matchup_mlp` | - | - | ✓ | ✓ | ✓ | ✓ |
| `nonstriker_gate` | - | - | ✓ | ✓ | ✓ | ✓ |
| Dual innings heads | - | - | - | - | ✓ | ✓ |
| Hierarchical embeddings | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* |

*Full has configurable hierarchical embeddings via `use_hierarchical` flag.

---

## Variant Descriptions

### 1. CricketHeteroGNN (Base)

**Location**: `src/model/hetero_gnn.py`

**Purpose**: Minimal viable heterogeneous GNN for cricket prediction.

**Architecture**:
- Node encoders for all 21 node types
- HeteroConvBlock for message passing
- Query-based readout
- Single prediction head

**Use case**: Baseline for ablation studies.

```python
model = CricketHeteroGNN(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1
)
```

---

### 2. CricketHeteroGNNWithPooling

**Extends**: CricketHeteroGNN

**Purpose**: Test hypothesis that explicit ball sequence pooling improves prediction.

**Added components**:
- `ball_attention`: Learned attention over ball representations
- `scatter_softmax`: Per-sample softmax for proper batching
- `global_add_pool`: Per-sample weighted sum

**Key code**:
```python
# Per-sample attention-weighted pooling
attn_scores = self.ball_attention(ball_repr).squeeze(-1)
attn_weights = scatter_softmax(attn_scores, ball_batch, dim=0)
weighted_balls = attn_weights.unsqueeze(-1) * ball_repr
pooled_balls = global_add_pool(weighted_balls, ball_batch, size=batch_size)
```

**Use case**: When ball sequence matters more than graph structure.

---

### 3. CricketHeteroGNNHybrid

**Extends**: CricketHeteroGNN

**Purpose**: Combine graph structure with explicit matchup modeling.

**Added components**:
- `matchup_mlp`: Explicit batter-bowler interaction modeling
- `nonstriker_gate`: Learned gate for non-striker influence

**Key insight**: The graph captures structure, but the matchup MLP explicitly models the batter-bowler interaction that dominates cricket outcomes.

```python
# Explicit matchup modeling
matchup = torch.cat([batter_repr, bowler_repr], dim=-1)
matchup_out = self.matchup_mlp(matchup)

# Gated non-striker contribution
gate = torch.sigmoid(self.nonstriker_gate(nonstriker_repr))
combined = query_out + matchup_out + gate * nonstriker_out
```

**Use case**: When explicit matchup matters (recommended default).

---

### 4. CricketHeteroGNNPhaseModulated

**Extends**: CricketHeteroGNNHybrid

**Purpose**: Test FiLM modulation for phase-conditional message passing.

**Key change**: Replaces HeteroConvBlock with PhaseModulatedConvBlock.

**FiLM mechanism**:
```python
# Phase conditions the message passing
gamma = self.film_gamma(phase_encoding)  # [batch, hidden_dim]
beta = self.film_beta(phase_encoding)    # [batch, hidden_dim]
modulated = gamma * node_repr + beta     # Affine transformation
```

**Phase encoding**: 6-dimensional one-hot (powerplay, middle_1, middle_2, etc.)

**Use case**: When game phase significantly affects prediction (T20 cricket).

---

### 5. CricketHeteroGNNInningsConditional

**Extends**: CricketHeteroGNNHybrid

**Purpose**: Model first/second innings asymmetry.

**Key insight**: Second innings has a target, changing optimal strategy:
- First innings: Maximize runs
- Second innings: Chase target efficiently

**Added components**:
- `innings_1_head`: Prediction head for first innings
- `innings_2_head`: Prediction head for second innings
- Innings routing logic

```python
# Route to appropriate head
if is_first_innings:
    logits = self.innings_1_head(combined)
else:
    logits = self.innings_2_head(combined)
```

**Use case**: When innings asymmetry matters (test/ODI cricket).

---

### 6. CricketHeteroGNNFull

**Extends**: nn.Module (standalone)

**Purpose**: Production-ready model combining all features with ablation flags.

**Configurable features**:
```python
model = CricketHeteroGNNFull(
    hidden_dim=128,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    use_film=True,           # FiLM modulation
    use_hierarchical=True,   # Hierarchical player embeddings
    use_innings_heads=True,  # Dual innings prediction heads
)
```

**Condition vector** (15-dim):
- Phase one-hot (6 dims)
- Chase context (3 dims): target, required_rate, is_chasing
- Wickets fallen one-hot (6 dims for 0-5+ wickets)

**Use case**: Production deployment with clean ablation support.

---

## Choosing a Variant

| Scenario | Recommended Variant |
|----------|---------------------|
| Baseline experiments | Base |
| Ball sequence hypothesis | WithPooling |
| Matchup-focused prediction | Hybrid |
| T20 with phase effects | PhaseModulated |
| Test/ODI with chase dynamics | InningsConditional |
| Production deployment | Full |
| Ablation studies | Full (with flags) |

---

## Trade-offs Summary

| Variant | Parameters | Complexity | Interpretability |
|---------|------------|------------|------------------|
| Base | ~500K | Low | High |
| WithPooling | ~520K | Low | High |
| Hybrid | ~600K | Medium | Medium |
| PhaseModulated | ~700K | High | Medium |
| InningsConditional | ~650K | Medium | Medium |
| Full | ~800K | High | Low |

*Parameter counts are approximate for hidden_dim=128, num_layers=3.*
