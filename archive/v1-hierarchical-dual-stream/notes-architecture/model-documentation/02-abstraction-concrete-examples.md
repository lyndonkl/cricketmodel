# Cricket Prediction Model: Abstraction Ladder

## Purpose

This document presents the model architecture at 5 levels of abstraction, from high-level principles to concrete implementation details. Use this to communicate with different audiences and to trace how abstract concepts manifest in code.

---

## Level 1: Abstract Principles (Universal)

**Core Insight:** Cricket ball prediction requires understanding WHAT (current situation) and HOW (historical patterns) in context.

1. **Hierarchical Context Matters**
   - Some factors are more fundamental than others
   - Global factors (venue, teams) shape match-level dynamics
   - Local factors (current players, momentum) drive ball-level outcomes

2. **Temporal Patterns Are Structured**
   - Cricket has natural temporal units (balls, overs, phases)
   - Player patterns repeat within these structures
   - Recent events matter more than distant ones

3. **Relationships Define Outcomes**
   - Batter vs bowler matchup
   - Partnership dynamics
   - Team strategic tendencies

4. **Dual Information Streams**
   - Static context (who, where) + Dynamic history (what happened)
   - Both streams contribute to prediction

---

## Level 2: Frameworks and Standards

### Framework: Hierarchical Attention for Cricket

```
┌─────────────────────────────────────────────────┐
│              HIERARCHY OF INFLUENCE             │
├─────────────────────────────────────────────────┤
│ L1: Global Context    │ Venue conditions, team   │
│     (most stable)     │ strengths, match type    │
├───────────────────────┼─────────────────────────┤
│ L2: Match State       │ Score, wickets, target,  │
│     (slowly changing) │ phase, pressure          │
├───────────────────────┼─────────────────────────┤
│ L3: Actor State       │ Current batter/bowler,   │
│     (changes per ball)│ partnership, form        │
├───────────────────────┼─────────────────────────┤
│ L4: Dynamics          │ Momentum, dot pressure,  │
│     (most volatile)   │ immediate pressure       │
└───────────────────────┴─────────────────────────┘
```

### Framework: Specialized Temporal Attention

```
┌─────────────────────────────────────────────────┐
│         TEMPORAL ATTENTION SPECIALIZATION       │
├─────────────────────────────────────────────────┤
│ Head 0: Recency       │ "What just happened?"    │
│                       │ Recent balls weighted    │
├───────────────────────┼─────────────────────────┤
│ Head 1: Bowler Spell  │ "How is this bowler      │
│                       │  performing today?"      │
├───────────────────────┼─────────────────────────┤
│ Head 2: Batter Form   │ "How is this batter      │
│                       │  playing right now?"     │
├───────────────────────┼─────────────────────────┤
│ Head 3: Free Learning │ Discover emergent        │
│                       │ patterns from data       │
└───────────────────────┴─────────────────────────┘
```

### Framework: Dual-Stream Architecture

```
         Input Batch
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌────────┐        ┌────────┐
│ Static │        │ Temporal│
│ Context│        │ History │
│ (GAT)  │        │ (Transf)│
└────┬───┘        └────┬───┘
     │                 │
     └────────┬────────┘
              │
        Late Fusion
              │
        Prediction
```

---

## Level 3: Methods and Approaches

### Method: Building Node Features for GAT

**Goal:** Transform raw data into 17 semantically meaningful nodes

**Approach:**
1. **Embed categorical entities** (players, venues, teams) → dense vectors
2. **Project numeric features** to common hidden dimension
3. **Compute derived features** from history (momentum, pressure)
4. **Organize into hierarchy** (global → state → actor → dynamics)

**Example: Building "striker_state" node**
```
Input: history_runs tensor (24 balls)
Compute:
  - runs = sum(history_runs) * 6
  - balls = count non-padding
  - strike_rate = runs / balls * 100
  - dots = count zeros
  - boundaries = count >= 4
  - setness = balls / 30 (capped at 1)
Output: [runs/100, balls/60, SR/200, dots/balls, setness, boundaries/10]
Project: 6-dim → 128-dim via Linear layer
```

### Method: Specialized Attention Biases

**Goal:** Encode cricket domain knowledge into attention mechanism

**Approach:**
1. Compute standard attention scores (Q·K^T / √d)
2. Add **learned bias** to specific heads based on masks
3. Apply softmax to get final attention weights

**Example: Same-bowler attention**
```
# Build mask: 1 where balls i and j have same bowler
same_bowler_mask[i, j] = (bowler[i] == bowler[j])

# Add bias to Head 1 scores
scores[:, 1, :, :] += same_bowler_strength * same_bowler_mask

# Effect: Head 1 naturally attends more to same-bowler balls
```

### Method: Hierarchical Conditioning

**Goal:** Let higher layers influence lower layers

**Approach:**
1. Process Layer 1 (Global) → get h_global
2. Process Layer 2 (State) with h_global as cross-attention key/value
3. Process Layer 3 (Actor) with h_state added as context bias
4. Process Layer 4 (Dynamics) with h_actor added as context bias

**Example: Actor layer conditioning**
```
context = Linear(h_state)  # Project state to same dim
actor_nodes = stack([striker_id, striker_state, ...])
actor_nodes = actor_nodes + context.unsqueeze(1)  # Broadcast add
# Now every actor node is "aware" of match state
```

---

## Level 4: Specific Implementations

### Implementation: GlobalContextAttention

**Input:** venue (128d), batting_team (128d), bowling_team (128d)
**Output:** h_global (128d), attention weights (3,)

```python
class GlobalContextAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4):
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, venue, batting_team, bowling_team):
        # Stack: [batch, 3, 128]
        global_nodes = torch.stack([venue, batting_team, bowling_team], dim=1)

        # Learned query attends to global nodes
        query = self.global_query.expand(batch_size, -1, -1)
        h_global, attn = self.attention(query, global_nodes, global_nodes)

        return self.norm(h_global.squeeze(1)), attn
```

**Why this design:**
- Learned query = model learns what global information matters
- Multi-head = captures different aspects (venue characteristics, team matchup)
- LayerNorm = stable training

### Implementation: ActorLayerGAT

**Input:** 5 actor nodes (each 128d), h_state (128d)
**Output:** h_actor (128d), GAT attention weights

```python
class ActorLayerGAT(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4):
        self.gat = GATv2Conv(hidden_dim, hidden_dim//4, heads=4, concat=True)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        # Fixed graph structure
        edges = [
            [0, 1], [1, 0],  # striker_id ↔ striker_state
            [2, 3], [3, 2],  # bowler_id ↔ bowler_state
            [0, 2], [2, 0],  # striker ↔ bowler (MATCHUP)
            [1, 4], [4, 1],  # striker_state ↔ partnership
            [3, 4], [4, 3],  # bowler_state ↔ partnership
        ]
        self.edge_index = torch.tensor(edges).t()

    def forward(self, striker_id, striker_state, bowler_id, bowler_state,
                partnership, h_state):
        # Add match state context to all actor nodes
        context = self.context_proj(h_state).unsqueeze(1)
        actor_nodes = torch.stack([...], dim=1) + context

        # Run GAT per batch item (current bottleneck)
        for i in range(batch_size):
            out, attn = self.gat(actor_nodes[i], self.edge_index)
```

**Why this design:**
- GATv2Conv = dynamic attention based on both source and target
- Fixed edges = encode cricket relationships (matchup is explicit edge)
- Context projection = state influences actor interactions

### Implementation: SpecializedMultiHeadAttention

**Input:** x (batch, 24, 128), same_bowler_mask, same_batsman_mask
**Output:** attended (batch, 24, 128), attention weights

```python
class SpecializedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4):
        # Standard projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # LEARNED bias strengths
        self.recency_strength = nn.Parameter(torch.tensor(0.5))
        self.same_bowler_strength = nn.Parameter(torch.tensor(2.0))
        self.same_batsman_strength = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, same_bowler_mask, same_batsman_mask):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = Q @ K.T / sqrt(head_dim)

        # Head 0: Recency bias
        positions = torch.arange(seq_len)
        scores[:, 0] += self.recency_strength * (positions / seq_len)

        # Head 1: Same-bowler bias
        scores[:, 1] += self.same_bowler_strength * same_bowler_mask

        # Head 2: Same-batsman bias
        scores[:, 2] += self.same_batsman_strength * same_batsman_mask

        # Head 3: No bias - free to learn
```

**Why this design:**
- Learnable strengths = model can adjust bias importance
- Additive bias = combines with learned attention, doesn't override
- Separate heads = each specialization has dedicated capacity

---

## Level 5: Concrete Details and Edge Cases

### Concrete Example: Full Forward Pass

**Scenario:** Ball 45 of 2nd innings, India 52/2 chasing 156 at MCG

**Input tensors:**
```python
batch = {
    "state": tensor([0.26, 0.2, 0.375, 1.0]),     # score/200, wkts/10, balls/120, inns
    "chase": tensor([0.52, 0.555, 1.0]),          # needed/200, RRR/15, is_chase
    "batter_idx": tensor(847),                    # Rohit Sharma
    "bowler_idx": tensor(234),                    # Jasprit Bumrah
    "venue_idx": tensor(15),                      # MCG
    "batting_team_idx": tensor(3),                # India
    "bowling_team_idx": tensor(7),                # Australia
    "history_runs": tensor([0, 0, 0, 0.17, ...]), # last 24 balls normalized
    "history_wickets": tensor([0, 0, 0, 0, ...]),
    "history_overs": tensor([0, 0, 0, 0.18, ...]),
    "history_batters": tensor([0, 0, 0, 847, ...]),
    "history_bowlers": tensor([0, 0, 0, 234, ...]),
}
```

**Step 1: Embeddings**
```python
embeddings = EmbeddingManager(batch)
# batter: 847 → [0.12, -0.34, ...] (64d)
# bowler: 234 → [0.08, 0.21, ...] (64d)
# venue: 15 → [-0.15, 0.45, ...] (32d)
# teams: 3 → [...], 7 → [...] (32d each)
```

**Step 2: Node Construction (17 nodes)**
```python
nodes = _build_node_features(batch)
# venue: (32d) → Linear → (128d)
# batting_team: (32d) → Linear → (128d)
# ...
# batting_momentum:
#   recent = history_runs[:, -12:]
#   momentum = (recent.sum() * 6 / 48) * 2 - 1
#   momentum.clamp(-1, 1) → Linear → (128d)
```

**Step 3: Hierarchical GAT**
```python
# Layer 1: Global
h_global = attention([venue, bat_team, bowl_team])  # 128d
# attn_weights might be [0.15, 0.42, 0.43] → teams matter more

# Layer 2: State (conditioned on h_global)
h_state = self_attn([score, chase, phase, time, wicket]) + cross_attn(h_global)

# Layer 3: Actor (conditioned on h_state)
h_actor = GAT([striker_id, striker_state, bowler_id, bowler_state, partnership])
# GAT attention on matchup edge might be high (0.35) → key matchup

# Layer 4: Dynamics (conditioned on h_actor)
h_dynamics = self_attn([bat_momentum, bowl_momentum, pressure, dot_pressure])
```

**Step 4: Temporal Transformer**
```python
# Embed history
ball_embeds = BallEmbedding(runs, wickets, overs, batters, bowlers)  # (24, 128)

# Add query token
x = cat([ball_embeds, query_token])  # (25, 128)

# Specialized attention
# Head 0 (recency): attends most to balls 20-23
# Head 1 (same_bowler): attends to balls where Bumrah bowled
# Head 2 (same_batsman): attends to balls where Rohit faced

temporal_out = x[:, -1]  # Extract query token output (128d)
```

**Step 5: Fusion and Output**
```python
combined = cat([gat_out, temporal_out])  # (256d)
fused = MLP(combined)  # (64d)
logits = Linear(fused)  # (7,)
probs = softmax(logits)
# [0.35, 0.28, 0.12, 0.05, 0.10, 0.05, 0.05]
#  dot   1     2     3     4     6     wicket
```

### Edge Case: Early Innings (Short History)

**Ball 3 of innings:**
```python
history_runs = [0.0, 0.0, ..., 0.17, 0.0]  # 21 zeros + 2 real balls
# Padding on LEFT preserves temporal order
# Position 23 is still most recent ball
# Transformer learns to ignore padded positions (zero runs, zero wickets)
```

### Edge Case: Unknown Player (Cold Start)

**New player not in training data:**
```python
player_idx = 0  # Unknown token
# EmbeddingTable returns embedding[0] (learned unknown embedding)
# OR if stats provided:
stats = PlayerStats(batting_sr=130, batting_avg=25, ...)
embedding = PlayerEmbedding.from_stats(stats)  # Feature-based generation
```

### Edge Case: Chase Calculation

**First innings (no target):**
```python
target = None
chase = tensor([0.0, 0.0, 0.0])  # All zeros
# Model learns that chase[2] == 0 means "set target mode"
```

**Second innings:**
```python
target = first_innings_total + 1
runs_needed = target - score
balls_remaining = 120 - balls_faced
RRR = runs_needed / (balls_remaining / 6)  # Required run rate
chase = tensor([runs_needed/200, RRR/15, 1.0])
```

---

## Audience Guide

| Audience | Start At | Focus On |
|----------|----------|----------|
| Executives | Level 1 | Principles and dual-stream concept |
| Product Managers | Level 2 | Frameworks, what each component does |
| ML Engineers | Level 3-4 | Methods, implementation patterns |
| Debuggers | Level 5 | Concrete values, edge cases |
| LLM Commentators | Level 4-5 | Attention weights, how to interpret |
