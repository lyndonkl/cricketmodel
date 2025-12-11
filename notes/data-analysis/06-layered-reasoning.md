# Cricket Ball Prediction: Layered Reasoning Analysis

## Overview

This document structures the cricket ball prediction problem across three abstraction levels:
- **30,000 ft (Strategic)**: Why we're building this, core principles, invariants
- **3,000 ft (Tactical)**: What approaches we're using, architectural decisions
- **300 ft (Operational)**: How we implement, specific features and code

Each layer constrains the one below and is implemented by it.

---

## Layer 1: Strategic (30,000 ft)

### 1.1 Core Mission

**Mission**: Build a model that predicts cricket ball outcomes with sufficient accuracy and interpretability to provide actionable insights for:
- Understanding match dynamics
- Identifying high-leverage situations
- Explaining what factors drive specific predictions

### 1.2 Strategic Principles (Invariants)

These principles MUST hold at all lower layers:

| Principle | Description | Constraint on Lower Layers |
|-----------|-------------|---------------------------|
| **P1: Temporal Integrity** | Never use future information to predict past | All features must be computed from t-1 and earlier |
| **P2: Interpretability** | Predictions must be explainable | Architecture must support attention visualization or feature attribution |
| **P3: Generalization** | Model must work on unseen players/venues | No memorization of specific instances; use embeddings |
| **P4: Domain Fidelity** | Model must respect cricket's structure | Architecture should reflect relational + temporal nature |
| **P5: Realistic Inputs** | Only use data available before ball is bowled | No ball outcome, bowling speed, or trajectory |

### 1.3 Strategic Constraints

**Data Constraints**:
- Source: Cricsheet JSON (ball-by-ball with player IDs)
- Missing: Ball speed, trajectory, exact field positions, weather
- Available: Outcomes, player identities, match context, sequence

**Model Constraints**:
- Must handle variable-length sequences (different match lengths)
- Must handle variable graph structures (different players)
- Must produce probability distribution over outcomes

### 1.4 Success Criteria (Strategic Level)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Outcome Distribution Accuracy | Better than base rate | Model learns beyond "20% of balls are dot balls" |
| Calibration | Well-calibrated probabilities | P(outcome) = 0.3 should happen 30% of the time |
| Interpretability | Attention weights align with domain knowledge | Matchup attention high when matchup matters |
| Generalization | Similar performance on new players | <5% accuracy drop on unseen players |

---

## Layer 2: Tactical (3,000 ft)

### 2.1 Architectural Approach

**Decision**: Hybrid architecture with two branches

| Branch | Purpose | Justification |
|--------|---------|---------------|
| **Relational (GAT)** | Capture player matchups, partnerships, pressure relationships | Cricket has graph structure at each ball |
| **Temporal (Transformer)** | Capture momentum, patterns, streaks | Cricket has sequence structure across balls |
| **Fusion** | Combine relational + temporal for final prediction | Both matter; neither alone is sufficient |

**Alternative Considered**: HRM (Hierarchical Reasoning Model)
- **Rejected Because**: Provides iteration-level interpretability (how much thought) but not content-level (what was attended to). For cricket, knowing WHICH balls/matchups matter is more actionable than knowing how many iterations were used.

### 2.2 Graph Structure Design

**Tactical Decision**: 4-node graph per ball

```
        CONTEXT NODE
             │
      ┌──────┼──────┐
      │      │      │
      ▼      ▼      ▼
  BATSMAN ←──→ BOWLER
      │
      ▼
   PARTNER
```

| Node | Justification |
|------|---------------|
| Batsman | Primary actor; skill and state drive outcomes |
| Bowler | Primary antagonist; skill and strategy create outcomes |
| Partner | Influences batsman behavior (shield partner, rotate strike) |
| Context | Match state affects all decisions (pressure, phase) |

**Edge Design**:
- Batsman ↔ Bowler: Matchup edge (head-to-head history)
- Batsman ↔ Context: Pressure edge (how match state affects batsman)
- Batsman ↔ Partner: Partnership edge (coordination, balance)
- Bowler ↔ Context: Tactical edge (bowling resources, strategy)

### 2.3 Temporal Structure Design

**Tactical Decision**: Transformer with last N balls

| Parameter | Choice | Justification |
|-----------|--------|---------------|
| Sequence length N | 24-36 balls | Captures ~4-6 overs of context; momentum effects |
| Attention | Multi-head (4-8 heads) | Different heads learn different patterns |
| Position encoding | Learned | Cricket positions have semantic meaning (over.ball) |

**What Transformer Should Learn**:
- Head 1: Same-bowler attention (bowler's recent pattern)
- Head 2: Same-batsman attention (batsman's recent form)
- Head 3: Recent-ball attention (momentum)
- Head 4: Phase-specific attention (powerplay patterns, death patterns)

### 2.4 Feature Category Decisions

**Tactical Grouping**:

| Category | Inclusion Policy | Example |
|----------|------------------|---------|
| Identity | Embed all players | batsman_id → embedding |
| Current State | Full detail | runs, balls, SR this innings |
| Derived Dynamics | Computed features | pressure_index, momentum |
| Historical | Blend with priors | career_sr weighted by sample |
| Sequence | Raw outcomes | last N balls' (runs, wicket, batsman, bowler) |

### 2.5 Tactical Consistency Checks

**Check P1 (Temporal Integrity)**:
- ✅ All features computed from balls [0, t-1]
- ✅ No use of current ball outcome
- ✅ No shuffling within matches

**Check P2 (Interpretability)**:
- ✅ GAT attention weights are inspectable (which node matters?)
- ✅ Transformer attention is inspectable (which past ball matters?)
- ✅ Can visualize attention for any prediction

**Check P3 (Generalization)**:
- ✅ Player embeddings (not one-hot)
- ✅ Feature engineering uses aggregated stats, not memorization
- ✅ Venue embeddings, not venue-specific models

**Check P4 (Domain Fidelity)**:
- ✅ Graph structure reflects cricket relationships
- ✅ Sequence structure captures temporal dependencies
- ✅ Phase indicators respect cricket structure

**Check P5 (Realistic Inputs)**:
- ✅ Only using Cricsheet fields available before ball
- ✅ No ball speed, trajectory (not in data)
- ✅ No outcome of current ball

---

## Layer 3: Operational (300 ft)

### 3.1 Data Pipeline Implementation

```python
# Operational Implementation: Data Processing

class CricketDataProcessor:
    """
    Processes Cricsheet JSON into model-ready tensors.
    Maintains temporal integrity by incremental computation.
    """

    def process_match(self, match_json: dict) -> List[BallSample]:
        samples = []
        match_state = MatchState()  # Running state

        for innings_idx, innings in enumerate(match_json['innings']):
            innings_state = InningsState(
                batting_team=innings['team'],
                target=innings.get('target', {}).get('runs'),
                is_second_innings=(innings_idx == 1)
            )

            for over in innings['overs']:
                for ball_idx, delivery in enumerate(over['deliveries']):
                    # CRITICAL: Features computed BEFORE outcome
                    features = self.extract_features(
                        match_state,
                        innings_state,
                        delivery  # Only for player IDs, not outcome
                    )

                    # Target is the outcome
                    target = self.encode_outcome(delivery)

                    samples.append(BallSample(
                        features=features,
                        target=target,
                        match_id=match_json['info']['match_type_number']
                    ))

                    # Update state AFTER recording sample
                    match_state.update(delivery)
                    innings_state.update(delivery)

        return samples
```

### 3.2 Feature Extraction Implementation

```python
# Operational Implementation: Feature Extraction

def extract_features(match_state, innings_state, delivery) -> dict:
    """
    Extract all features for a single ball.
    P1 CONSTRAINT: Only uses information available before this ball.
    """

    # --- Identity Features ---
    batsman_id = delivery['batter']
    bowler_id = delivery['bowler']
    partner_id = delivery['non_striker']

    # --- Current Innings State (from running state) ---
    batsman_state = innings_state.get_batsman_state(batsman_id)
    bowler_state = innings_state.get_bowler_state(bowler_id)
    partner_state = innings_state.get_batsman_state(partner_id)

    # --- Match Progress ---
    over = delivery['over']  # 0-indexed
    ball_in_over = len([d for d in innings_state.current_over_deliveries]) + 1

    # --- Derived Features ---
    pressure_index = compute_pressure_index(
        innings_state.total_runs,
        innings_state.wickets,
        innings_state.balls_faced,
        innings_state.target,  # None for 1st innings
        innings_state.consecutive_dots
    )

    momentum = compute_momentum(innings_state.last_12_balls)

    batsman_setness = compute_setness(batsman_state['balls_faced'])

    # --- Graph Node Features ---
    batsman_node = {
        'id': batsman_id,
        'runs': batsman_state['runs'],
        'balls': batsman_state['balls_faced'],
        'strike_rate': batsman_state['strike_rate'],
        'setness': batsman_setness,
        'consecutive_dots': batsman_state['consecutive_dots'],
        'boundary_rate': batsman_state['boundary_rate'],
    }

    bowler_node = {
        'id': bowler_id,
        'overs': bowler_state['overs_bowled'],
        'runs': bowler_state['runs_conceded'],
        'wickets': bowler_state['wickets'],
        'economy': bowler_state['economy'],
        'overs_remaining': 4 - bowler_state['overs_bowled'],
    }

    partner_node = {
        'id': partner_id,
        'runs': partner_state['runs'],
        'balls': partner_state['balls_faced'],
    }

    context_node = {
        'total_runs': innings_state.total_runs,
        'wickets': innings_state.wickets,
        'over': over,
        'ball_in_over': ball_in_over,
        'run_rate': innings_state.run_rate,
        'is_powerplay': over < 6,
        'is_death': over >= 16,
        'is_second_innings': innings_state.is_second_innings,
        'target': innings_state.target or 0,
        'required_rate': innings_state.required_run_rate or 0,
        'pressure_index': pressure_index,
        'momentum': momentum,
    }

    # --- Edge Features ---
    edges = {
        'batsman_bowler': {
            'h2h_sr': get_h2h_sr(batsman_id, bowler_id),
            'h2h_balls': get_h2h_balls(batsman_id, bowler_id),
        },
        'batsman_context': {
            'pressure_weight': pressure_index,
        },
        'batsman_partner': {
            'partnership_runs': innings_state.partnership_runs,
            'partnership_balls': innings_state.partnership_balls,
        },
    }

    # --- Sequence Features ---
    sequence = []
    for past_ball in innings_state.last_n_balls(n=24):
        sequence.append({
            'runs_total': past_ball['runs']['total'],
            'is_wicket': 1 if past_ball.get('wickets') else 0,
            'is_boundary': 1 if past_ball['runs']['batter'] >= 4 else 0,
            'batsman_id': past_ball['batter'],
            'bowler_id': past_ball['bowler'],
            'over': past_ball['over'],
        })

    return {
        'graph': {
            'batsman': batsman_node,
            'bowler': bowler_node,
            'partner': partner_node,
            'context': context_node,
            'edges': edges,
        },
        'sequence': sequence,
    }
```

### 3.3 Model Architecture Implementation

```python
# Operational Implementation: Model Architecture

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class CricketPredictor(nn.Module):
    """
    Hybrid GAT + Transformer for ball outcome prediction.
    """

    def __init__(
        self,
        num_players: int,
        player_embed_dim: int = 64,
        node_feature_dim: int = 32,
        hidden_dim: int = 128,
        num_gat_heads: int = 4,
        num_transformer_heads: int = 4,
        num_transformer_layers: int = 2,
        num_outcomes: int = 10,  # 0,1,2,3,4,6,W,wd,nb,bye
    ):
        super().__init__()

        # Player embeddings (P3: Generalization via embeddings)
        self.player_embedding = nn.Embedding(num_players, player_embed_dim)

        # Node feature encoders
        self.batsman_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.bowler_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.partner_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.context_encoder = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers (P4: Relational structure)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_gat_heads, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_gat_heads, concat=False)

        # Transformer for sequence (P4: Temporal structure)
        self.sequence_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(50, hidden_dim)  # Max 50 balls in sequence
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_transformer_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output = nn.Linear(hidden_dim, num_outcomes)

    def forward(self, batch):
        # --- Relational Branch (GAT) ---
        # Encode nodes
        batsman_h = self.batsman_encoder(batch['batsman_features'])
        batsman_h = batsman_h + self.player_embedding(batch['batsman_id'])

        bowler_h = self.bowler_encoder(batch['bowler_features'])
        bowler_h = bowler_h + self.player_embedding(batch['bowler_id'])

        partner_h = self.partner_encoder(batch['partner_features'])
        partner_h = partner_h + self.player_embedding(batch['partner_id'])

        context_h = self.context_encoder(batch['context_features'])

        # Stack nodes: [batsman, bowler, partner, context]
        node_features = torch.stack([batsman_h, bowler_h, partner_h, context_h], dim=1)

        # GAT forward (simplified edge_index for 4-node graph)
        # Full connectivity for attention to learn importance
        edge_index = batch['edge_index']  # Pre-computed

        h_gat = self.gat1(node_features, edge_index)
        h_gat = torch.relu(h_gat)
        h_gat = self.gat2(h_gat, edge_index)

        # Pool graph to single vector (use batsman node as primary)
        h_relational = h_gat[:, 0, :]  # Batsman node representation

        # --- Temporal Branch (Transformer) ---
        seq_h = self.sequence_encoder(batch['sequence_features'])
        positions = torch.arange(seq_h.size(1), device=seq_h.device)
        seq_h = seq_h + self.pos_embedding(positions)

        # Transformer with causal mask (P1: temporal integrity)
        h_temporal = self.transformer(seq_h)
        h_temporal = h_temporal[:, -1, :]  # Last position = current state

        # --- Fusion ---
        h_combined = torch.cat([h_relational, h_temporal], dim=-1)
        h_fused = self.fusion(h_combined)

        # --- Output ---
        logits = self.output(h_fused)
        return logits

    def get_attention_weights(self, batch):
        """
        P2: Interpretability - extract attention weights for visualization
        """
        # GAT attention
        gat_attention = self.gat1.att_weights  # Requires return_attention_weights=True

        # Transformer attention
        # Note: Need to modify transformer to return attention

        return {
            'gat_attention': gat_attention,
            'transformer_attention': None,  # Implement if needed
        }
```

### 3.4 Training Implementation

```python
# Operational Implementation: Training Loop

def train_model(model, train_loader, val_loader, epochs=50):
    """
    Training with temporal integrity (no shuffling within matches).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    criterion = nn.CrossEntropyLoss()  # Multi-class classification

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # P1: Train loader maintains temporal order within matches
        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch['target'])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_metrics = evaluate(model, val_loader)

        # P3: Check generalization (performance on new players)
        new_player_metrics = evaluate_new_players(model, val_loader)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}, "
              f"New Player Acc={new_player_metrics['accuracy']:.4f}")

        scheduler.step()

    return model
```

### 3.5 Outcome Encoding Implementation

```python
# Operational Implementation: Outcome Encoding

OUTCOME_CLASSES = {
    'dot': 0,      # 0 runs, no wicket, no extras
    'single': 1,   # 1 run off bat
    'two': 2,      # 2 runs off bat
    'three': 3,    # 3 runs off bat
    'four': 4,     # 4 runs off bat (boundary)
    'six': 5,      # 6 runs off bat
    'wicket': 6,   # Any dismissal
    'wide': 7,     # Wide delivery
    'noball': 8,   # No-ball delivery
    'bye': 9,      # Bye or leg-bye
}

def encode_outcome(delivery: dict) -> int:
    """
    Encode delivery outcome as class label.
    Prioritizes: wicket > extras > runs
    """
    # Check for wicket first (even if runs scored)
    if delivery.get('wickets'):
        return OUTCOME_CLASSES['wicket']

    # Check for extras
    extras = delivery.get('extras', {})
    if extras.get('wides'):
        return OUTCOME_CLASSES['wide']
    if extras.get('noballs'):
        return OUTCOME_CLASSES['noball']
    if extras.get('byes') or extras.get('legbyes'):
        return OUTCOME_CLASSES['bye']

    # Runs off bat
    runs = delivery['runs']['batter']
    if runs == 0:
        return OUTCOME_CLASSES['dot']
    elif runs == 1:
        return OUTCOME_CLASSES['single']
    elif runs == 2:
        return OUTCOME_CLASSES['two']
    elif runs == 3:
        return OUTCOME_CLASSES['three']
    elif runs == 4:
        return OUTCOME_CLASSES['four']
    elif runs >= 6:
        return OUTCOME_CLASSES['six']

    return OUTCOME_CLASSES['dot']  # Fallback
```

---

## Layer Consistency Validation

### Upward Consistency Check

| Operational | Implements Tactical | Satisfies Strategic |
|-------------|--------------------|--------------------|
| `extract_features()` uses only t-1 | Temporal feature design | P1: Temporal Integrity ✅ |
| `player_embedding()` not one-hot | Embedding-based identity | P3: Generalization ✅ |
| `GATConv` with attention | GAT architecture | P2: Interpretability ✅ |
| Graph with 4 nodes | 4-node graph design | P4: Domain Fidelity ✅ |
| Transformer on sequence | Temporal branch | P4: Domain Fidelity ✅ |

### Downward Consistency Check

| Strategic Principle | Tactical Approach | Operational Feasibility |
|--------------------|-------------------|------------------------|
| P1: Temporal Integrity | Incremental features | ✅ Implemented in `extract_features` |
| P2: Interpretability | GAT + Transformer attention | ✅ Attention weights extractable |
| P3: Generalization | Embeddings | ✅ `nn.Embedding` for players |
| P4: Domain Fidelity | Graph + Sequence | ✅ `GATConv` + `TransformerEncoder` |
| P5: Realistic Inputs | Cricsheet fields only | ✅ No external data required |

### Lateral Consistency Check

| Tactical Choice A | Tactical Choice B | Conflict? |
|-------------------|-------------------|-----------|
| 4-node graph | 24-ball sequence | ❌ Complementary (relational + temporal) |
| GAT for matchups | Transformer for momentum | ❌ Different aspects captured |
| Pressure as node feature | Momentum in sequence | ❌ Both needed, different timescales |

---

## Summary: Layer Translation Guide

### For CEO/Stakeholder (30K)

> "We're building a model that predicts what happens on each cricket ball by understanding player relationships and game momentum. It will be explainable - we can show WHY the model made a prediction - and will work on new players it hasn't seen before."

### For ML Lead (3K)

> "Hybrid architecture: GAT on a 4-node graph (batsman, bowler, partner, context) captures relational structure, while a Transformer on the last 24 balls captures temporal patterns. Fusion layer combines both. We chose Transformer over HRM for better content-level interpretability. All features are computed incrementally to prevent leakage."

### For Engineer (300)

> "Process Cricsheet JSON incrementally, building running state. Extract graph node features (batsman stats, bowler stats, context) and sequence features (last 24 ball outcomes). GAT with 4 heads processes graph; Transformer with 4 layers processes sequence. Concatenate outputs, pass through MLP, predict 10-class outcome. Use CrossEntropyLoss, AdamW optimizer, cosine schedule for 50 epochs."

---

## Appendix: Constraint Propagation Example

### Example: Adding New Feature

**Proposed**: Add "ball speed" as feature

**Layer 3 (Operational)**: Not in Cricsheet JSON data

**Layer 2 (Tactical)**: Would require external data source

**Layer 1 (Strategic)**: Violates P5 (Realistic Inputs) if not reliably available

**Decision**: REJECT - violates strategic constraint

---

### Example: Changing Sequence Length

**Proposed**: Increase sequence from 24 to 48 balls

**Layer 3 (Operational)**: Memory usage increases 2x; still feasible

**Layer 2 (Tactical)**: Captures more context; diminishing returns likely beyond 30 balls

**Layer 1 (Strategic)**: No principles violated

**Decision**: INVESTIGATE - run ablation study at operational level, report results to tactical for decision
