# Temporal Attention: Cross-Ball Patterns

## Design Goal

Beyond within-ball attention, we need to capture **which past balls matter** for predicting the current outcome. This enables insights like:

> "The model attended to balls 43, 37, 31 - all previous deliveries from this bowler. The pattern shows the batsman has hit boundaries on 2 of those 3 balls, informing the boundary prediction."

## Transformer for Temporal Patterns

We use a Transformer encoder over the last N balls because:
1. **Attention is interpretable**: Can visualize which balls attend to which
2. **No recurrence**: All positions processed in parallel
3. **Multi-head specialization**: Different heads learn different patterns

### What Attention Heads Should Learn

| Head | Expected Pattern | Interpretation |
|------|------------------|----------------|
| **Head 1** | Same-bowler attention | "What happened when this bowler bowled before?" |
| **Head 2** | Same-batsman attention | "How has this batsman been performing?" |
| **Head 3** | Recent-ball attention | "What's the immediate momentum?" |
| **Head 4** | Over-boundary attention | "What happened at over changes?" |
| **Head 5** | Wicket-adjacent attention | "What led to/followed wickets?" |
| **Head 6** | Boundary-cluster attention | "Where were the boundary clusters?" |

## Architecture

### Sequence Encoding

```python
class TemporalEncoder(nn.Module):
    """
    Encodes the last N balls as a sequence for Transformer attention.
    Each ball is represented by its hierarchical graph representation.
    """

    def __init__(self, ball_dim, hidden_dim, num_heads=6, num_layers=2, max_seq_len=36):
        super().__init__()

        # Project ball representation to hidden dim
        self.ball_projection = nn.Linear(ball_dim, hidden_dim)

        # Positional encoding: captures ball position in sequence
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Additional positional features for cricket-specific structure
        self.over_embedding = nn.Embedding(20, hidden_dim // 4)  # Which over
        self.ball_in_over_embedding = nn.Embedding(6, hidden_dim // 4)  # Ball 1-6

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, ball_sequence, positions, overs, balls_in_over, mask=None):
        """
        Args:
            ball_sequence: (batch, seq_len, ball_dim) - hierarchical ball representations
            positions: (batch, seq_len) - sequence positions
            overs: (batch, seq_len) - over number for each ball
            balls_in_over: (batch, seq_len) - ball position within over
            mask: (batch, seq_len) - padding mask
        """
        # Project to hidden dim
        h = self.ball_projection(ball_sequence)

        # Add positional information
        h = h + self.position_embedding(positions)
        h = h + self.over_embedding(overs)
        h = h + self.ball_in_over_embedding(balls_in_over)

        # Transformer forward (with custom hook to capture attention)
        h = self.transformer(h, src_key_padding_mask=mask)

        # Use last position as current ball representation
        h_temporal = h[:, -1, :]

        return h_temporal
```

### Attention Head Specialization

We can encourage head specialization through architectural choices:

```python
class SpecializedTemporalAttention(nn.Module):
    """
    Transformer with explicitly specialized attention heads.
    Each head has a bias toward certain attention patterns.
    """

    def __init__(self, hidden_dim, num_heads=6):
        super().__init__()

        # Shared projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Head-specific bias terms
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        # Learned bias for different patterns
        # These can be initialized to encourage specific attention patterns
        self.recency_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.same_actor_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x, same_bowler_mask, same_batsman_mask):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            same_bowler_mask: (batch, seq_len, seq_len) - 1 where bowler matches
            same_batsman_mask: (batch, seq_len, seq_len) - 1 where batsman matches
        """
        batch, seq_len, _ = x.shape

        # Standard QKV
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)

        # Add pattern-specific biases per head
        # Head 0: Recency bias (attend to recent balls)
        recency_positions = torch.arange(seq_len, device=x.device).float()
        recency_bias = -0.1 * (seq_len - 1 - recency_positions)  # Recent = higher
        scores[:, 0, :, :] += recency_bias.unsqueeze(0).unsqueeze(0)

        # Head 1: Same-bowler bias
        scores[:, 1, :, :] += same_bowler_mask * 2.0  # Boost same-bowler attention

        # Head 2: Same-batsman bias
        scores[:, 2, :, :] += same_batsman_mask * 2.0  # Boost same-batsman attention

        # Heads 3-5: Learn freely

        # Causal mask (can't attend to future)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

        # Softmax and apply
        attn_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attn_weights  # Store for interpretability

        output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        output = output.reshape(batch, seq_len, -1)

        return output, attn_weights
```

## Multi-Scale Temporal Attention

Cricket has patterns at multiple time scales:

| Scale | Balls | Captures |
|-------|-------|----------|
| **Micro** | Last 6 | Current over dynamics |
| **Short** | Last 12-18 | Recent momentum |
| **Medium** | Last 24-36 | Extended patterns, bowler rotations |
| **Long** | Full innings | Innings trajectory (too long for full attention) |

### Hierarchical Temporal Encoding

```python
class MultiScaleTemporalEncoder(nn.Module):
    """
    Process ball sequence at multiple temporal scales.
    """

    def __init__(self, ball_dim, hidden_dim):
        super().__init__()

        # Micro-scale: current over (6 balls)
        self.micro_encoder = TemporalEncoder(ball_dim, hidden_dim, num_heads=2, max_seq_len=6)

        # Short-scale: recent balls (12 balls)
        self.short_encoder = TemporalEncoder(ball_dim, hidden_dim, num_heads=3, max_seq_len=12)

        # Medium-scale: extended context (36 balls)
        self.medium_encoder = TemporalEncoder(ball_dim, hidden_dim, num_heads=4, max_seq_len=36)

        # Long-scale: over summaries (20 overs max)
        self.over_summarizer = nn.GRU(ball_dim, hidden_dim, batch_first=True)
        self.long_encoder = TemporalEncoder(hidden_dim, hidden_dim, num_heads=2, max_seq_len=20)

        # Fusion
        self.scale_fusion = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, full_sequence, current_over_start, over_summaries):
        """
        Args:
            full_sequence: (batch, seq_len, ball_dim) - all balls so far
            current_over_start: index where current over started
            over_summaries: (batch, num_overs, ball_dim) - per-over summary
        """
        batch = full_sequence.size(0)

        # Micro: last 6 balls (current over)
        micro_seq = full_sequence[:, -6:, :]
        h_micro = self.micro_encoder(micro_seq)

        # Short: last 12 balls
        short_seq = full_sequence[:, -12:, :]
        h_short = self.short_encoder(short_seq)

        # Medium: last 36 balls
        medium_seq = full_sequence[:, -36:, :]
        h_medium = self.medium_encoder(medium_seq)

        # Long: over-level patterns
        h_long = self.long_encoder(over_summaries)

        # Fuse all scales
        h_temporal = self.scale_fusion(torch.cat([h_micro, h_short, h_medium, h_long], dim=-1))

        return h_temporal, {
            'micro': self.micro_encoder.attention_weights,
            'short': self.short_encoder.attention_weights,
            'medium': self.medium_encoder.attention_weights,
            'long': self.long_encoder.attention_weights,
        }
```

## Temporal Attention Visualization

### Attention Heatmap Format

```json
{
  "temporal_attention": {
    "scale": "medium",
    "current_ball": 47,
    "sequence_range": [12, 47],

    "attention_by_head": {
      "head_0_recency": {
        "pattern": "recency",
        "top_balls": [46, 45, 44, 43],
        "weights": [0.25, 0.18, 0.12, 0.08]
      },
      "head_1_same_bowler": {
        "pattern": "same_bowler",
        "bowler": "J Bumrah",
        "top_balls": [43, 37, 31, 25],
        "weights": [0.22, 0.18, 0.15, 0.10],
        "outcomes": ["4", "1", "4", "0"]
      },
      "head_2_same_batsman": {
        "pattern": "same_batsman",
        "batsman": "V Kohli",
        "top_balls": [46, 44, 42, 40],
        "weights": [0.20, 0.15, 0.12, 0.10],
        "outcomes": ["1", "2", "0", "4"]
      },
      "head_3_boundary_context": {
        "pattern": "learned",
        "top_balls": [42, 38, 35],
        "weights": [0.15, 0.12, 0.10],
        "note": "all were boundaries"
      }
    },

    "aggregate_attention": {
      "last_over": 0.35,
      "same_bowler": 0.25,
      "same_batsman": 0.20,
      "boundaries": 0.12,
      "other": 0.08
    }
  }
}
```

### LLM-Ready Summary

For an LLM to consume:

```python
def summarize_temporal_attention(attention_data, ball_history):
    """
    Generate human-readable summary of temporal attention.
    """
    summary = []

    # Recency
    if attention_data['aggregate']['last_over'] > 0.3:
        summary.append(f"Strong focus on current over ({attention_data['aggregate']['last_over']:.0%})")

    # Same-bowler pattern
    bowler_attn = attention_data['head_1_same_bowler']
    if sum(bowler_attn['weights']) > 0.4:
        outcomes = bowler_attn['outcomes']
        boundaries = outcomes.count('4') + outcomes.count('6')
        summary.append(
            f"Attended to {bowler_attn['bowler']}'s previous balls: "
            f"{boundaries}/{len(outcomes)} were boundaries"
        )

    # Momentum
    recent_outcomes = [ball_history[i]['outcome'] for i in attention_data['head_0_recency']['top_balls']]
    recent_runs = sum(int(o) if o.isdigit() else 0 for o in recent_outcomes)
    summary.append(f"Recent momentum: {recent_runs} runs in last 4 attended balls")

    return " | ".join(summary)
```

**Example output**:
> "Strong focus on current over (35%) | Attended to J Bumrah's previous balls: 2/4 were boundaries | Recent momentum: 7 runs in last 4 attended balls"

## Integration with Hierarchical Graph Attention

The temporal attention output is combined with the hierarchical graph output:

```python
class FullModel(nn.Module):
    def forward(self, batch):
        # 1. Hierarchical within-ball attention
        h_ball = self.hierarchical_attention(batch['graph_data'])

        # 2. Temporal cross-ball attention
        h_temporal = self.temporal_attention(batch['sequence'])

        # 3. Fusion
        h_combined = self.fusion(h_ball, h_temporal)

        # 4. Output
        logits = self.output_head(h_combined)

        return logits

    def get_full_attention_profile(self, batch):
        """
        Extract complete attention profile for LLM consumption.
        """
        return {
            'within_ball': self.hierarchical_attention.get_attention_weights(),
            'across_balls': self.temporal_attention.get_attention_weights(),
            'fusion_weights': self.fusion.get_weights(),
        }
```

## Next: Derived vs Learned Features

See [05-derived-vs-learned.md](./05-derived-vs-learned.md) for analysis of which features to compute explicitly vs let attention learn.
