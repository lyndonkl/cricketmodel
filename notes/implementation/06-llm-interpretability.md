# LLM-Consumable Interpretability

## Design Goal

The cricket prediction model's attention patterns should be structured so an LLM can:
1. Observe the attention weights in real-time
2. Understand what the model focused on
3. Generate natural language insights for viewers

## Attention Data Format

### Full Attention Profile (per ball)

```json
{
  "ball_id": "match_12345_ball_47",
  "match_context": {
    "teams": ["India", "Australia"],
    "venue": "Melbourne Cricket Ground",
    "innings": 2,
    "current_score": "145/4",
    "target": 186,
    "overs": "15.5"
  },

  "prediction": {
    "outcome_distribution": {
      "dot": 0.22,
      "single": 0.28,
      "two": 0.08,
      "boundary": 0.25,
      "six": 0.10,
      "wicket": 0.05,
      "extras": 0.02
    },
    "predicted_outcome": "single",
    "confidence": 0.28
  },

  "hierarchical_attention": {
    "layer_importance": {
      "global_context": 0.12,
      "match_state": 0.38,
      "actor": 0.28,
      "dynamics": 0.22
    },

    "global_context": {
      "venue": 0.45,
      "team_context": 0.35,
      "match_importance": 0.20
    },

    "match_state": {
      "score_state": 0.12,
      "chase_state": 0.42,
      "phase_state": 0.18,
      "time_pressure": 0.18,
      "wicket_buffer": 0.10
    },

    "actor": {
      "batsman_identity": 0.30,
      "batsman_state": 0.22,
      "bowler_identity": 0.18,
      "bowler_state": 0.12,
      "partnership": 0.18
    },

    "dynamics": {
      "batting_momentum": 0.32,
      "bowling_momentum": 0.15,
      "pressure_index": 0.38,
      "dot_ball_pressure": 0.15
    }
  },

  "temporal_attention": {
    "sequence_length": 24,
    "attention_by_head": {
      "recency": {
        "focus": "recent_balls",
        "top_balls": [46, 45, 44, 43],
        "weights": [0.22, 0.18, 0.14, 0.10]
      },
      "same_bowler": {
        "focus": "bowler_pattern",
        "bowler": "Mitchell Starc",
        "balls": [43, 37, 31],
        "outcomes": ["4", "1", "0"],
        "weights": [0.28, 0.22, 0.15]
      },
      "same_batsman": {
        "focus": "batsman_form",
        "batsman": "Virat Kohli",
        "balls": [46, 44, 42, 40],
        "outcomes": ["1", "2", "1", "4"],
        "weights": [0.20, 0.16, 0.12, 0.10]
      },
      "boundary_context": {
        "focus": "boundaries",
        "boundary_balls": [42, 38, 35, 28],
        "weights": [0.18, 0.14, 0.12, 0.08]
      }
    }
  },

  "key_features": {
    "pressure_index": 0.72,
    "required_run_rate": 9.8,
    "current_run_rate": 7.6,
    "batsman_setness": 0.85,
    "consecutive_dots": 0,
    "partnership_runs": 42,
    "batsman_runs": 38,
    "batsman_balls": 32
  }
}
```

## LLM Insight Generation

### System Prompt for Insight LLM

```
You are a cricket analyst AI that generates insights from model attention patterns.

Given: Model attention weights, prediction, and match context.
Task: Generate 2-3 sentences explaining why the model made this prediction.

Guidelines:
1. Reference specific attention weights (e.g., "38% focus on chase state")
2. Connect attention to cricket domain knowledge
3. Mention temporal patterns if relevant (same-bowler, recent form)
4. Keep language accessible for TV/streaming audience
5. Don't mention technical terms like "attention weights" - describe what the model "focused on" or "considered"

Format: Short, punchy insights suitable for on-screen graphics.
```

### Example Generations

**Input**: Attention profile above

**Generated Insight**:
> "The model predicts a single with 28% confidence. Focus is heavily on the chase equation - India need 41 from 25 at nearly 10 runs per over, putting significant pressure on Kohli. The model looked at Starc's previous deliveries to Kohli this innings - a boundary, a single, and a dot - suggesting a pattern of targeting singles rather than risk. Kohli's settled at 38 off 32, reducing aggressive shot probability."

### Insight Templates

For common scenarios, templates ensure consistent quality:

**High Chase Pressure**:
```python
if attention['match_state']['chase_state'] > 0.35 and key_features['required_run_rate'] > 10:
    template = (
        "The chase is critical here - {team} need {runs_required} from {balls_remaining} balls. "
        "Model attention is {chase_attention:.0%} on the chase equation, "
        "with pressure index at {pressure_index:.0%}. "
        "{batsman} has been {batsman_assessment} - expect {shot_prediction}."
    )
```

**Bowler Dominance**:
```python
if temporal_attention['same_bowler']['outcomes'].count('0') >= 2:
    template = (
        "{bowler} has been tough on {batsman} - "
        "the model sees {dot_count} dots in their last {encounters} balls together. "
        "Attention to this matchup is {matchup_attention:.0%}. "
        "The model expects {prediction} with {confidence:.0%} confidence."
    )
```

**Set Batsman Momentum**:
```python
if key_features['batsman_setness'] > 0.8 and attention['dynamics']['batting_momentum'] > 0.3:
    template = (
        "{batsman} is well set on {runs} off {balls}. "
        "Model attention is high on batting momentum ({momentum_attention:.0%}) - "
        "recent balls show {recent_pattern}. "
        "With {rrr:.1f} required and {batsman} in form, expect {prediction}."
    )
```

## Real-Time Dashboard Design

### Attention Visualization Components

**1. Layer Bar Chart**
```
Global     ███░░░░░░░ 12%
Match State ███████░░░ 38%  ◄ Dominant
Actor      █████░░░░░ 28%
Dynamics   ████░░░░░░ 22%
```

**2. Within-Layer Breakdown** (for dominant layer)
```
Match State:
├── Chase State   ████████░░ 42%
├── Phase State   ███░░░░░░░ 18%
├── Time Pressure ███░░░░░░░ 18%
├── Score State   ██░░░░░░░░ 12%
└── Wicket Buffer █░░░░░░░░░ 10%
```

**3. Temporal Attention Map**
```
Ball: 23  24  25  26  27  28  29  30  31  32  33  ...  46  47
      ░░  ░░  ░░  ░░  ░░  ██  ░░  ░░  ██  ░░  ░░  ...  ██  ▶
                         Starc      Starc             Recent
```

**4. Key Numbers**
```
┌────────────────────────────────────────┐
│ Pressure: 72%  │  RRR: 9.8  │  CRR: 7.6 │
│ Setness: 85%   │  Partnership: 42 runs │
└────────────────────────────────────────┘
```

## API for Live Integration

### Attention Endpoint

```python
@app.route('/api/attention/<match_id>/<ball_number>')
def get_attention(match_id, ball_number):
    """
    Returns attention profile for a specific ball.
    """
    profile = model.get_attention_profile(match_id, ball_number)
    return jsonify(profile)


@app.route('/api/insight/<match_id>/<ball_number>')
def get_insight(match_id, ball_number):
    """
    Returns LLM-generated insight for a ball.
    """
    profile = model.get_attention_profile(match_id, ball_number)
    insight = llm.generate_insight(profile)
    return jsonify({
        'insight': insight,
        'attention_summary': summarize_attention(profile),
        'key_factors': extract_key_factors(profile)
    })
```

### Streaming Updates

For live matches:

```python
async def stream_predictions(match_id):
    """
    Stream predictions and attention as balls happen.
    """
    async for ball_event in match_stream(match_id):
        # Get prediction before ball
        prediction = model.predict(ball_event.state)
        attention = model.get_attention_profile(ball_event)

        yield {
            'type': 'pre_ball',
            'ball': ball_event.ball_number,
            'prediction': prediction,
            'attention': attention,
            'insight': llm.generate_insight(attention)
        }

        # After ball, compare to outcome
        actual = await wait_for_outcome(match_id, ball_event.ball_number)

        yield {
            'type': 'post_ball',
            'ball': ball_event.ball_number,
            'predicted': prediction['predicted_outcome'],
            'actual': actual,
            'correct': prediction['predicted_outcome'] == actual
        }
```

## Example: Full Live Commentary Integration

**Before ball 47**:

```
┌─────────────────────────────────────────────────────────────────┐
│ PREDICTION: Single (28%)  |  Boundary: 25%  |  Dot: 22%        │
├─────────────────────────────────────────────────────────────────┤
│ MODEL FOCUS:                                                     │
│ • Chase equation (42%) - 41 needed from 25 at RRR 9.8           │
│ • Pressure (38%) - currently at 72%, high for this stage        │
│ • Kohli's form (32%) - 38 off 32, well set                      │
│ • Starc pattern (28%) - 1 boundary, 1 single, 1 dot in matchup  │
├─────────────────────────────────────────────────────────────────┤
│ INSIGHT: "The model sees a battle between chase pressure and    │
│ Kohli's set innings. Starc has been tough to score off - expect │
│ Kohli to target rotation rather than risk with the RRR at 9.8." │
└─────────────────────────────────────────────────────────────────┘
```

**After ball 47** (outcome: 2 runs):

```
┌─────────────────────────────────────────────────────────────────┐
│ RESULT: 2 runs (predicted: single)                              │
├─────────────────────────────────────────────────────────────────┤
│ MODEL REFLECTION:                                                │
│ Prediction was for rotation (single), actual was aggressive     │
│ running (2). Model underestimated Kohli's running between       │
│ wickets ability. Attention to partnership (18%) may have been   │
│ insufficient - Kohli-Rahane have 42 together with good rotation.│
└─────────────────────────────────────────────────────────────────┘
```

## Summary

| Component | Purpose | Output |
|-----------|---------|--------|
| **Hierarchical Attention** | What the model focused on within this ball | Layer weights, node weights |
| **Temporal Attention** | Which past balls mattered | Ball-level weights, head patterns |
| **Key Features** | Computed metrics driving prediction | Pressure, RRR, setness, etc. |
| **LLM Insight** | Human-readable explanation | 2-3 sentence narrative |
| **Visualization** | Graphical attention display | Bar charts, heatmaps, numbers |

The entire system is designed so an LLM can observe the structured attention data and generate insights that explain the model's reasoning to a cricket audience.
