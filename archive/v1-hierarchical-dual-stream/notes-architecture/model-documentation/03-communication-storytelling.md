# Cricket Prediction Model: Communication Guide for LLM Commentary

## Headline

**The cricket prediction model thinks like a seasoned commentator: it considers the venue and teams, reads the match situation, evaluates the batter-bowler matchup, and feels the momentum—all before each ball.**

---

## The Story: How the Model "Thinks"

### Act 1: Setting the Stage (Global Context)

Before any ball is bowled, the model asks: "Where are we, and who's playing?"

**What it sees:**
- **Venue embedding:** Is this MCG (pace-friendly, high-scoring) or Chennai (spin-friendly, slower)?
- **Team embeddings:** India's aggressive batting approach vs Australia's pace attack

**Attention reveals:** Which matters more right now?
- If venue attention is 0.4 but team attention totals 0.6 → "The teams are the story today"
- If venue dominates → "This pitch is playing a huge role"

**Commentary insight:** *"At the MCG, we know boundaries flow freely, but today the battle is really between India's star-studded lineup and Australia's fearsome pace battery."*

---

### Act 2: Reading the Match (State Layer)

Now the model asks: "What's the situation?"

**What it sees:**
- Score: 52/2 in 7.3 overs (chasing 156)
- Phase: Middle overs (not powerplay, not death)
- Time pressure: 75 balls remaining, RRR 8.3
- Wicket buffer: 2 down, 8 in hand

**Attention reveals:** What's driving decisions?
- High chase_state attention → "The equation is the story"
- High wicket_buffer attention → "Those two early wickets still loom large"

**Commentary insight:** *"India need 104 from 75 balls—a very gettable target, but at over 8 an over, they can't afford another quiet period. Those two early wickets mean Rohit needs to take calculated risks."*

---

### Act 3: The Protagonists (Actor Layer)

This is where the model gets personal: "Who's facing whom?"

**What it sees:**
- Striker (Rohit): 24 off 18 balls (SR 133), 3 boundaries, "set" indicator high
- Bowler (Bumrah): 2-0-12-1 this spell, economy 6.0, threat level high
- Partnership: 28 off 22 balls, stable

**GAT attention reveals:** The matchup edge
- High attention on striker↔bowler edge → "This duel is crucial"
- High attention on partnership node → "The partnership stability matters"

**Commentary insight:** *"Rohit is set and striking at 133, but he's facing Bumrah who's been exceptional today—just 6 an over with a wicket. This is the contest within the contest. The partnership with Kohli has steadied things, but against Bumrah, every ball is an event."*

---

### Act 4: The Pulse (Dynamics Layer)

Finally, the model feels the rhythm: "What's the momentum?"

**What it sees:**
- Batting momentum: +0.3 (slightly positive, recent scoring)
- Bowling momentum: -0.3 (inverse)
- Pressure index: 0.45 (moderate—wickets lost + RRR pressure)
- Dot pressure: 2 consecutive dots, 8 balls since last boundary

**Attention reveals:** What's creating tension?
- High dot_pressure attention → "The dots are building pressure"

**Commentary insight:** *"Two dot balls to start this over from Bumrah. It's been 8 balls since a boundary. You can feel the pressure building—Rohit will be looking to break free, but Bumrah senses the opportunity."*

---

### The Prediction: What Happens Next?

The model combines all four layers and produces:

```
Probabilities:
  Dot ball:  35%  ← Most likely given Bumrah's control
  Single:    28%  ← Rohit might rotate strike
  Four:      12%  ← Set batter, but tough bowler
  Two:       10%  ← Running hard possible
  Six:        5%  ← High risk against Bumrah
  Wicket:     5%  ← Pressure situation, but set batter
  Three:      5%  ← Least likely
```

**Commentary prediction:** *"Given Bumrah's control and those two dots, I'd expect either another dot or Rohit trying to work a single to ease the pressure. There's maybe a 1 in 8 chance he goes big for a boundary here—the pressure might force his hand. But equally, that pressure creates a wicket chance for Australia."*

---

## How to Extract Commentary Signals

### 1. Run Inference with Attention

```python
output = model.forward(batch, return_attention=True)

# Extract components
probs = output["probs"]                    # Outcome probabilities
gat_attn = output["gat_attention"]         # Hierarchical attention
temporal_attn = output["temporal_attention"]  # History attention
```

### 2. Interpret Layer Importance

```python
layer_weights = gat_attn["layer_importance"]
# {"global": 0.22, "match_state": 0.28, "actor": 0.30, "dynamics": 0.20}

dominant_layer = max(layer_weights, key=layer_weights.get)
# "actor" → The batter-bowler matchup is most important right now
```

**Commentary mapping:**
| Dominant Layer | Commentary Focus |
|----------------|------------------|
| global | Venue conditions, team matchup |
| match_state | Score pressure, chase equation |
| actor | Individual form, matchup dynamics |
| dynamics | Momentum, immediate pressure |

### 3. Interpret Global Attention

```python
global_attn = gat_attn["global"]
# {"venue": 0.15, "batting_team": 0.42, "bowling_team": 0.43}

if global_attn["venue"] > 0.35:
    # "The pitch is playing a big role"
if global_attn["batting_team"] + global_attn["bowling_team"] > 0.8:
    # "This is all about the team battle"
```

### 4. Interpret Temporal Attention

```python
head_attn = temporal_attn["head_attention"]

# Recency head (Head 0)
recent_focus = sum(head_attn["recency"][-6:])  # Last over attention
if recent_focus > 0.7:
    # "The last over is dominating the model's thinking"

# Same-bowler head (Head 1)
bowler_coherence = sum(head_attn["same_bowler"])
# High value → "The bowler's spell pattern is significant"

# Same-batsman head (Head 2)
batsman_form = sum(head_attn["same_batsman"])
# High value → "The batter's recent form is key"
```

### 5. Build Commentary Template

```python
def generate_commentary_context(output, batch):
    probs = output["probs"]
    attn = output["gat_attention"]

    context = {
        "top_outcome": OUTCOMES[probs.argmax()],
        "top_prob": probs.max().item(),
        "dominant_layer": get_dominant_layer(attn["layer_importance"]),
        "venue_influence": attn["global"]["venue"],
        "matchup_importance": get_matchup_attention(attn["actor"]),
        "recent_pressure": get_dot_pressure(batch),
        "momentum_direction": "batting" if batch["momentum"] > 0 else "bowling",
    }

    return context

# LLM prompt template
COMMENTARY_PROMPT = """
Based on the cricket prediction model's analysis:

Current situation:
- Score: {score}/{wickets} chasing {target}
- Batter: {batter} (SR: {strike_rate})
- Bowler: {bowler} (Econ: {economy})

Model signals:
- Most likely outcome: {top_outcome} ({top_prob:.0%})
- Key factor: {dominant_layer}
- Venue influence: {venue_influence:.0%}
- Matchup tension: {matchup_importance:.0%}
- Dot pressure: {recent_pressure} consecutive dots
- Momentum: {momentum_direction}

Generate a 2-sentence commentary prediction for the next ball.
"""
```

---

## Narrative Templates by Situation

### High Chase Pressure (RRR > 10)

**Model signals:** High chase_state attention, high pressure_index
**Template:** *"The equation demands {RRR} an over. With {wickets_remaining} wickets in hand, {batter} knows a boundary here could change everything—but {bowler} is bowling for a breakthrough. The model gives {boundary_prob}% for a boundary, {wicket_prob}% for a wicket."*

### Set Batter vs Quality Bowler

**Model signals:** High actor layer importance, high matchup edge attention
**Template:** *"This is the duel everyone's watching. {batter} is set on {runs} off {balls}, but {bowler} has {wickets}-{runs_conceded} today. The model sees a {dot_prob}% chance of a dot—{bowler}'s control—but a {boundary_prob}% chance {batter} breaks free."*

### Momentum Shift Building

**Model signals:** High dynamics attention, consecutive dots
**Template:** *"It's been {dots} balls without a run. The pressure is palpable. {batter} needs to find a way to score, but that desperation is exactly what {bowler} is banking on. Dot probability: {dot_prob}%, but wicket chance has crept up to {wicket_prob}%."*

### Death Overs Explosion

**Model signals:** High time_pressure attention, phase = death
**Template:** *"Death overs cricket. Everything changes now. {batter} will be looking to maximize every ball, {bowler} searching for those crucial yorkers. Boundary probability spikes to {boundary_prob}%—this is when games are won and lost."*

---

## Key Metrics for LLM Integration

| Metric | What to Extract | Commentary Use |
|--------|-----------------|----------------|
| `probs[0]` | Dot ball probability | "Control vs attack" narrative |
| `probs[4] + probs[5]` | Boundary probability | "Big shot potential" |
| `probs[6]` | Wicket probability | "Danger level" |
| `layer_importance["actor"]` | Matchup focus | Individual duel narrative |
| `layer_importance["dynamics"]` | Momentum focus | Pressure/momentum narrative |
| `temporal_attn["recency"][-6:]` | Last over focus | "Recent events matter" |
| `same_bowler_strength` | Spell coherence | Bowler's day narrative |

---

## Summary

The cricket prediction model provides rich intermediate signals that map directly to cricket commentary:

1. **Layer importance** → What's driving the game right now
2. **Global attention** → Venue vs team narratives
3. **Actor attention (GAT)** → Matchup and partnership stories
4. **Dynamics attention** → Momentum and pressure feel
5. **Temporal attention** → Recent form and spell patterns
6. **Output probabilities** → Likelihood of different outcomes

An LLM can use these signals to generate contextual, data-driven commentary that sounds like an expert analyst who's been watching every ball.
