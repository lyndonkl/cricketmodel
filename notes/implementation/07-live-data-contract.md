# Live Data Contract

## Purpose

This document specifies the exact input/output contract for live match predictions. The goal is minimal, practical input that's available from any scorecard feed.

## Input: Per-Ball Prediction Request

### Required Fields (7 values)

```python
@dataclass
class LiveBallInput:
    """Minimal input required for each ball prediction."""

    striker_id: str          # Player identifier (e.g., "virat_kohli" or UUID)
    non_striker_id: str      # Partner at other end
    bowler_id: str           # Current bowler

    score: int               # Current innings total runs
    wickets: int             # Current innings wickets fallen (0-9)
    over_ball: float         # Current over.ball (e.g., 15.3 = over 16, ball 4)

    target: Optional[int]    # Chase target (None for 1st innings)

    # Optional: info for unknown players/venues (for cold-start handling)
    player_info: Optional[Dict[str, PlayerInfo]] = None
    venue_info: Optional[VenueInfo] = None
```

### Optional: Cold-Start Info

For unknown players or venues, provide additional info to generate embeddings:

```python
@dataclass
class PlayerInfo:
    """Optional info for unknown players. See 08-cold-start-embeddings.md."""

    # Role-based fallback
    role: Optional[str] = None           # "opener_aggressive", "death_pace", etc.
    batting_position: Optional[int] = None  # 1-11

    # Stats-based generation (preferred if available)
    career_strike_rate: Optional[float] = None
    career_average: Optional[float] = None
    career_economy: Optional[float] = None    # For bowlers
    matches_played: Optional[int] = None

    # Type indicators
    is_batsman: bool = True
    is_pace: Optional[bool] = None
    is_spin: Optional[bool] = None


@dataclass
class VenueInfo:
    """Optional info for unknown venues. See 08-cold-start-embeddings.md."""

    # Location-based fallback
    country: Optional[str] = None
    venue_type: Optional[str] = None  # "high_scoring_flat", "spin_friendly", etc.

    # Stats-based generation (preferred if available)
    avg_first_innings_score: Optional[float] = None
    avg_second_innings_score: Optional[float] = None
    boundary_percentage: Optional[float] = None
```

**Note**: The model generates embeddings from features, not IDs. See [08-cold-start-embeddings.md](./08-cold-start-embeddings.md) for details on how this works.

### Example Request (Known Players)

```json
{
  "striker_id": "virat_kohli",
  "non_striker_id": "kl_rahul",
  "bowler_id": "mitchell_starc",
  "score": 145,
  "wickets": 4,
  "over_ball": 15.3,
  "target": 186
}
```

### Example Request (With Unknown Player)

```json
{
  "striker_id": "new_debutant_xyz",
  "non_striker_id": "virat_kohli",
  "bowler_id": "mitchell_starc",
  "score": 45,
  "wickets": 2,
  "over_ball": 6.3,
  "target": null,

  "player_info": {
    "new_debutant_xyz": {
      "role": "middle_order_aggressor",
      "batting_position": 5,
      "career_strike_rate": 142.5,
      "career_average": 28.3,
      "matches_played": 15,
      "is_batsman": true
    }
  }
}
```

### Example Request (Unknown Venue)

```json
{
  "striker_id": "virat_kohli",
  "non_striker_id": "kl_rahul",
  "bowler_id": "mitchell_starc",
  "score": 45,
  "wickets": 1,
  "over_ball": 8.2,
  "target": null,

  "venue_info": {
    "country": "United States",
    "venue_type": "high_scoring_flat",
    "avg_first_innings_score": 175.0
  }
}
```

### Match Context (once per match)

Provided at match start, not per-ball:

```python
@dataclass
class MatchContext:
    """Context provided once at match initialization."""

    match_id: str
    venue: str                    # Venue name or ID
    batting_team: str             # Team currently batting
    bowling_team: str             # Team currently bowling
    innings: int                  # 1 or 2
    toss_winner: Optional[str]    # Which team won toss
    toss_decision: Optional[str]  # "bat" or "field"
```

## State: Maintained by Model

The model maintains running state from accumulated ball outcomes:

```python
@dataclass
class ModelState:
    """Internal state maintained across balls."""

    # Ball-by-ball sequence
    ball_sequence: List[BallOutcome]  # All balls this innings

    # Per-batsman stats (computed from sequence)
    batsman_innings: Dict[str, BatsmanInnings]

    # Per-bowler stats (computed from sequence)
    bowler_spells: Dict[str, BowlerSpell]

    # Partnership tracking
    current_partnership: Partnership
    last_wicket_ball_index: int

    # Derived features (recomputed each ball)
    pressure_index: float
    momentum_score: float
    consecutive_dots: int


@dataclass
class BatsmanInnings:
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    dots: int = 0


@dataclass
class BowlerSpell:
    balls: int = 0
    runs: int = 0
    wickets: int = 0
    dots: int = 0
    extras: int = 0


@dataclass
class Partnership:
    runs: int = 0
    balls: int = 0
    striker_runs: int = 0
    partner_runs: int = 0
```

## Output: Prediction Response

```python
@dataclass
class PredictionResponse:
    """Model output for each ball."""

    # Outcome probabilities
    outcome_probs: Dict[str, float]  # {"dot": 0.22, "single": 0.28, ...}
    predicted_outcome: str            # Most likely outcome
    confidence: float                 # Probability of predicted outcome

    # Attention weights for interpretability
    attention: AttentionProfile

    # Key derived features (for display)
    key_features: KeyFeatures


@dataclass
class KeyFeatures:
    """Key computed features for display."""
    pressure_index: float
    required_run_rate: Optional[float]
    current_run_rate: float
    batsman_strike_rate: float
    batsman_balls: int
    partnership_runs: int
    consecutive_dots: int
```

### Example Response

```json
{
  "outcome_probs": {
    "dot": 0.22,
    "single": 0.28,
    "two": 0.08,
    "three": 0.02,
    "boundary": 0.25,
    "six": 0.10,
    "wicket": 0.05
  },
  "predicted_outcome": "single",
  "confidence": 0.28,

  "attention": {
    "layer_importance": {
      "global": 0.12,
      "match_state": 0.38,
      "actor": 0.28,
      "dynamics": 0.22
    },
    "top_factors": [
      {"node": "chase_state", "weight": 0.42},
      {"node": "pressure_index", "weight": 0.38},
      {"node": "batting_momentum", "weight": 0.32}
    ],
    "temporal": {
      "same_bowler_balls": [43, 37, 31],
      "recent_balls": [46, 45, 44]
    }
  },

  "key_features": {
    "pressure_index": 0.72,
    "required_run_rate": 9.8,
    "current_run_rate": 7.6,
    "batsman_strike_rate": 118.75,
    "batsman_balls": 32,
    "partnership_runs": 42,
    "consecutive_dots": 0
  }
}
```

## State Update: After Ball Completion

When a ball outcome is known, update state:

```python
@dataclass
class BallOutcome:
    """Outcome of a completed ball."""
    runs_batter: int          # Runs off bat
    runs_extras: int          # Extra runs
    runs_total: int           # Total runs
    is_wicket: bool           # Dismissal occurred
    wicket_kind: Optional[str]  # "caught", "bowled", etc.
    extras_type: Optional[str]  # "wide", "noball", etc.

    # For state tracking
    striker_id: str
    bowler_id: str
    over_ball: float
```

### State Update Flow

```python
def update_state(state: ModelState, outcome: BallOutcome):
    """Update model state after ball completion."""

    # 1. Append to sequence
    state.ball_sequence.append(outcome)

    # 2. Update batsman stats
    batsman = state.batsman_innings[outcome.striker_id]
    batsman.runs += outcome.runs_batter
    batsman.balls += 1
    if outcome.runs_batter == 4:
        batsman.fours += 1
    elif outcome.runs_batter == 6:
        batsman.sixes += 1
    elif outcome.runs_total == 0:
        batsman.dots += 1

    # 3. Update bowler stats
    bowler = state.bowler_spells[outcome.bowler_id]
    bowler.balls += 1
    bowler.runs += outcome.runs_total
    if outcome.is_wicket:
        bowler.wickets += 1
    if outcome.runs_total == 0 and not outcome.extras_type:
        bowler.dots += 1

    # 4. Update partnership
    state.current_partnership.runs += outcome.runs_total
    state.current_partnership.balls += 1
    state.current_partnership.striker_runs += outcome.runs_batter

    # 5. Handle wicket - reset partnership
    if outcome.is_wicket:
        state.current_partnership = Partnership()
        state.last_wicket_ball_index = len(state.ball_sequence)

    # 6. Recompute derived features
    state.pressure_index = compute_pressure_index(state)
    state.momentum_score = compute_momentum(state.ball_sequence[-12:])
    state.consecutive_dots = count_consecutive_dots(state.ball_sequence)
```

## API Endpoints

### Initialize Match

```
POST /api/match/init

Request:
{
  "match_id": "ind_aus_2024_01",
  "venue": "melbourne_cricket_ground",
  "batting_team": "india",
  "bowling_team": "australia",
  "innings": 2,
  "target": 186
}

Response:
{
  "status": "initialized",
  "match_id": "ind_aus_2024_01"
}
```

### Predict Next Ball

```
POST /api/predict

Request:
{
  "match_id": "ind_aus_2024_01",
  "striker_id": "virat_kohli",
  "non_striker_id": "kl_rahul",
  "bowler_id": "mitchell_starc",
  "score": 145,
  "wickets": 4,
  "over_ball": 15.3
}

Response:
{
  "outcome_probs": {...},
  "predicted_outcome": "single",
  "confidence": 0.28,
  "attention": {...},
  "key_features": {...}
}
```

### Update After Ball

```
POST /api/update

Request:
{
  "match_id": "ind_aus_2024_01",
  "outcome": {
    "runs_batter": 1,
    "runs_extras": 0,
    "runs_total": 1,
    "is_wicket": false,
    "striker_id": "virat_kohli",
    "bowler_id": "mitchell_starc",
    "over_ball": 15.3
  }
}

Response:
{
  "status": "updated",
  "balls_processed": 94
}
```

### Get Insight

```
GET /api/insight/{match_id}/{ball_number}

Response:
{
  "insight": "The model sees chase pressure building...",
  "attention_summary": "Focus on chase (42%) and pressure (38%)",
  "key_factors": ["required_run_rate: 9.8", "pressure_index: 0.72"]
}
```

## Data Sources

| Data | Source | Frequency |
|------|--------|-----------|
| Live ball input | Scorecard API (Cricbuzz, ESPNCricinfo) | Per ball |
| Ball outcome | Scorecard API | After each ball |
| Player IDs | Player registry (pre-built) | Match start |
| Career stats | ESPNCricinfo API | Match start (for cold-start) |
| Venue info | Pre-loaded database | Match start |

## Validation Rules

```python
def validate_input(input: LiveBallInput) -> bool:
    assert 0 <= input.wickets <= 9
    assert 0 <= input.score <= 300  # Reasonable T20 max
    assert 0 <= input.over_ball < 20.0
    assert input.over_ball == int(input.over_ball) + (input.over_ball % 1) * 10 / 10
    # over_ball should be X.Y where Y is 0-5
    ball_in_over = int((input.over_ball % 1) * 10)
    assert 0 <= ball_in_over <= 5
    if input.target is not None:
        assert input.target > 0
    return True
```
