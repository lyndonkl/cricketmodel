# Cricket Ball Prediction: Systems Thinking & Leverage Analysis

## System Overview

**System Boundary**: A single cricket innings viewed as a dynamic system
**Problem Statement**: Predict ball-by-ball outcomes in a system with multiple interacting feedback loops
**Time Horizon**: Within-match dynamics (120 balls in T20)

---

## Step 1: Stock and Flow Identification

### 1.1 Stocks (Things That Accumulate)

| Stock | Description | Range (T20) | Accumulation Pattern |
|-------|-------------|-------------|----------------------|
| **Runs** | Total score | 0 → ~200 | Monotonic increase |
| **Wickets** | Dismissals | 0 → 10 | Monotonic increase |
| **Balls Consumed** | Progress through innings | 0 → 120 | Monotonic increase |
| **Batsman Confidence** | "Set-ness" of striker | 0 → high | Increases with balls faced, resets on wicket |
| **Bowler Fatigue** | Accumulated bowling effort | 0 → max overs | Increases with overs bowled |
| **Pressure** | Psychological state | low → high | Varies with situation |
| **Momentum** | Recent success/failure run | negative → positive | Rolling window effect |

### 1.2 Flows (Rates of Change)

| Flow | Description | Units | Driver |
|------|-------------|-------|--------|
| **Run Rate** | Runs per over | RPO | Batting aggression + bowling quality |
| **Wicket Rate** | Wickets per ball | W/ball | Risk-taking + bowling threat |
| **Boundary Rate** | 4s and 6s per over | Boundaries/over | Batsman intent + field settings |
| **Dot Ball Rate** | Scoreless deliveries per over | Dots/over | Bowling accuracy + defensive batting |
| **Extras Rate** | Wides + no-balls per over | Extras/over | Bowler discipline + pressure |

---

## Step 2: Feedback Loop Mapping

### 2.1 Reinforcing Loops (R) - Amplify Change

#### R1: Confidence Spiral (Positive)
```
Batsman scores runs
        ↓ (+)
Batsman confidence increases
        ↓ (+)
Shot selection improves
        ↓ (+)
More runs scored
        ↓ (+)
(loops back)
```
**Impact on Prediction**: "Set" batsmen (30+ runs) have higher boundary probability than new batsmen at the crease.

#### R2: Confidence Collapse (Negative)
```
Dot balls accumulate
        ↓ (+)
Pressure on batsman increases
        ↓ (+)
Batsman takes risky shot
        ↓ (+)
Wicket probability increases
        ↓ (+)
If wicket: New batsman (low confidence)
        ↓ (+)
More dot balls (new batsman cautious)
        ↓ (+)
(loops back)
```
**Impact on Prediction**: Consecutive dot balls (3+) increase both wicket AND boundary probability (desperation shot).

#### R3: Momentum Cascade
```
Team scores boundary
        ↓ (+)
Team morale boost
        ↓ (+)
Bowling team under pressure
        ↓ (+)
Poor line/length OR defensive field
        ↓ (+)
More scoring opportunities
        ↓ (+)
More runs/boundaries
        ↓ (+)
(loops back)
```
**Impact on Prediction**: Recent boundaries (last 6-12 balls) increase subsequent boundary probability.

#### R4: Bowling Dominance
```
Bowler takes wicket
        ↓ (+)
Bowler confidence increases
        ↓ (+)
Better execution
        ↓ (+)
New batsman faces in-form bowler
        ↓ (+)
Higher wicket probability
        ↓ (+)
(loops back)
```
**Impact on Prediction**: A bowler who just took a wicket has elevated wicket probability for next few balls.

### 2.2 Balancing Loops (B) - Resist Change

#### B1: Required Rate Pressure
```
Run rate falls below required rate
        ↓ (+)
Gap to target increases
        ↓ (+)
Required rate increases
        ↓ (+)
Batting team takes more risk
        ↓ (+)
Either: More runs (closes gap)
     Or: More wickets (makes gap worse)
        ↓ (-) or (+)
(negative feedback if runs, positive if wickets)
```
**Impact on Prediction**: Required rate is a PRIMARY driver of risk-taking. High RRR → high boundary attempts → high wicket risk.

#### B2: Wicket Conservation
```
Wickets fall
        ↓ (+)
Remaining wickets decrease
        ↓ (+)
Risk appetite decreases
        ↓ (+)
More conservative play
        ↓ (-)
Wicket rate decreases
        ↓ (-)
(negative feedback stabilizes wicket loss)
```
**Impact on Prediction**: 7+ wickets down → survival mode → dot ball probability increases, boundary probability decreases.

#### B3: Bowler Quota Constraint
```
Bowler bowls over
        ↓ (+)
Remaining overs decrease
        ↓ (+)
If best bowler exhausted
        ↓ (+)
Must use weaker options
        ↓ (+)
Run rate increases (offsetting)
```
**Impact on Prediction**: Death over bowler availability is crucial. If premium bowler has overs left = lower run rate expected.

#### B4: Powerplay Expiry
```
Powerplay runs freely
        ↓ (+)
Overs 1-6 consumed
        ↓ (+)
Powerplay ends (over 7)
        ↓ (+)
Field spreads (5 fielders outside circle)
        ↓ (-)
Boundary rate decreases
        ↓ (-)
(self-limiting growth phase)
```
**Impact on Prediction**: Over 6 → Over 7 transition sees significant drop in boundary rate (15-20% reduction typical).

---

## Step 3: System Archetypes in Cricket

### Archetype 1: "Limits to Growth" - The Innings Ceiling

```
REINFORCING LOOP (Engine):          BALANCING LOOP (Limit):

Runs scored → Confidence →          Wickets lost → Batting
More runs → Higher SR →             strength depleted →
More boundaries →                    Tail exposed →
                                    Run rate decreases →

    Both loops interact:
    Run chase pushes reinforcing loop
    but wicket losses activate balancing loop
```

**Prediction Insight**: Early innings (overs 1-10), reinforcing loop dominates. Late innings with wickets down (overs 15-20 with 6+ down), balancing loop dominates. The PHASE matters for which loop is active.

### Archetype 2: "Fixes That Fail" - The Pressure Response

```
SYMPTOM: Run rate below required rate

QUICK FIX: Hit boundary shot
    ↓
Short-term: Run rate improves
    ↓
Side effect: Wicket risk increases
    ↓
If wicket falls: New batsman, run rate drops more
    ↓
More pressure → More quick fixes → Collapse
```

**Prediction Insight**: Teams behind required rate don't just have higher boundary probability—they have BIMODAL outcomes (either boundary OR wicket, fewer singles).

### Archetype 3: "Shifting the Burden" - Partnership Dependency

```
SYMPTOMATIC SOLUTION:             FUNDAMENTAL SOLUTION:
Rely on set batsman               Build partnership
to score all runs                 with rotation of strike
    ↓                                 ↓
Works while set batsman           Both batsmen develop
is at crease                      confidence
    ↓                                 ↓
Set batsman exhausted OR          Partnership resilient
gets out                          to single wicket
    ↓                                 ↓
Partner not ready,                Sustained scoring
innings collapses
```

**Prediction Insight**: Partnership STATE (both batsmen's balls faced) matters, not just striker state. If striker has 40 runs but partner has 2 balls faced, WICKET RISK for striker is elevated (trying to shield partner).

---

## Step 4: Leverage Point Hierarchy for Cricket Prediction

### Level 12-10: Parameters (Low Leverage - Include but don't over-weight)

| Parameter | Data Field | Why Low Leverage |
|-----------|------------|------------------|
| Ball speed | Not in data | Would be high leverage if available, but missing |
| Exact field position | Not in data | Would matter, but abstracted away |
| Weather conditions | Not in data | Affects grip, swing - but missing |
| Umpire tendencies | `officials` | Marginal effect, mostly noise |

### Level 9-8: Buffers & Delays (Medium Leverage)

| Buffer/Delay | Data Field | Prediction Impact |
|--------------|------------|-------------------|
| **Wickets in hand** | `wickets` | Buffer against collapse. 8 wickets left = can take risks. 2 left = survival mode. |
| **Balls remaining** | Derived | Time buffer. 60 balls left = rebuild possible. 12 balls left = now or never. |
| **Bowler overs remaining** | Derived | Resource buffer. Premium bowlers left = constraint on batting. |
| **Confidence lag** | Batsman balls faced | Delay: Takes 10-15 balls for batsman to "get eye in". |

**Leverage Insight**: Model should explicitly represent BUFFERS (wickets × balls remaining interaction) as they fundamentally change risk calculus.

### Level 7-6: Feedback Loop Strength (High Leverage)

| Feedback | Representation | Why High Leverage |
|----------|----------------|-------------------|
| **Momentum** | Last N balls outcomes | Captures R3 (momentum cascade). Hot streak predicts continued success. |
| **Dot ball pressure** | Consecutive dots | Captures R2 (confidence collapse). Pressure builds non-linearly. |
| **Partnership stability** | Partnership balls faced | Captures B2 (wicket conservation). Established partnership = stability. |
| **Required rate gap** | RRR - current RR | Captures B1 (chase pressure). Gap drives risk-taking behavior. |

**Leverage Insight**: These DERIVED features capture system dynamics that raw features miss. Must compute and include explicitly.

### Level 5-4: Information & Rules (High Leverage)

| Information/Rule | Data Field | Why High Leverage |
|------------------|------------|-------------------|
| **Target visibility** | `target.runs` | 2nd innings knows exactly what's needed. Changes entire strategy. |
| **Powerplay rules** | `powerplays` | Fielding restrictions fundamentally change optimal strategy. |
| **Overs per bowler limit** | Implicit (4 max) | Forces bowler rotation, creates opportunity windows. |
| **Batsman order knowledge** | Implicit | Next batsman's quality affects current batsman's risk-taking. |

**Leverage Insight**: The TARGET and POWERPLAY features are not just parameters—they CHANGE THE GAME RULES. Model should treat these as high-leverage inputs.

### Level 3-2: Goals & Paradigms (Highest Leverage - Implicit)

| Goal/Paradigm | How Manifested | Model Implication |
|---------------|----------------|-------------------|
| **Team goal: Win** | Implicit | In 2nd innings, goal is clear. In 1st innings, goal is "enough runs" (estimated). |
| **Batsman goal: Context-dependent** | Phase-dependent | New batsman: survive. Set batsman: accelerate. These shift probability distributions. |
| **Bowler goal: Context-dependent** | Phase-dependent | Powerplay: wickets priority. Death: economy priority. |

**Leverage Insight**: The model must INFER IMPLICIT GOALS from context. A "dot ball" means different things in different contexts (good economy control vs bad scoring).

---

## Step 5: High-Leverage Features for Model

### 5.1 Features That Capture Feedback Loops

| Feature | Captures | Computation | Priority |
|---------|----------|-------------|----------|
| **Batsman balls faced** | R1 (confidence spiral) | Count from innings data | CRITICAL |
| **Consecutive dot balls** | R2 (pressure buildup) | Rolling window | HIGH |
| **Runs in last 6 balls** | R3 (momentum) | Rolling sum | HIGH |
| **Bowler wickets this spell** | R4 (bowling dominance) | Sum from current spell | MEDIUM |
| **Required run rate** | B1 (chase pressure) | (target - score) / balls_remaining | CRITICAL for 2nd innings |
| **Wickets remaining** | B2 (conservation) | 10 - wickets_fallen | CRITICAL |
| **Partnership balls** | Archetype 3 (burden shifting) | Balls since last wicket | MEDIUM |

### 5.2 Features That Capture Buffers

| Feature | Buffer Type | Computation | Priority |
|---------|-------------|-------------|----------|
| **Wickets × Balls product** | Combined buffer | wickets_remaining * balls_remaining | HIGH |
| **Bowler overs remaining** | Bowling resource | 4 - overs_bowled (per bowler) | MEDIUM |
| **Powerplay balls remaining** | Opportunity buffer | max(0, 36 - balls_faced) | MEDIUM |

### 5.3 Features That Capture Rule Changes

| Feature | Rule/Information | Computation | Priority |
|---------|------------------|-------------|----------|
| **Is powerplay** | Fielding rules | over < 6 | HIGH |
| **Is death overs** | Strategic shift | over >= 16 | HIGH |
| **Has target** | Information asymmetry | innings == 2 | CRITICAL |
| **Balls per boundary needed** | Goal clarity | runs_needed / expected_boundaries_remaining | HIGH |

---

## Step 6: System Dynamics-Based Feature Engineering

### 6.1 Pressure Index (Composite Leverage Feature)

Combines multiple feedback loops into single high-leverage feature:

```python
def compute_pressure_index(match_state):
    """
    Composite feature capturing system pressure level
    Scale: 0 (no pressure) to 1 (maximum pressure)
    """

    # Component 1: Run rate pressure (B1 loop)
    if match_state.innings == 2:
        rrr = match_state.required_run_rate
        current_rr = match_state.current_run_rate
        rr_pressure = min(1.0, max(0, (rrr - current_rr) / 6.0))  # 6+ behind = max pressure
    else:
        # 1st innings: pressure relative to par score at this stage
        expected = match_state.venue_par * (match_state.balls_faced / 120)
        rr_pressure = min(1.0, max(0, (expected - match_state.runs) / 30))

    # Component 2: Wicket pressure (B2 loop)
    wickets_down = match_state.wickets
    wicket_pressure = wickets_down / 10  # Linear scale

    # Component 3: Dot ball pressure (R2 loop)
    consecutive_dots = match_state.consecutive_dots
    dot_pressure = min(1.0, consecutive_dots / 6)  # 6+ dots = max pressure

    # Component 4: Time pressure (buffer depletion)
    balls_remaining = 120 - match_state.balls_faced
    time_pressure = 1.0 - (balls_remaining / 120)  # Increases as match progresses

    # Weighted combination (weights can be learned)
    pressure_index = (
        0.35 * rr_pressure +      # Run rate most important
        0.25 * wicket_pressure +  # Wickets second
        0.20 * dot_pressure +     # Recent pressure
        0.20 * time_pressure      # Time urgency
    )

    return pressure_index
```

### 6.2 Momentum Vector (Capturing R3)

```python
def compute_momentum_vector(last_n_balls, n=12):
    """
    Captures momentum cascade feedback loop
    Returns: (run_momentum, boundary_momentum, wicket_momentum)
    """

    runs_last_n = sum(b.runs.total for b in last_n_balls[-n:])
    boundaries_last_n = sum(1 for b in last_n_balls[-n:] if b.runs.batter >= 4)
    wickets_last_n = sum(1 for b in last_n_balls[-n:] if b.wickets)

    # Normalize relative to expected
    expected_runs = n * 1.3  # ~7.8 RPO typical
    expected_boundaries = n * 0.15  # ~15% boundary rate
    expected_wickets = n * 0.05  # ~5% wicket rate

    run_momentum = (runs_last_n - expected_runs) / expected_runs
    boundary_momentum = (boundaries_last_n - expected_boundaries) / expected_boundaries
    wicket_momentum = (expected_wickets - wickets_last_n) / expected_wickets  # Inverted (fewer wickets = good momentum)

    return run_momentum, boundary_momentum, wicket_momentum
```

### 6.3 Batsman State Vector (Capturing R1)

```python
def compute_batsman_state(batsman_innings):
    """
    Captures confidence spiral state
    """
    balls = batsman_innings.balls_faced
    runs = batsman_innings.runs

    # "Set-ness" - non-linear confidence buildup
    if balls < 5:
        setness = 0.0  # Brand new
    elif balls < 15:
        setness = 0.3 + 0.7 * ((balls - 5) / 10)  # Building
    else:
        setness = 1.0  # Fully set

    # Strike rate relative to career (form indicator)
    current_sr = (runs / balls * 100) if balls > 0 else 0
    sr_deviation = (current_sr - batsman_innings.career_sr) / batsman_innings.career_sr

    # Boundary hitting form
    boundaries = batsman_innings.fours + batsman_innings.sixes
    boundary_rate = boundaries / balls if balls > 0 else 0

    return {
        'setness': setness,
        'sr_form': sr_deviation,
        'boundary_rate': boundary_rate,
        'balls_faced': balls,
        'is_new': balls < 10,
        'is_set': balls >= 20
    }
```

---

## Summary: High-Leverage Feature Recommendations

### Must-Include Features (Capture Key Feedback Loops)

| Feature | Loop(s) Captured | Data Required | Priority |
|---------|------------------|---------------|----------|
| Batsman balls faced | R1 (confidence) | Derived from sequence | 1 |
| Required run rate | B1 (chase pressure) | target, score, balls | 1 |
| Wickets remaining | B2 (conservation) | wickets | 1 |
| Consecutive dot balls | R2 (collapse trigger) | Derived from sequence | 2 |
| Partnership balls | R1/Archetype 3 | Derived from sequence | 2 |
| Last 6/12 balls outcomes | R3 (momentum) | Sequence data | 2 |
| Pressure index (composite) | All loops | Computed feature | 2 |
| Bowler current spell figures | R4 (dominance) | Derived from sequence | 3 |

### Features Model Should Learn To Weight Appropriately

| Feature | Why It Should Be Learned | Expected Learning |
|---------|-------------------------|-------------------|
| Phase indicators (powerplay/middle/death) | Feedback loops have different strengths in different phases | GAT attention should vary by phase |
| Innings indicator (1st/2nd) | 2nd innings has TARGET information, changes everything | Completely different weight distribution |
| Batsman-bowler edge | Matchup importance varies by situation | Attention mechanism learns when matchup matters |

### Warning: Low-Leverage Features (Don't Over-Invest)

| Feature | Why Low Leverage | Recommendation |
|---------|------------------|----------------|
| Venue (alone) | Static, doesn't capture dynamics | Include but as background context only |
| Officials | Minimal impact | Exclude |
| Exact timestamp | No predictive value | Exclude |
| Historical win/loss | Correlational, not causal | Low weight |
