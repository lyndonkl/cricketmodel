# Cricket Ball Prediction: Decomposition & Reconstruction Analysis

## System Definition

**System**: Cricket match ball-by-ball prediction
**Goal**: Predict the outcome of the next ball given current match state
**Boundaries**:
- In scope: All available Cricsheet JSON data fields, derived features computable from this data
- Out of scope: Real-time video analysis, weather data, pitch maps, exact ball trajectories

**Success Criteria**: Model that accurately predicts ball outcome distribution (runs: 0,1,2,3,4,6 + wicket + extras)

---

## Step 1: Raw Data Decomposition

### 1.1 Match-Level Components (Static per match)

| Component | Data Fields | Description |
|-----------|-------------|-------------|
| **Match Context** | `dates`, `season`, `event.name`, `event.match_number`, `event.group`, `event.stage` | When, what competition, stakes level |
| **Match Type** | `match_type`, `match_type_number`, `balls_per_over`, `overs` | Format constraints (T20 = 20 overs, 6 balls/over) |
| **Venue** | `venue`, `city` | Physical location affecting pitch behavior |
| **Teams** | `teams[]`, `team_type` | Who is playing (international vs club) |
| **Squad** | `players[team][]` | 11 players per side with unique IDs |
| **Toss** | `toss.winner`, `toss.decision` | First innings advantage/disadvantage |
| **Officials** | `officials.umpires[]`, `officials.match_referees[]` | Decision makers |
| **Outcome** | `outcome.winner`, `outcome.by` | Final result (available retrospectively) |

### 1.2 Innings-Level Components (Static per innings)

| Component | Data Fields | Description |
|-----------|-------------|-------------|
| **Batting Team** | `innings[].team` | Which team is batting |
| **Target** | `innings[].target.runs`, `innings[].target.overs` | Chase target (2nd innings only) |
| **Powerplays** | `innings[].powerplays[]` | Fielding restriction periods |

### 1.3 Delivery-Level Components (Per ball - the core prediction unit)

| Component | Data Fields | Description |
|-----------|-------------|-------------|
| **Over Context** | `over` (0-indexed) | Current over number |
| **Ball in Over** | Derived from delivery position in array | Ball 1-6 (plus extras) |
| **Striker** | `batter` | Current batsman facing |
| **Non-Striker** | `non_striker` | Partner at other end |
| **Bowler** | `bowler` | Current bowler delivering |
| **Runs Scored** | `runs.batter`, `runs.extras`, `runs.total` | Outcome breakdown |
| **Extras Type** | `extras.wides`, `extras.noballs`, `extras.byes`, `extras.legbyes`, `extras.penalty` | How extras occurred |
| **Wicket** | `wickets[].kind`, `wickets[].player_out`, `wickets[].fielders[]` | Dismissal details |
| **Review** | `review.by`, `review.decision`, `review.umpires_call` | DRS usage |
| **Replacements** | `replacements.match[]`, `replacements.role[]` | Player substitutions |

---

## Step 2: Feature Impact Analysis

### 2.1 Direct Impact Features (Immediately Observable)

These features DIRECTLY influence the next ball's outcome probability distribution:

#### A. Current Ball Context

| Feature | Impact Mechanism | Why It Matters |
|---------|------------------|----------------|
| **Ball in Over (1-6)** | Bowlers bowl different deliveries at different stages; batsmen adjust rhythm | Ball 1: Testing line. Ball 6: Likely yorker/slower ball. Batsman knows last ball of over = safe to rotate strike |
| **Over Number (0-19)** | Phase of innings changes strategy | Overs 0-5: Powerplay aggression. Overs 6-14: Consolidation. Overs 15-19: Death overs acceleration |
| **Powerplay Status** | Fielding restrictions mandate different bowling | Only 2 fielders outside 30-yard circle = more boundaries, more risk-taking |

#### B. Current Player State

| Feature | Impact Mechanism | Why It Matters |
|---------|------------------|----------------|
| **Current Batsman** | Individual skill, style, form | Rohit Sharma vs tail-ender = entirely different outcome distributions |
| **Batsman's Current Score** | "Set" batsman vs new batsman | 0-10: Survival mode. 10-30: Building. 30+: Attacking |
| **Batsman's Balls Faced** | Strike rate pressure, confidence | Few balls + low score = pressure. Many balls + good SR = confidence |
| **Current Bowler** | Speed, spin, variation | Fast bowler vs spinner = different scoring patterns |
| **Bowler's Figures (current spell)** | Confidence, workload | 4-0-10-0: Pressure to get wicket. 2-0-25-0: Defensive |
| **Bowler's Overs Bowled (total)** | Fatigue, remaining quota | T20: Max 4 overs. Approaching limit changes captaincy |

#### C. Match State

| Feature | Impact Mechanism | Why It Matters |
|---------|------------------|----------------|
| **Total Score** | Pressure level | 200/2 at over 18 = aggressive. 80/6 at over 15 = defensive |
| **Wickets Fallen** | Risk appetite | 2 down: Can attack. 7 down: Protect tail |
| **Run Rate** | Momentum | 12+ RPO: Team dominant. <6 RPO: Struggling |
| **Required Run Rate (2nd innings)** | Chase pressure | RRR = 6: Comfortable. RRR = 12: Must attack every ball |
| **Balls Remaining** | Time pressure | 120 balls: Plan innings. 12 balls: Desperation |

### 2.2 Historical Features (Require Aggregation)

These require computing across multiple matches:

#### A. Player Career Statistics

| Feature | Computation | Impact Mechanism |
|---------|-------------|------------------|
| **Batsman Career SR vs Pace** | Avg SR when facing pace bowlers | Some batsmen dominate pace, struggle vs spin |
| **Batsman Career SR vs Spin** | Avg SR when facing spin bowlers | Opposite pattern for others |
| **Batsman SR by Phase** | Avg SR in powerplay/middle/death | Death overs specialist vs powerplay player |
| **Bowler Economy Rate** | Runs per over career | Control bowler vs attacking bowler |
| **Bowler Strike Rate** | Balls per wicket | Wicket-taking threat |
| **Bowler Extras Rate** | Wides + No-balls per over | Discipline indicator |

#### B. Head-to-Head Statistics

| Feature | Computation | Impact Mechanism |
|---------|-------------|------------------|
| **Batsman vs This Bowler** | SR, dismissals in matchup history | Known weaknesses get exploited |
| **Bowler vs This Batsman** | Wickets, economy in matchup history | Confidence from past success |
| **Batsman vs This Bowling Style** | SR vs pace/spin generally | Style matchup more robust than individual |

#### C. Recent Form

| Feature | Computation | Impact Mechanism |
|---------|-------------|------------------|
| **Batsman Last 5 Matches SR** | Recent performance | Current form vs career average |
| **Bowler Last 5 Matches Economy** | Recent performance | In-form vs out-of-form |
| **Team Recent Form** | Win/loss last 5 matches | Confidence, pressure |

### 2.3 Contextual Features (Situational)

| Feature | Impact Mechanism | Why It Matters |
|---------|------------------|----------------|
| **Venue History** | High-scoring vs low-scoring ground | Dubai: 160 is competitive. Bangalore: 200+ common |
| **Toss Effect** | Batting first vs chasing at this venue | Some venues favor chasing (dew factor) |
| **Tournament Stage** | Group vs knockout pressure | Finals: Different decision-making |
| **Day/Night** | Dew, visibility | Evening: Ball harder to grip for bowlers |

---

## Step 3: Component Relationship Mapping

### 3.1 Dependency Graph

```
                    MATCH CONTEXT
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      VENUE         TEAMS/SQUADS      TOSS
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                   INNINGS START
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    BATTING TEAM    TARGET (2nd)    POWERPLAY
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                   ┌──────────┐
                   │ DELIVERY │ ←─── Prediction Target
                   └──────────┘
                         ▲
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────┴────┐    ┌─────┴─────┐   ┌─────┴─────┐
    │ BATSMAN │    │  BOWLER   │   │  MATCH    │
    │  STATE  │    │   STATE   │   │   STATE   │
    └────┬────┘    └─────┬─────┘   └─────┬─────┘
         │               │               │
         ▼               ▼               ▼
    Career Stats    Career Stats    Score/Wickets
    Recent Form     Recent Form     Run Rate
    This Innings    This Spell      Pressure Level
    vs This Bowler  vs This Batsman Target (if 2nd)
```

### 3.2 Critical Paths for Prediction

**Path 1: Batsman-Centric**
```
Current Batsman → Batsman's Current State → vs This Bowler Type → Outcome
```
- Primary driver: Who is batting
- Modulated by: How set they are, their matchup

**Path 2: Situation-Centric**
```
Match State → Required Rate Pressure → Risk Level → Outcome
```
- Primary driver: What does the team need?
- Modulated by: How much pressure, wickets in hand

**Path 3: Bowler-Centric**
```
Current Bowler → Bowler's Strategy → Ball Type → Outcome
```
- Primary driver: What the bowler delivers
- Modulated by: Phase, batsman, field

---

## Step 4: Reconstruction for Ball Prediction

### 4.1 Bottleneck Analysis

**Most Important Features (High Variance Explanation)**:

1. **Current Batsman Identity** - Single biggest predictor
   - Virat Kohli vs #11 batsman = fundamentally different distributions
   - Why: Skill gap creates 3-5x difference in boundary probability

2. **Current Bowler Identity** - Second biggest predictor
   - Death overs specialist vs part-timer = different threat levels
   - Why: 2-3x difference in wicket probability, economy varies

3. **Phase of Innings** - Third biggest predictor
   - Powerplay (1-6) vs Middle (7-15) vs Death (16-20)
   - Why: Strategy completely changes, acceptable risk level shifts

4. **Match Situation Pressure** - Fourth biggest predictor
   - Required Rate + Wickets in Hand + Balls Remaining
   - Why: Determines risk appetite (conservative vs aggressive)

5. **Batsman-Bowler Matchup History** - Fifth biggest predictor
   - Some batsmen dominate certain bowlers (and vice versa)
   - Why: Technical matchups + psychological edge

### 4.2 Reconstruction: Optimal Feature Groupings for GAT Graph

Based on decomposition, here's how to structure the graph for each ball:

**Node Type 1: BATSMAN NODE**
```
Features:
- Identity embedding (learned)
- Current innings: runs, balls, SR, dots, boundaries
- Career stats: avg SR vs pace, vs spin, by phase
- Form: last 5 matches SR
- vs current bowler: SR, dismissals
```

**Node Type 2: BOWLER NODE**
```
Features:
- Identity embedding (learned)
- Current spell: overs, runs, wickets, economy
- Career stats: economy, SR, extras rate
- Form: last 5 matches economy
- vs current batsman: wickets, economy
```

**Node Type 3: MATCH STATE NODE**
```
Features:
- Score: runs, wickets
- Progress: over.ball (normalized 0-1)
- Pressure: run rate, required rate (if chasing)
- Phase: powerplay flag, death overs flag
- Innings: 1st vs 2nd
```

**Node Type 4: NON-STRIKER NODE** (Partner influence)
```
Features:
- Identity embedding
- Current innings stats
- Partnership: runs, balls together
```

**Edge Types**:
- Batsman ↔ Bowler: Matchup relationship (attention should learn importance)
- Batsman ↔ Non-Striker: Partnership synergy
- Batsman ↔ Match State: Pressure transmission
- Bowler ↔ Match State: Tactical response

### 4.3 Simplification Opportunities

**Remove/Collapse Low-Information Features**:
- `officials` → Minimal impact on ball outcomes (umpire variance is noise)
- `event.group` → Can be captured by `event.stage` (group vs knockout sufficient)
- `supersubs` → Rarely used in modern cricket
- `review` → Post-hoc information, not predictive

**Consolidate Redundant Features**:
- Instead of separate `wides`, `noballs`, `byes`, `legbyes` → Single `extras_type` categorical
- Instead of all career stats → PCA-compressed "batsman profile" embedding
- Instead of full match history → Rolling window (last 10 balls) sufficient for momentum

---

## Step 5: Feature Priority Ranking

### Tier 1: Must-Have (Core Prediction Features)

| Rank | Feature | Type | Impact Score |
|------|---------|------|--------------|
| 1 | Current Batsman ID | Categorical | 10/10 |
| 2 | Current Bowler ID | Categorical | 9/10 |
| 3 | Over Number (0-19) | Numeric | 8/10 |
| 4 | Ball in Over (1-6) | Numeric | 7/10 |
| 5 | Total Score | Numeric | 8/10 |
| 6 | Wickets Fallen | Numeric | 8/10 |
| 7 | Innings (1st/2nd) | Binary | 7/10 |
| 8 | Target (2nd innings) | Numeric | 9/10 for 2nd innings |
| 9 | Batsman Current Runs | Numeric | 7/10 |
| 10 | Batsman Current Balls | Numeric | 7/10 |

### Tier 2: High-Value (Enhance Prediction)

| Rank | Feature | Type | Impact Score |
|------|---------|------|--------------|
| 11 | Bowler Current Overs | Numeric | 6/10 |
| 12 | Bowler Current Runs | Numeric | 6/10 |
| 13 | Required Run Rate | Numeric | 8/10 for chases |
| 14 | Powerplay Status | Binary | 7/10 for overs 1-6 |
| 15 | Non-Striker ID | Categorical | 5/10 |
| 16 | Venue | Categorical | 6/10 |
| 17 | Team Type (int'l/club) | Binary | 4/10 |

### Tier 3: Supplementary (Derived Features)

| Rank | Feature | Computation | Impact Score |
|------|---------|-------------|--------------|
| 18 | Batsman Strike Rate (this innings) | runs/balls * 100 | 6/10 |
| 19 | Current Run Rate | total_runs / overs_completed | 6/10 |
| 20 | Balls Remaining | (20 - over) * 6 - ball_in_over | 7/10 |
| 21 | Runs Required | target - score | 8/10 for chases |
| 22 | Partnership Runs | Derived from sequence | 5/10 |
| 23 | Partnership Balls | Derived from sequence | 5/10 |
| 24 | Recent Momentum | Runs in last 6/12 balls | 6/10 |
| 25 | Dot Ball Pressure | Consecutive dots | 6/10 |

---

## Summary: Reconstructed System for Prediction

### Core Input Structure (Per Ball)

```
BALL PREDICTION INPUT
│
├── STATIC CONTEXT
│   ├── Match ID (for batching)
│   ├── Venue embedding
│   └── Team embeddings (batting, bowling)
│
├── CURRENT STATE (changes every ball)
│   ├── Score: runs, wickets
│   ├── Progress: over.ball (0.0 to 19.5)
│   ├── Target: runs needed, RRR (if 2nd innings)
│   └── Phase: powerplay/middle/death
│
├── BATSMAN STATE
│   ├── ID embedding
│   ├── This innings: runs, balls, SR
│   └── Career profile: compressed stats
│
├── BOWLER STATE
│   ├── ID embedding
│   ├── This spell: overs, runs, wickets
│   └── Career profile: compressed stats
│
├── PARTNERSHIP STATE
│   ├── Non-striker ID embedding
│   ├── Partnership: runs, balls
│   └── Strike rotation pattern
│
└── SEQUENCE CONTEXT (for temporal model)
    └── Last N balls: outcomes, batsmen, bowlers
```

### Output Distribution

```
P(outcome | input) = softmax over:
- 0 runs (dot ball)
- 1 run
- 2 runs
- 3 runs
- 4 runs (boundary)
- 5 runs (rare)
- 6 runs (six)
- Wicket
- Wide
- No-ball
- Bye/Leg-bye
```

---

## Next Steps

1. **Systems Thinking Analysis** → Identify feedback loops and high-leverage features
2. **Synthesis Analysis** → Map data to graph structure with analogies
3. **Derived Features Catalog** → Complete list of computable supplementary features
