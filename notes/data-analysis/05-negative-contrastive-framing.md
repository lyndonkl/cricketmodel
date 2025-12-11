# Cricket Ball Prediction: Negative Contrastive Framing

## Purpose

This document defines what makes a GOOD predictive feature for cricket ball outcomes by showing what features are NOT useful, near-miss features that seem predictive but fail, and common failure patterns in cricket feature engineering.

---

## Part 1: Defining "Good Predictive Feature"

### Positive Definition

A good predictive feature for cricket ball outcomes:
- Has causal or strong correlational relationship to ball outcome
- Is available at prediction time (before ball is bowled)
- Varies meaningfully across different match situations
- Adds information not captured by other features
- Can be reliably computed from available data

### Why This Definition Is Fuzzy

The positive definition leaves questions:
- How strong must the correlation be?
- What counts as "meaningfully varies"?
- How do we know if it's redundant?
- What about features that matter only in specific contexts?

**Solution**: Define by contrast - show what features FAIL and WHY.

---

## Part 2: Anti-Goals (What Predictive Features Are NOT)

### Anti-Goal 1: Features That Leak Future Information

**Definition**: Information that is only known AFTER the ball outcome

| Anti-Pattern | Why It's Wrong | Example |
|--------------|----------------|---------|
| Next ball's bowler | Only known after current ball | Can't use `delivery[t+1].bowler` to predict `delivery[t]` |
| Ball outcome itself | Tautological | Using `runs.batter` to predict `runs.batter` |
| Post-ball field changes | Future information | Fielding changes made after seeing outcome |
| Match result | Only known at end | `outcome.winner` reveals what happens |

**Near-Miss**: Using cumulative stats that include current ball
- ❌ `batsman_runs` that includes the current delivery
- ✅ `batsman_runs` computed from deliveries [0, t-1]

---

### Anti-Goal 2: Features With No Variance

**Definition**: Features that are constant or near-constant, providing no discrimination

| Anti-Pattern | Why It's Wrong | Example |
|--------------|----------------|---------|
| `balls_per_over = 6` | Always 6 in T20 (constant) | No predictive value |
| `gender = male` | All matches in dataset are male | No variance |
| `match_type = T20` | All matches are T20 | Constant within dataset |
| `innings ∈ {1,2}` | Only 2 values, low entropy | Still useful! (this is a near-miss - see below) |

**Near-Miss**: `innings` seems low-variance (only 2 values) but is HIGHLY predictive because:
- 2nd innings has TARGET information
- Completely changes strategy and outcome distribution
- This shows: variance alone doesn't determine value; CONDITIONAL variance matters

---

### Anti-Goal 3: Features That Are Proxies For Target

**Definition**: Features that encode the outcome in disguised form

| Anti-Pattern | Why It's Wrong | Example |
|--------------|----------------|---------|
| `is_boundary` as feature | Directly encodes outcome (4 or 6) | Leaks target |
| `wicket_this_ball` | Directly encodes wicket outcome | Leaks target |
| `extras_this_delivery` | Part of the outcome | Leaks target |

**Near-Miss**: `runs_last_ball` as feature for current ball
- Seems like leakage but is NOT - it's the PREVIOUS ball's outcome
- Previous outcomes legitimately predict current outcomes (momentum)
- ✅ This is a good feature

---

### Anti-Goal 4: Features That Don't Generalize

**Definition**: Features that overfit to training data but fail on new matches

| Anti-Pattern | Why It's Wrong | Example |
|--------------|----------------|---------|
| `match_id` as feature | Memorizes specific matches | No generalization |
| Exact `date` as numeric | Memorizes specific days | No pattern |
| `umpire_id` without aggregation | Very sparse signal | Unreliable |
| Player ID without embedding | Memorizes specific players | Fails for new players |

**Near-Miss**: `venue` as categorical without embedding
- ❌ Using venue as one-hot (1000+ venues, sparse)
- ✅ Using venue embedding or venue statistics (avg score, boundary rate)

---

### Anti-Goal 5: Features With Spurious Correlation

**Definition**: Features correlated with outcome in training data but not causally related

| Anti-Pattern | Why It's Wrong | Example |
|--------------|----------------|---------|
| Day of week | No causal mechanism | Spurious |
| Match ID parity | Random correlation | Overfitting |
| Player name length | No relationship | Noise |
| Umpire nationality | Not decision-relevant | Confounded |

**Near-Miss**: `time_of_day` (day vs night match)
- Seems spurious but DOES have causal mechanism: dew factor in evening
- Dew affects bowling grip → more wides, harder to grip spinners
- ✅ This is actually predictive for specific outcomes (extras rate)

---

## Part 3: Near-Miss Features (Almost Good But Fail)

### Near-Miss 1: Raw Counts Without Normalization

**Feature**: `total_runs = 150`

**Why It Seems Good**: Score is obviously important

**Why It Fails**: Doesn't account for progress through innings
- 150 after 10 overs is excellent (15 RPO)
- 150 after 18 overs is poor (8.3 RPO)

**The Fix**: Normalize by overs or use run rate
- ✅ `run_rate = total_runs / overs_completed`
- ✅ `score_vs_par = total_runs - expected_score_at_this_stage`

---

### Near-Miss 2: Career Stats Without Context

**Feature**: `batsman_career_average = 45`

**Why It Seems Good**: Career quality should predict performance

**Why It Fails**: Career stats don't account for:
- Current form (slump vs hot streak)
- Matchup (this bowler may have dismissed them 5 times)
- Conditions (career avg in India ≠ performance in Australia)
- Format (Test average ≠ T20 performance)

**The Fix**: Use career stats as BASELINE, adjust with:
- ✅ Recent form (last 5-10 matches)
- ✅ Format-specific stats (T20 career SR, not Test avg)
- ✅ Matchup stats (vs this bowler type)

---

### Near-Miss 3: Single-Dimension Pressure

**Feature**: `required_run_rate = 12`

**Why It Seems Good**: High RRR means pressure to score

**Why It Fails**: RRR alone doesn't capture full pressure:
- RRR = 12 with 8 wickets in hand = achievable
- RRR = 12 with 2 wickets in hand = nearly impossible
- RRR = 12 in over 18 ≠ RRR = 12 in over 5

**The Fix**: Composite pressure that includes:
- ✅ RRR × (1 + wickets_lost/10) - penalize wicket loss
- ✅ RRR × (1 + over_number/20) - penalize late stage
- ✅ Or use full pressure index formula from systems analysis

---

### Near-Miss 4: Binary Phase Indicators

**Feature**: `is_powerplay = True/False`

**Why It Seems Good**: Powerplay has different rules

**Why It Fails**: Creates cliff at boundary
- Ball 36 (last powerplay ball) → is_powerplay = True
- Ball 37 (first middle over) → is_powerplay = False
- But behavior doesn't change discontinuously

**The Fix**: Smooth transitions
- ✅ `powerplay_balls_remaining = max(0, 36 - balls_faced)`
- ✅ Or use continuous phase encoding (over_number / 20)
- ✅ Or use multiple overlapping indicators

---

### Near-Miss 5: Partnership Stats Without Roles

**Feature**: `partnership_runs = 45`

**Why It Seems Good**: Partnership indicates stability

**Why It Fails**: Doesn't capture WHO contributed
- Partnership 45 (striker: 40, partner: 5) = striker dominant, partner vulnerable
- Partnership 45 (striker: 20, partner: 25) = balanced, both set
- These have different implications for current ball

**The Fix**: Include role information
- ✅ `striker_share = striker_runs / partnership_runs`
- ✅ `both_batsmen_set = (striker_balls > 15) AND (partner_balls > 15)`
- ✅ `partnership_balance = 1 - abs(striker_share - 0.5) * 2`

---

### Near-Miss 6: Head-to-Head Without Sample Size

**Feature**: `h2h_strike_rate = 180`

**Why It Seems Good**: Matchup history is predictive

**Why It Fails**: Small sample sizes are unreliable
- SR of 180 from 5 balls = high variance, unreliable
- SR of 180 from 50 balls = meaningful signal
- Treating both the same introduces noise

**The Fix**: Weight by sample size or use prior
- ✅ `h2h_weight = min(1.0, h2h_balls / 30)` (30 balls = full weight)
- ✅ `adjusted_h2h_sr = h2h_weight * h2h_sr + (1 - h2h_weight) * career_sr`
- ✅ Or use Bayesian update with career as prior

---

### Near-Miss 7: Consecutive Events Without Decay

**Feature**: `consecutive_dot_balls = 5`

**Why It Seems Good**: Dot ball pressure builds

**Why It Fails**: Treats all consecutive events equally
- 5 consecutive dots just now = high pressure (fresh)
- 5 dots earlier, then 2 runs, then 3 dots = less pressure (interrupted)
- The feature conflates these

**The Fix**: Use recency weighting
- ✅ Weighted count with exponential decay
- ✅ Or rolling window (last 6 balls) instead of consecutive
- ✅ Or track interruptions: `uninterrupted_dots` vs `total_recent_dots`

---

### Near-Miss 8: Bowler Stats This Match Only

**Feature**: `bowler_economy_this_match = 6.5`

**Why It Seems Good**: Current form matters

**Why It Fails**: Small sample in a single match
- Bowler's first over: 1 over = 6 balls = extreme variance
- May have given 2 boundaries but is actually good
- Or may have gotten lucky early

**The Fix**: Blend match stats with career
- ✅ `blended_economy = match_weight * this_match_econ + (1 - match_weight) * career_econ`
- ✅ Where `match_weight = min(1.0, overs_bowled / 3)` (3 overs = full weight)

---

## Part 4: Failure Pattern Taxonomy

### Pattern 1: Temporal Leakage

**What It Is**: Using information from future balls to predict past balls

**How It Happens**:
- Computing features on entire match before splitting into balls
- Using aggregate stats that include the ball being predicted
- Shuffling balls during training (breaks temporal order)

**Detection**:
- Training accuracy >> test accuracy (model memorizes)
- Unrealistically high performance on "hard" balls

**Prevention**:
- Compute features incrementally up to ball t-1
- Never shuffle within a match during training
- Use strict temporal splits for validation

---

### Pattern 2: Identity Memorization

**What It Is**: Model learns player-specific patterns that don't generalize

**How It Happens**:
- Using player IDs as dense features without embedding
- Training/test split that leaks player-specific information
- Small number of matches per player → overfitting

**Detection**:
- High accuracy on seen players, low on new players
- Feature importance dominated by player IDs

**Prevention**:
- Use player embeddings (learn representations)
- Include matches with new players in test set
- Regularize player-specific features

---

### Pattern 3: Context Collapse

**What It Is**: Treating features as context-independent when they're not

**How It Happens**:
- Using `required_run_rate` in 1st innings (doesn't exist)
- Using same feature weights in powerplay and death overs
- Ignoring interaction between features

**Detection**:
- Model performs well in some phases, poorly in others
- Feature importance varies wildly by segment

**Prevention**:
- Use context-aware models (attention mechanisms)
- Include explicit phase indicators
- Allow feature interactions (non-linear models)

---

### Pattern 4: Aggregation Artifacts

**What It Is**: Aggregating data in ways that lose predictive information

**How It Happens**:
- Using match-level averages for ball-level prediction
- Averaging across different contexts (powerplay + death)
- Losing sequence information in aggregation

**Detection**:
- Ball-level model outperforms aggregated model
- Residual patterns visible in aggregated predictions

**Prevention**:
- Predict at ball level, aggregate predictions if needed
- Maintain context when aggregating (per-phase averages)
- Preserve sequence information (use sequence models)

---

### Pattern 5: Survivorship Bias

**What It Is**: Only seeing data from balls that happened, not balls that didn't

**How It Happens**:
- Batsman stats only from balls they faced (not balls they were rotated away from)
- Only seeing partnerships that continued (not ones about to break)
- Training on completed innings (not innings interrupted by rain)

**Detection**:
- Model underestimates wicket probability late in partnerships
- Model overestimates performance of batsmen who "survived"

**Prevention**:
- Include all data points, even partial innings
- Be aware of selection effects in feature computation
- Consider counterfactuals (what if wicket had fallen?)

---

## Part 5: Decision Criteria Checklist

### Feature Inclusion Checklist

Before including a feature, verify:

```
□ 1. AVAILABLE: Can it be computed BEFORE the ball is bowled?
     - Not a post-hoc statistic
     - Not dependent on current ball outcome

□ 2. VARIES: Does it have meaningful variance in the dataset?
     - Not constant or near-constant
     - Varies across match situations

□ 3. PREDICTIVE: Is there a plausible causal/correlational mechanism?
     - Can explain WHY it would predict outcomes
     - Not purely spurious correlation

□ 4. NON-REDUNDANT: Does it add information beyond existing features?
     - Not a linear combination of other features
     - Captures something new

□ 5. GENERALIZABLE: Will it work on new matches/players/venues?
     - Not memorizing specific instances
     - Uses embeddings or aggregations appropriately

□ 6. NORMALIZED: Is it properly scaled and contextualized?
     - Accounts for stage of innings
     - Handles small sample sizes
     - Doesn't create artificial boundaries

□ 7. TEMPORALLY CLEAN: No leakage from future?
     - Computed incrementally
     - Uses only past information
```

### Feature Rejection Criteria

**Automatic Reject If**:
- Uses outcome of current ball (leakage)
- Constant in all rows (no variance)
- No plausible causal mechanism AND no empirical correlation
- Highly correlated (>0.95) with existing feature

**Investigate Further If**:
- Small sample sizes in computation
- Context-dependent (may need interaction terms)
- Potentially confounded (correlation != causation)

---

## Part 6: Summary - Good Features vs Bad Features

### Side-by-Side Comparison

| Dimension | ❌ Bad Feature | ✅ Good Feature |
|-----------|---------------|-----------------|
| **Temporal** | Uses current/future ball info | Uses only past balls |
| **Variance** | Constant (balls_per_over=6) | Varies meaningfully |
| **Normalization** | Raw count (runs=150) | Rate or relative (run_rate=12) |
| **Context** | Context-blind (career avg) | Context-aware (recent form + matchup) |
| **Sample Size** | Ignores reliability (h2h from 3 balls) | Weights by sample (Bayesian blend) |
| **Boundaries** | Hard boundaries (is_powerplay) | Smooth transitions (balls_remaining) |
| **Aggregation** | Over-aggregated (match average) | Granular (per-ball state) |
| **Generalization** | Memorizes (match_id) | Learns patterns (venue embedding) |

### The Golden Rule

> A good predictive feature is one that, if you knew it and NOTHING else about the current ball, would shift your prediction of the outcome distribution in a direction that's correct more often than chance.

**Test**: Can you explain to a cricket expert WHY this feature matters? If yes → likely good. If no → investigate further.
