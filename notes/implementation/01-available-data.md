# Available Data for Cricket Ball Prediction

## Data Source

**Primary Source**: [Cricsheet JSON Format](https://cricsheet.org/format/json/) (v1.0.0, v1.1.0)
- 3,046 male T20 matches
- Ball-by-ball delivery data with player identifiers

## What's Available

### Match-Level (Static per match)

| Field | Description | Use |
|-------|-------------|-----|
| `venue`, `city` | Match location | Venue embedding, par scores |
| `teams[]` | Participating teams | Team embeddings |
| `players[team][]` | Squad with unique IDs | Player embeddings |
| `toss.winner`, `toss.decision` | Toss outcome | Context feature |
| `dates`, `season` | When played | Temporal features |
| `event.name`, `event.stage` | Tournament context | Match importance |

### Innings-Level (Static per innings)

| Field | Description | Use |
|-------|-------------|-----|
| `innings[].team` | Batting team | Team context |
| `innings[].target.runs` | Chase target (2nd innings) | Chase state node |
| `innings[].powerplays[]` | Fielding restrictions | Phase state node |

### Delivery-Level (Per ball)

| Field | Description | Use |
|-------|-------------|-----|
| `batter` | Striker ID | Batsman identity node |
| `bowler` | Bowler ID | Bowler identity node |
| `non_striker` | Partner ID | Partnership node |
| `runs.batter` | Runs off bat | Target label |
| `runs.extras` | Extra runs | Target label |
| `runs.total` | Total runs | Target label |
| `extras.*` | Extra type breakdown | Target label |
| `wickets[]` | Dismissal details | Target label |

## What's NOT Available

| Missing Data | Impact | Mitigation |
|--------------|--------|------------|
| **Ball speed/trajectory** | Cannot model ball type directly | Infer from bowler type, outcomes |
| **Field positions** | Cannot model field settings | Use phase proxies (powerplay restrictions) |
| **Shot type played** | Only outcome, not how | Learn from outcome patterns |
| **Weather conditions** | Affects grip, swing | Ignore (accept noise) |
| **Pitch condition** | Changes during match | Infer from innings progression |
| **Player attributes** | Height, age, handedness | Use external data if available |

## Derived Features

We compute ~110 derived features from raw data. Full catalog with computation code:
**[See: Derived Features Catalog](../data-analysis/04-derived-features-catalog.md)**

### Summary by Category

| Category | Count | Priority | Node Assignment |
|----------|-------|----------|-----------------|
| Match State | 16 | P0-P1 | Score State, Chase State, Time Pressure |
| Batsman State | 18 | P0-P1 | Batsman State node |
| Bowler State | 15 | P1-P2 | Bowler State node |
| Partnership | 12 | P1-P2 | Partnership node |
| Momentum | 12 | P1 | Batting/Bowling Momentum nodes |
| Pressure | 9 | P1 | Pressure Index node |
| Historical | 16 | P2-P3 | Requires external data |
| Sequence | 12 | P1 | Temporal layer input |

## Feature Priority Tiers

For implementation phases, features are prioritized:

| Tier | Description | Count | Example Features |
|------|-------------|-------|------------------|
| **P0 (Critical)** | Required for any prediction | 14 | `batsman_id`, `bowler_id`, `score`, `wickets`, `over.ball`, `target`, `innings` |
| **P1 (High)** | Significant accuracy improvement | 47 | `pressure_index`, `momentum`, `setness`, `partnership`, `consecutive_dots` |
| **P2 (Medium)** | Moderate improvement | 45 | `career_stats`, `h2h`, `venue_effects` |
| **P3 (Low)** | Marginal improvement | 4 | `event_metadata`, `officials` |

**Full priority breakdown**: [Derived Features Catalog § Summary](../data-analysis/04-derived-features-catalog.md#summary-feature-count-by-category)

## External Data Sources (Optional)

If external data is sourced, these would enhance the model:

| Data Type | Potential Source | Use |
|-----------|------------------|-----|
| Career statistics | ESPNCricinfo API | Career stats nodes |
| Head-to-head history | Computed from historical matches | Matchup edge features |
| Player attributes | Cricbuzz, ESPNCricinfo | Player embedding enrichment |
| Venue historical scores | Computed from data | Venue node features |

## Data Integrity Constraints

From [negative contrastive analysis](../data-analysis/05-negative-contrastive-framing.md):

1. **Temporal Integrity**: All features computed from balls [0, t-1] only
2. **No Leakage**: Current ball outcome never used as feature
3. **No Shuffling**: Maintain temporal order within matches
4. **Sample Size Awareness**: Weight features by reliability (Bayesian blend for small samples)

## Next Steps

With available data understood, see:
- [02-graph-structure.md](./02-graph-structure.md) - How data maps to graph nodes
- [05-derived-vs-learned.md](./05-derived-vs-learned.md) - Which features to compute vs learn
