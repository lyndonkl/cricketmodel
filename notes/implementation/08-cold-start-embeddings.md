# Cold-Start Embeddings: Players and Venues

## The Problem

When a player or venue not seen during training appears in a live match, we need to generate a reasonable embedding. This is the "cold-start" problem.

## Key Insight: Embeddings Are Generated, Not Looked Up

**The model never sees player IDs or venue names.** It only sees embedding vectors.

```
WRONG mental model:
  player_id "kohli" → lookup table[kohli] → embedding

CORRECT mental model:
  player_stats {sr: 137, avg: 48, ...} → encoder_network() → embedding
```

This means:
- The encoder learns a **meaningful embedding space** during training
- New players/venues get embeddings based on their **features**, not their ID
- Similar features → similar embeddings → similar model behavior

## Solution: Hierarchical Fallback with Adaptation

```
┌─────────────────────────────────────────────────────────────────┐
│              PLAYER EMBEDDING RESOLUTION                         │
│                                                                  │
│  Player ID ──► Known Player? ─── YES ──► Learned Embedding      │
│                     │                                            │
│                     NO                                           │
│                     ▼                                            │
│               Career Stats ─── YES ──► Generate from Stats       │
│               Available?                                         │
│                     │                                            │
│                     NO                                           │
│                     ▼                                            │
│               Role/Position ─── YES ──► Role Prototype           │
│               Known?                                             │
│                     │                                            │
│                     NO                                           │
│                     ▼                                            │
│               Average Player Embedding                           │
│                                                                  │
│  THEN: Adapt embedding during match based on observed behavior  │
└─────────────────────────────────────────────────────────────────┘
```

## Component 1: Stat-Based Embedding Generator

### Architecture

```python
class PlayerEmbeddingGenerator(nn.Module):
    """
    Generates player embedding from career statistics.
    Trained to reconstruct learned embeddings from stats.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()

        # Batsman stat encoder
        self.batsman_encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim)
        )

        # Bowler stat encoder
        self.bowler_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim)
        )

    def generate_batsman_embedding(self, stats: BatsmanStats) -> torch.Tensor:
        """Generate embedding from batsman career stats."""
        features = torch.tensor([
            stats.t20_strike_rate / 150.0,      # Normalize to ~1
            stats.t20_average / 40.0,
            stats.matches / 100.0,
            stats.innings / 100.0,
            stats.sr_vs_pace / 150.0,
            stats.sr_vs_spin / 150.0,
            stats.sr_powerplay / 150.0,
            stats.sr_death / 180.0,
            stats.boundary_percentage / 20.0,
            stats.dot_ball_percentage / 40.0,
            float(stats.is_opener),
            float(stats.is_finisher),
        ])
        return self.batsman_encoder(features)

    def generate_bowler_embedding(self, stats: BowlerStats) -> torch.Tensor:
        """Generate embedding from bowler career stats."""
        features = torch.tensor([
            stats.t20_economy / 10.0,           # Normalize to ~1
            stats.t20_strike_rate / 25.0,
            stats.matches / 100.0,
            stats.wickets / 100.0,
            stats.economy_powerplay / 10.0,
            stats.economy_death / 12.0,
            stats.dot_ball_percentage / 50.0,
            float(stats.is_pace),
            float(stats.is_spin),
            float(stats.is_death_specialist),
        ])
        return self.bowler_encoder(features)
```

### Input Stats Format

```python
@dataclass
class BatsmanStats:
    """Career statistics for batsman embedding generation."""
    t20_strike_rate: float      # Career T20 SR (e.g., 135.5)
    t20_average: float          # Career T20 average (e.g., 32.4)
    matches: int                # T20 matches played
    innings: int                # T20 innings batted
    sr_vs_pace: float           # SR against pace bowling
    sr_vs_spin: float           # SR against spin bowling
    sr_powerplay: float         # SR in powerplay overs
    sr_death: float             # SR in death overs
    boundary_percentage: float  # % of runs from boundaries
    dot_ball_percentage: float  # % of balls that are dots
    is_opener: bool             # Usually opens
    is_finisher: bool           # Usually bats 5-7


@dataclass
class BowlerStats:
    """Career statistics for bowler embedding generation."""
    t20_economy: float          # Career T20 economy
    t20_strike_rate: float      # Balls per wicket
    matches: int                # T20 matches played
    wickets: int                # T20 wickets taken
    economy_powerplay: float    # Economy in powerplay
    economy_death: float        # Economy in death overs
    dot_ball_percentage: float  # % of balls that are dots
    is_pace: bool               # Pace bowler
    is_spin: bool               # Spin bowler
    is_death_specialist: bool   # Good death bowler
```

### Training the Generator

```python
def train_embedding_generator(
    generator: PlayerEmbeddingGenerator,
    learned_embeddings: nn.Embedding,
    player_stats: Dict[str, PlayerStats],
    epochs: int = 100
):
    """
    Train generator to reconstruct learned embeddings from stats.
    """
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for player_id, stats in player_stats.items():
            if player_id not in learned_embeddings:
                continue

            # Target: the learned embedding
            target = learned_embeddings[player_id]

            # Generated: from stats
            if stats.is_batsman:
                generated = generator.generate_batsman_embedding(stats.batting)
            else:
                generated = generator.generate_bowler_embedding(stats.bowling)

            loss = criterion(generated, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
```

## Component 2: Role Prototypes

### Predefined Role Embeddings

```python
class RolePrototypes:
    """
    Learned prototype embeddings for player roles.
    Used when no career stats available.
    """

    BATTING_ROLES = [
        "opener_aggressive",      # Rohit Sharma type
        "opener_anchor",          # Shikhar Dhawan type
        "top_order_accumulator",  # Kohli type
        "middle_order_aggressor", # Hardik Pandya type
        "finisher",               # Dhoni/Russell type
        "lower_order",            # Bowlers who bat
        "tailender",              # #10, #11
    ]

    BOWLING_ROLES = [
        "powerplay_pace",         # New ball specialist
        "death_pace",             # Yorker specialist
        "pace_allrounder",        # Medium pace, can bat
        "leg_spin",               # Leg spinners
        "off_spin",               # Off spinners
        "left_arm_spin",          # Left arm orthodox
        "mystery_spin",           # Variation bowlers
    ]

    def __init__(self, embed_dim: int = 64):
        # These are learned during training
        self.batting_prototypes = nn.Embedding(len(self.BATTING_ROLES), embed_dim)
        self.bowling_prototypes = nn.Embedding(len(self.BOWLING_ROLES), embed_dim)

    def get_batting_prototype(self, role: str) -> torch.Tensor:
        idx = self.BATTING_ROLES.index(role)
        return self.batting_prototypes(torch.tensor(idx))

    def get_bowling_prototype(self, role: str) -> torch.Tensor:
        idx = self.BOWLING_ROLES.index(role)
        return self.bowling_prototypes(torch.tensor(idx))
```

### Learning Prototypes

```python
def learn_role_prototypes(
    prototypes: RolePrototypes,
    learned_embeddings: nn.Embedding,
    player_roles: Dict[str, str]
):
    """
    Learn prototypes as cluster centers of known players.
    """
    # Group players by role
    role_groups = defaultdict(list)
    for player_id, role in player_roles.items():
        if player_id in learned_embeddings:
            role_groups[role].append(learned_embeddings[player_id])

    # Set prototype as mean of group
    for role, embeddings in role_groups.items():
        if embeddings:
            mean_embedding = torch.stack(embeddings).mean(dim=0)
            if role in prototypes.BATTING_ROLES:
                idx = prototypes.BATTING_ROLES.index(role)
                prototypes.batting_prototypes.weight.data[idx] = mean_embedding
            elif role in prototypes.BOWLING_ROLES:
                idx = prototypes.BOWLING_ROLES.index(role)
                prototypes.bowling_prototypes.weight.data[idx] = mean_embedding
```

## Component 3: In-Match Adaptation

### Adaptive Embedding

```python
class AdaptiveEmbedding:
    """
    Embedding that adapts during match based on observed behavior.
    """

    def __init__(
        self,
        initial_embedding: torch.Tensor,
        learning_rate: float = 0.1,
        min_balls_for_adaptation: int = 6
    ):
        self.embedding = initial_embedding.clone().requires_grad_(True)
        self.initial = initial_embedding.clone()
        self.lr = learning_rate
        self.balls_seen = 0
        self.min_balls = min_balls_for_adaptation

    def get_embedding(self) -> torch.Tensor:
        """Return current (possibly adapted) embedding."""
        return self.embedding

    def update(
        self,
        model: nn.Module,
        context: dict,
        actual_outcome: int,
        predicted_probs: torch.Tensor
    ):
        """
        Update embedding based on prediction error.

        If player consistently outperforms/underperforms predictions,
        adjust embedding toward behavior that explains observations.
        """
        self.balls_seen += 1

        # Don't adapt with too few observations
        if self.balls_seen < self.min_balls:
            return

        # Compute gradient of loss w.r.t. embedding
        loss = F.cross_entropy(predicted_probs.unsqueeze(0), torch.tensor([actual_outcome]))
        grad = torch.autograd.grad(loss, self.embedding, retain_graph=True)[0]

        # Update embedding
        with torch.no_grad():
            self.embedding -= self.lr * grad

            # Regularize: don't drift too far from initial
            drift = self.embedding - self.initial
            if drift.norm() > 1.0:
                self.embedding = self.initial + drift / drift.norm()

        # Decay learning rate
        self.lr *= 0.95

    def get_adaptation_info(self) -> dict:
        """Return info about how much embedding has adapted."""
        drift = (self.embedding - self.initial).norm().item()
        return {
            "balls_seen": self.balls_seen,
            "drift_magnitude": drift,
            "learning_rate": self.lr
        }
```

## Component 4: Unified Embedding Manager

```python
class PlayerEmbeddingManager:
    """
    Manages all player embeddings: learned, generated, and adaptive.
    """

    def __init__(
        self,
        learned_embeddings: nn.Embedding,
        generator: PlayerEmbeddingGenerator,
        prototypes: RolePrototypes,
        embed_dim: int = 64
    ):
        self.learned = learned_embeddings
        self.generator = generator
        self.prototypes = prototypes
        self.embed_dim = embed_dim

        # Cache for generated embeddings
        self.generated_cache: Dict[str, torch.Tensor] = {}

        # Adaptive embeddings for current match
        self.adaptive: Dict[str, AdaptiveEmbedding] = {}

        # Average embedding fallback
        self.average_embedding = learned_embeddings.weight.mean(dim=0)

    def get_embedding(
        self,
        player_id: str,
        player_info: Optional[PlayerInfo] = None,
        use_adaptive: bool = True
    ) -> torch.Tensor:
        """
        Get embedding for a player, using best available method.
        """
        # 1. Check adaptive (if in current match)
        if use_adaptive and player_id in self.adaptive:
            return self.adaptive[player_id].get_embedding()

        # 2. Check learned embeddings
        if player_id in self.learned:
            return self.learned[player_id]

        # 3. Check generated cache
        if player_id in self.generated_cache:
            return self.generated_cache[player_id]

        # 4. Generate from stats if available
        if player_info and player_info.has_career_stats:
            embedding = self._generate_from_stats(player_info)
            self.generated_cache[player_id] = embedding
            return embedding

        # 5. Use role prototype if role known
        if player_info and player_info.role:
            return self._get_role_prototype(player_info.role, player_info.is_batsman)

        # 6. Fallback to average
        return self.average_embedding

    def init_adaptive_embedding(self, player_id: str, player_info: Optional[PlayerInfo] = None):
        """
        Initialize adaptive embedding for a player in current match.
        """
        base_embedding = self.get_embedding(player_id, player_info, use_adaptive=False)
        self.adaptive[player_id] = AdaptiveEmbedding(base_embedding)

    def update_adaptive(
        self,
        player_id: str,
        model: nn.Module,
        context: dict,
        actual_outcome: int,
        predicted_probs: torch.Tensor
    ):
        """
        Update adaptive embedding after ball completion.
        """
        if player_id in self.adaptive:
            self.adaptive[player_id].update(model, context, actual_outcome, predicted_probs)

    def reset_adaptive(self):
        """Reset adaptive embeddings (call at match end)."""
        self.adaptive.clear()

    def _generate_from_stats(self, player_info: PlayerInfo) -> torch.Tensor:
        if player_info.is_batsman:
            return self.generator.generate_batsman_embedding(player_info.batting_stats)
        else:
            return self.generator.generate_bowler_embedding(player_info.bowling_stats)

    def _get_role_prototype(self, role: str, is_batsman: bool) -> torch.Tensor:
        if is_batsman:
            return self.prototypes.get_batting_prototype(role)
        else:
            return self.prototypes.get_bowling_prototype(role)
```

## Data Sources for Career Stats

| Source | API Available | Data Quality |
|--------|---------------|--------------|
| ESPNCricinfo | Unofficial scraping | Excellent |
| Cricbuzz | Limited API | Good |
| Cricket Archive | Paid API | Excellent |
| Cricsheet historical | Compute from data | Good for common players |

### Fetching Stats Example

```python
async def fetch_player_stats(player_id: str) -> Optional[PlayerInfo]:
    """
    Fetch career stats from external source.
    """
    # Try ESPNCricinfo first
    try:
        stats = await espncricinfo_api.get_player_stats(player_id, format="T20")
        return PlayerInfo(
            player_id=player_id,
            batting_stats=BatsmanStats(
                t20_strike_rate=stats["batting"]["strike_rate"],
                t20_average=stats["batting"]["average"],
                matches=stats["matches"],
                # ... etc
            ),
            bowling_stats=BowlerStats(...) if stats.get("bowling") else None,
            role=infer_role(stats),
            is_batsman=stats["batting"]["innings"] > 10
        )
    except Exception:
        pass

    # Fallback: compute from Cricsheet historical data
    try:
        stats = compute_from_cricsheet(player_id)
        return stats
    except Exception:
        pass

    return None
```

## Expected Performance by Method

| Method | % of Learned Quality | When to Use |
|--------|---------------------|-------------|
| Learned embedding | 100% | Player in training data |
| Generated from full stats | ~90% | Player with good career record |
| Generated from partial stats | ~80% | Limited stats available |
| Role prototype | ~70% | Role known, no stats |
| Average embedding | ~50% | Complete unknown |
| Adapted (after 12+ balls) | ~85% | Unknown who has played enough |

## Integration with Model

```python
class CricketPredictor(nn.Module):
    def __init__(self, ...):
        # ...
        self.embedding_manager = PlayerEmbeddingManager(...)

    def forward(self, batch):
        # Get embeddings (handles cold-start automatically)
        batsman_emb = self.embedding_manager.get_embedding(batch["striker_id"])
        bowler_emb = self.embedding_manager.get_embedding(batch["bowler_id"])
        partner_emb = self.embedding_manager.get_embedding(batch["non_striker_id"])

        # Rest of forward pass...
```

---

## Venue Embeddings

Venues also need cold-start handling. Same principle: generate from features, not lookup by name.

### Venue Feature Encoder

```python
class VenueEmbeddingGenerator(nn.Module):
    """
    Generates venue embedding from venue statistics.
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def generate(self, stats: VenueStats) -> torch.Tensor:
        features = torch.tensor([
            stats.avg_first_innings_score / 180.0,
            stats.avg_second_innings_score / 170.0,
            stats.avg_first_innings_wickets / 7.0,
            stats.boundary_percentage / 15.0,
            stats.six_percentage / 5.0,
            stats.pace_wicket_percentage / 60.0,
            stats.spin_wicket_percentage / 40.0,
            stats.toss_bat_first_win_pct / 100.0,
            float(stats.is_subcontinent),
            float(stats.has_dew_factor),
        ])
        return self.encoder(features)


@dataclass
class VenueStats:
    """Statistics for venue embedding generation."""
    avg_first_innings_score: float    # Average 1st innings score
    avg_second_innings_score: float   # Average 2nd innings score
    avg_first_innings_wickets: float  # Average wickets in 1st innings
    boundary_percentage: float        # % of runs from boundaries
    six_percentage: float             # % of runs from sixes
    pace_wicket_percentage: float     # % of wickets to pace
    spin_wicket_percentage: float     # % of wickets to spin
    toss_bat_first_win_pct: float    # Win % batting first
    is_subcontinent: bool            # India/SL/Pak/BD
    has_dew_factor: bool             # Evening matches affected by dew
```

### Venue Embedding Manager

```python
class VenueEmbeddingManager:
    """
    Manages venue embeddings with cold-start handling.
    """

    # Venue type prototypes for unknown venues
    VENUE_TYPES = [
        "high_scoring_flat",      # Dubai, Bangalore
        "spin_friendly",          # Chennai, Sharjah
        "pace_friendly",          # Perth, Johannesburg
        "balanced",               # Melbourne, Lords
        "slow_low",               # Kolkata, Dhaka
    ]

    def __init__(self, generator: VenueEmbeddingGenerator, embed_dim: int = 32):
        self.generator = generator
        self.embed_dim = embed_dim
        self.cache: Dict[str, torch.Tensor] = {}
        self.type_prototypes = nn.Embedding(len(self.VENUE_TYPES), embed_dim)
        self.average_embedding = None  # Set after training

    def get_embedding(
        self,
        venue_id: str,
        venue_info: Optional[VenueInfo] = None
    ) -> torch.Tensor:
        """Get embedding for a venue."""

        # 1. Check cache
        if venue_id in self.cache:
            return self.cache[venue_id]

        # 2. Generate from stats if available
        if venue_info and venue_info.has_stats:
            embedding = self.generator.generate(venue_info.stats)
            self.cache[venue_id] = embedding
            return embedding

        # 3. Use venue type prototype
        if venue_info and venue_info.venue_type:
            idx = self.VENUE_TYPES.index(venue_info.venue_type)
            return self.type_prototypes(torch.tensor(idx))

        # 4. Infer from location
        if venue_info and venue_info.country:
            return self._infer_from_country(venue_info.country)

        # 5. Fallback to average
        return self.average_embedding

    def _infer_from_country(self, country: str) -> torch.Tensor:
        """Infer venue type from country."""
        country_to_type = {
            "India": "spin_friendly",
            "Sri Lanka": "spin_friendly",
            "Pakistan": "spin_friendly",
            "Bangladesh": "slow_low",
            "Australia": "pace_friendly",
            "South Africa": "pace_friendly",
            "New Zealand": "balanced",
            "England": "balanced",
            "UAE": "high_scoring_flat",
            "West Indies": "high_scoring_flat",
        }
        venue_type = country_to_type.get(country, "balanced")
        idx = self.VENUE_TYPES.index(venue_type)
        return self.type_prototypes(torch.tensor(idx))
```

### Computing Venue Stats from Historical Data

```python
def compute_venue_stats(venue_id: str, matches: List[Match]) -> VenueStats:
    """
    Compute venue statistics from historical matches at this venue.
    """
    venue_matches = [m for m in matches if m.venue == venue_id]

    if len(venue_matches) < 5:
        return None  # Not enough data

    first_innings_scores = [m.innings[0].total_runs for m in venue_matches]
    second_innings_scores = [m.innings[1].total_runs for m in venue_matches if len(m.innings) > 1]

    # Compute stats
    return VenueStats(
        avg_first_innings_score=np.mean(first_innings_scores),
        avg_second_innings_score=np.mean(second_innings_scores) if second_innings_scores else 150,
        avg_first_innings_wickets=np.mean([m.innings[0].wickets for m in venue_matches]),
        boundary_percentage=compute_boundary_pct(venue_matches),
        six_percentage=compute_six_pct(venue_matches),
        pace_wicket_percentage=compute_pace_wicket_pct(venue_matches),
        spin_wicket_percentage=compute_spin_wicket_pct(venue_matches),
        toss_bat_first_win_pct=compute_bat_first_win_pct(venue_matches),
        is_subcontinent=infer_subcontinent(venue_id),
        has_dew_factor=infer_dew_factor(venue_id),
    )
```

---

## Team Embeddings

Teams also need cold-start handling, though this is less common (most teams are known).

### Team Embedding Generator

```python
class TeamEmbeddingGenerator(nn.Module):
    """
    Generates team embedding from team statistics.
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def generate(self, stats: TeamStats) -> torch.Tensor:
        features = torch.tensor([
            stats.win_percentage / 100.0,
            stats.avg_score / 180.0,
            stats.avg_powerplay_score / 55.0,
            stats.avg_death_overs_score / 60.0,
            stats.bowling_avg_economy / 9.0,
            stats.chase_success_rate / 100.0,
            float(stats.is_international),
            stats.matches_played / 200.0,
        ])
        return self.encoder(features)


@dataclass
class TeamStats:
    """Statistics for team embedding generation."""
    win_percentage: float        # Overall win %
    avg_score: float             # Average batting score
    avg_powerplay_score: float   # Average powerplay score
    avg_death_overs_score: float # Average death overs score
    bowling_avg_economy: float   # Team bowling economy
    chase_success_rate: float    # % of successful chases
    is_international: bool       # National team vs franchise
    matches_played: int          # Experience level
```

---

## Summary: Unified Cold-Start Approach

| Entity | Features Used | Fallback Chain |
|--------|---------------|----------------|
| **Player** | Career stats (SR, avg, etc.) | Stats → Role → Average |
| **Venue** | Historical scores, conditions | Stats → Type → Country → Average |
| **Team** | Win %, scoring patterns | Stats → Type (international/franchise) → Average |

### The Model Never Sees IDs

```
┌────────────────────────────────────────────────────────────────────┐
│                     EMBEDDING FLOW                                  │
│                                                                     │
│  Input IDs              Feature Extraction        Embeddings        │
│  ──────────            ──────────────────        ──────────        │
│                                                                     │
│  "virat_kohli"   ──►   {sr: 137, avg: 48}   ──►   [0.23, -0.45, ...]│
│                              │                                      │
│                              ▼                                      │
│                        encoder_network()                            │
│                                                                     │
│  "new_player"    ──►   {sr: 125, avg: 28}   ──►   [0.18, -0.32, ...]│
│                              │                                      │
│                              ▼                                      │
│                    SAME encoder_network()                           │
│                                                                     │
│  The model sees only the embedding vectors, never the IDs.         │
│  Similar stats → similar embeddings → similar predictions.         │
└────────────────────────────────────────────────────────────────────┘
```

### Training vs Inference

| Phase | Known Entities | Unknown Entities |
|-------|----------------|------------------|
| **Training** | Learn encoder that maps stats → good embeddings | N/A |
| **Inference** | Use encoder with known stats | Use encoder with fetched/provided stats |
| **Live Match** | Cached embeddings | Generate on-demand + adapt during match |
