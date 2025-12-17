"""Match state tracking for live predictions."""

from dataclasses import dataclass, field


@dataclass
class BatsmanState:
    """Current batsman's innings state."""

    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    dots: int = 0

    @property
    def strike_rate(self) -> float:
        return (self.runs / self.balls * 100) if self.balls > 0 else 0.0

    @property
    def dot_percentage(self) -> float:
        return (self.dots / self.balls) if self.balls > 0 else 0.0

    def update(self, runs_batter: int, is_dot: bool) -> None:
        self.balls += 1
        self.runs += runs_batter
        if runs_batter == 4:
            self.fours += 1
        elif runs_batter >= 6:
            self.sixes += 1
        if is_dot:
            self.dots += 1


@dataclass
class BowlerState:
    """Current bowler's spell state."""

    balls: int = 0
    runs: int = 0
    wickets: int = 0
    dots: int = 0
    extras: int = 0

    @property
    def overs(self) -> float:
        return self.balls // 6 + (self.balls % 6) / 10

    @property
    def economy(self) -> float:
        overs = self.balls / 6
        return (self.runs / overs) if overs > 0 else 0.0

    @property
    def dot_percentage(self) -> float:
        return (self.dots / self.balls) if self.balls > 0 else 0.0

    def update(self, runs_total: int, is_wicket: bool, extras: int) -> None:
        self.balls += 1
        self.runs += runs_total
        self.extras += extras
        if is_wicket:
            self.wickets += 1
        if runs_total == 0 and extras == 0:
            self.dots += 1


@dataclass
class Partnership:
    """Current partnership state."""

    runs: int = 0
    balls: int = 0
    striker_contribution: int = 0
    non_striker_contribution: int = 0

    @property
    def run_rate(self) -> float:
        overs = self.balls / 6
        return (self.runs / overs) if overs > 0 else 0.0

    def update(self, runs_total: int, striker_runs: int) -> None:
        self.runs += runs_total
        self.balls += 1
        self.striker_contribution += striker_runs

    def reset(self) -> None:
        self.runs = 0
        self.balls = 0
        self.striker_contribution = 0
        self.non_striker_contribution = 0


@dataclass
class MatchState:
    """Full match state for an innings."""

    # Score state
    score: int = 0
    wickets: int = 0
    balls: int = 0
    target: int | None = None

    # Per-player states
    batsman_states: dict[str, BatsmanState] = field(default_factory=dict)
    bowler_states: dict[str, BowlerState] = field(default_factory=dict)

    # Current actors
    striker: str = ""
    non_striker: str = ""
    bowler: str = ""

    # Partnership
    partnership: Partnership = field(default_factory=Partnership)

    # Ball sequence (last N balls for temporal)
    ball_sequence: list[dict] = field(default_factory=list)

    # Derived state
    consecutive_dots: int = 0
    last_boundary_balls_ago: int = 999

    @property
    def over(self) -> int:
        return self.balls // 6

    @property
    def ball_in_over(self) -> int:
        return self.balls % 6

    @property
    def run_rate(self) -> float:
        overs = self.balls / 6
        return (self.score / overs) if overs > 0 else 0.0

    @property
    def required_run_rate(self) -> float | None:
        if self.target is None:
            return None
        balls_remaining = 120 - self.balls
        if balls_remaining <= 0:
            return None
        runs_needed = self.target - self.score
        return (runs_needed / (balls_remaining / 6))

    def get_batsman_state(self, player: str) -> BatsmanState:
        if player not in self.batsman_states:
            self.batsman_states[player] = BatsmanState()
        return self.batsman_states[player]

    def get_bowler_state(self, player: str) -> BowlerState:
        if player not in self.bowler_states:
            self.bowler_states[player] = BowlerState()
        return self.bowler_states[player]

    def update(
        self,
        striker: str,
        bowler: str,
        non_striker: str,
        runs_batter: int,
        runs_extras: int,
        runs_total: int,
        is_wicket: bool,
    ) -> None:
        """Update state after a ball."""
        self.striker = striker
        self.non_striker = non_striker
        self.bowler = bowler

        # Update score
        self.score += runs_total
        self.balls += 1
        if is_wicket:
            self.wickets += 1

        # Update batsman
        is_dot = runs_total == 0
        batsman_state = self.get_batsman_state(striker)
        batsman_state.update(runs_batter, is_dot)

        # Update bowler
        bowler_state = self.get_bowler_state(bowler)
        bowler_state.update(runs_total, is_wicket, runs_extras)

        # Update partnership
        self.partnership.update(runs_total, runs_batter)

        # Reset partnership on wicket
        if is_wicket:
            self.partnership.reset()

        # Track consecutive dots
        if runs_total == 0:
            self.consecutive_dots += 1
        else:
            self.consecutive_dots = 0

        # Track last boundary
        if runs_batter >= 4:
            self.last_boundary_balls_ago = 0
        else:
            self.last_boundary_balls_ago += 1

        # Add to ball sequence
        self.ball_sequence.append({
            "striker": striker,
            "bowler": bowler,
            "runs_batter": runs_batter,
            "runs_total": runs_total,
            "is_wicket": is_wicket,
            "over": self.over,
            "ball_in_over": self.ball_in_over,
        })

    def to_feature_dict(self) -> dict[str, float]:
        """Export current state as feature dictionary."""
        features = {
            # Score state
            "score": self.score,
            "wickets": self.wickets,
            "balls": self.balls,
            "over": self.over,
            "ball_in_over": self.ball_in_over,
            "run_rate": self.run_rate,

            # Chase state
            "is_chase": float(self.target is not None),
            "target": self.target or 0,
            "runs_needed": (self.target - self.score) if self.target else 0,
            "required_run_rate": self.required_run_rate or 0,

            # Partnership
            "partnership_runs": self.partnership.runs,
            "partnership_balls": self.partnership.balls,

            # Pressure indicators
            "consecutive_dots": self.consecutive_dots,
            "last_boundary_balls_ago": min(self.last_boundary_balls_ago, 30),
            "wickets_in_hand": 10 - self.wickets,
        }

        # Striker state
        striker_state = self.batsman_states.get(self.striker)
        if striker_state:
            features["striker_runs"] = striker_state.runs
            features["striker_balls"] = striker_state.balls
            features["striker_sr"] = striker_state.strike_rate
            features["striker_dots"] = striker_state.dots

        # Bowler state
        bowler_state = self.bowler_states.get(self.bowler)
        if bowler_state:
            features["bowler_balls"] = bowler_state.balls
            features["bowler_runs"] = bowler_state.runs
            features["bowler_wickets"] = bowler_state.wickets
            features["bowler_economy"] = bowler_state.economy

        return features
