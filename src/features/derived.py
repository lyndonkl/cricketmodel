"""Derived features that attention cannot learn efficiently."""

from enum import IntEnum
from typing import Sequence


class Phase(IntEnum):
    """Match phase for T20."""

    POWERPLAY = 0  # Overs 1-6
    MIDDLE = 1  # Overs 7-15
    DEATH = 2  # Overs 16-20


def compute_phase(over: int) -> Phase:
    """Determine match phase from over number."""
    if over < 6:
        return Phase.POWERPLAY
    elif over < 15:
        return Phase.MIDDLE
    return Phase.DEATH


def compute_pressure_index(
    score: int,
    wickets: int,
    balls: int,
    target: int | None,
    consecutive_dots: int = 0,
) -> float:
    """
    Compute pressure index (0-1).

    Combines:
    - Wickets lost (more wickets = more pressure)
    - Run rate gap (behind required = more pressure)
    - Dot ball buildup
    - Match stage
    """
    # Base pressure from wickets (0-0.3)
    wicket_pressure = wickets / 10 * 0.3

    # Run rate pressure (0-0.4)
    rate_pressure = 0.0
    if target is not None and balls > 0:
        balls_remaining = 120 - balls
        if balls_remaining > 0:
            runs_needed = target - score
            rrr = runs_needed / (balls_remaining / 6)
            crr = score / (balls / 6)
            # Pressure increases when RRR exceeds CRR
            rate_gap = (rrr - crr) / 10  # Normalize
            rate_pressure = min(max(rate_gap, 0), 1) * 0.4

    # Dot ball pressure (0-0.2)
    dot_pressure = min(consecutive_dots / 6, 1) * 0.2

    # Stage pressure (0-0.1) - more pressure in death overs
    over = balls // 6
    stage_pressure = 0.0
    if over >= 15:
        stage_pressure = 0.1
    elif over >= 10:
        stage_pressure = 0.05

    return min(wicket_pressure + rate_pressure + dot_pressure + stage_pressure, 1.0)


def compute_momentum(
    ball_sequence: Sequence[dict],
    window: int = 12,
) -> float:
    """
    Compute batting momentum from recent balls (-1 to 1).

    Positive = batsman dominated
    Negative = bowler dominated
    """
    if not ball_sequence:
        return 0.0

    recent = list(ball_sequence)[-window:]
    if not recent:
        return 0.0

    # Scoring rate component
    total_runs = sum(b.get("runs_total", 0) for b in recent)
    max_runs = window * 4  # 4 per ball is very aggressive
    scoring_momentum = (total_runs / max_runs) * 2 - 1  # Scale to -1 to 1

    # Boundary component
    boundaries = sum(1 for b in recent if b.get("runs_batter", 0) >= 4)
    boundary_boost = boundaries / window * 0.5

    # Wicket penalty
    wickets = sum(1 for b in recent if b.get("is_wicket", False))
    wicket_penalty = wickets * 0.3

    momentum = scoring_momentum + boundary_boost - wicket_penalty
    return max(min(momentum, 1.0), -1.0)


def compute_batsman_setness(balls_faced: int) -> float:
    """
    Compute how 'set' a batsman is (0-1).

    Based on typical T20 batting patterns:
    - 0-10 balls: Getting eye in
    - 10-25 balls: Building
    - 25+ balls: Fully set
    """
    if balls_faced <= 0:
        return 0.0
    if balls_faced >= 30:
        return 1.0
    # Sigmoid-like curve
    return 1.0 - (1.0 / (1.0 + balls_faced / 10))


def compute_bowler_threat(
    bowler_economy: float,
    bowler_wickets: int,
    bowler_dots: int,
    bowler_balls: int,
) -> float:
    """
    Compute bowler threat level (0-1).

    High threat = low economy, wicket-taking ability.
    """
    if bowler_balls <= 0:
        return 0.5  # Unknown bowler

    # Economy component (lower is better)
    # T20 economy 6-8 is average
    economy_score = max(0, 1 - (bowler_economy - 4) / 10)

    # Wicket-taking
    wicket_rate = bowler_wickets / (bowler_balls / 6) if bowler_balls > 0 else 0
    wicket_score = min(wicket_rate / 2, 1)  # 2 wickets/over max

    # Dot ball pressure
    dot_rate = bowler_dots / bowler_balls if bowler_balls > 0 else 0
    dot_score = dot_rate

    return (economy_score * 0.4 + wicket_score * 0.3 + dot_score * 0.3)


def compute_partnership_stability(
    partnership_runs: int,
    partnership_balls: int,
    team_score: int,
) -> float:
    """
    Compute partnership stability (0-1).

    Higher = more stable, established partnership.
    """
    if partnership_balls <= 0:
        return 0.0

    # Ball survival component
    survival = min(partnership_balls / 30, 1) * 0.5

    # Run contribution component
    contribution = 0.0
    if team_score > 0:
        contribution = min(partnership_runs / team_score, 1) * 0.3

    # Run rate component
    partnership_rr = partnership_runs / (partnership_balls / 6) if partnership_balls > 0 else 0
    rr_score = min(partnership_rr / 8, 1) * 0.2  # 8 RPO is good partnership rate

    return survival + contribution + rr_score
