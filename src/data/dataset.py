"""PyTorch Dataset for cricket ball prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .loader import Match, Innings, Delivery, load_matches_from_dir


@dataclass
class BallSample:
    """Single training sample - state before a ball and its outcome."""

    # Match context
    match_id: str
    venue: str
    batting_team: str
    bowling_team: str
    innings_num: int  # 1 or 2

    # Current ball actors
    batter: str
    bowler: str
    non_striker: str

    # Match state before this ball
    score: int
    wickets: int
    over: int
    ball_in_over: int
    target: Optional[int]  # None for first innings

    # Ball history (for temporal attention)
    ball_history: list[dict]  # Last N balls with outcomes

    # Label: outcome of this ball
    outcome: int  # 0=dot, 1=1, 2=2, 3=3, 4=4, 5=6, 6=wicket


OUTCOME_MAP = {
    (0, False): 0,  # dot
    (1, False): 1,  # single
    (2, False): 2,  # two
    (3, False): 3,  # three
    (4, False): 4,  # boundary
    (6, False): 5,  # six
}


def _delivery_to_outcome(d: Delivery) -> int:
    """Map delivery to outcome class."""
    if d.is_wicket:
        return 6  # wicket
    runs = d.runs_batter
    if runs == 0:
        return 0
    elif runs == 1:
        return 1
    elif runs == 2:
        return 2
    elif runs == 3:
        return 3
    elif runs == 4:
        return 4
    elif runs >= 6:
        return 5
    return 0


def _extract_samples_from_innings(
    match: Match,
    innings: Innings,
    innings_num: int,
    target: Optional[int],
    history_len: int,
) -> list[BallSample]:
    """Extract training samples from single innings."""
    samples = []
    score = 0
    wickets = 0

    for i, delivery in enumerate(innings.deliveries):
        # Build ball history (previous balls)
        start_idx = max(0, i - history_len)
        history = []
        for h_d in innings.deliveries[start_idx:i]:
            history.append({
                "batter": h_d.batter,
                "bowler": h_d.bowler,
                "runs": h_d.runs_total,
                "is_wicket": h_d.is_wicket,
                "over": h_d.over,
                "ball_in_over": h_d.ball_in_over,
            })

        sample = BallSample(
            match_id=match.match_id,
            venue=match.venue,
            batting_team=innings.batting_team,
            bowling_team=innings.bowling_team,
            innings_num=innings_num,
            batter=delivery.batter,
            bowler=delivery.bowler,
            non_striker=delivery.non_striker,
            score=score,
            wickets=wickets,
            over=delivery.over,
            ball_in_over=delivery.ball_in_over,
            target=target,
            ball_history=history,
            outcome=_delivery_to_outcome(delivery),
        )
        samples.append(sample)

        # Update state for next ball
        score += delivery.runs_total
        if delivery.is_wicket:
            wickets += 1

    return samples


def extract_samples_from_match(match: Match, history_len: int = 24) -> list[BallSample]:
    """Extract all training samples from a match."""
    samples = []

    for innings_num, innings in enumerate(match.innings, start=1):
        # Second innings has a target
        target = None
        if innings_num == 2 and len(match.innings) >= 1:
            target = match.innings[0].total_runs + 1

        samples.extend(
            _extract_samples_from_innings(match, innings, innings_num, target, history_len)
        )

    return samples


class CricketDataset(Dataset):
    """Dataset of ball-by-ball samples."""

    def __init__(
        self,
        data_dir: str | Path,
        history_len: int = 24,
        min_balls: int = 60,
    ):
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.samples: list[BallSample] = []

        # Build unique entity sets for embedding indices
        self.players: set[str] = set()
        self.venues: set[str] = set()
        self.teams: set[str] = set()

        self._load_data(min_balls)

        # Create index mappings
        self.player_to_idx = {p: i for i, p in enumerate(sorted(self.players))}
        self.venue_to_idx = {v: i for i, v in enumerate(sorted(self.venues))}
        self.team_to_idx = {t: i for i, t in enumerate(sorted(self.teams))}

    def _load_data(self, min_balls: int) -> None:
        """Load all matches and extract samples."""
        for match in load_matches_from_dir(self.data_dir, min_balls):
            # Collect entities
            self.venues.add(match.venue)
            self.teams.update(match.teams)
            for innings in match.innings:
                for d in innings.deliveries:
                    self.players.add(d.batter)
                    self.players.add(d.bowler)
                    self.players.add(d.non_striker)

            # Extract samples
            self.samples.extend(extract_samples_from_match(match, self.history_len))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Match state features
        total_balls = sample.over * 6 + sample.ball_in_over
        balls_remaining = 120 - total_balls  # T20 max

        state = torch.tensor([
            sample.score / 200.0,  # Normalize to typical max
            sample.wickets / 10.0,
            total_balls / 120.0,
            sample.innings_num - 1,  # 0 or 1
        ], dtype=torch.float32)

        # Chase features (only for 2nd innings)
        if sample.target is not None:
            runs_needed = sample.target - sample.score
            rrr = (runs_needed / (balls_remaining / 6)) if balls_remaining > 0 else 0
            chase = torch.tensor([
                runs_needed / 200.0,
                rrr / 15.0,  # Normalize RRR
                1.0,  # Is chase
            ], dtype=torch.float32)
        else:
            chase = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # Entity indices
        batter_idx = self.player_to_idx.get(sample.batter, 0)
        bowler_idx = self.player_to_idx.get(sample.bowler, 0)
        non_striker_idx = self.player_to_idx.get(sample.non_striker, 0)
        venue_idx = self.venue_to_idx.get(sample.venue, 0)
        batting_team_idx = self.team_to_idx.get(sample.batting_team, 0)
        bowling_team_idx = self.team_to_idx.get(sample.bowling_team, 0)

        # Ball history sequence
        history_runs = []
        history_wickets = []
        history_overs = []
        history_batters = []
        history_bowlers = []

        for h in sample.ball_history[-self.history_len:]:
            history_runs.append(h["runs"] / 6.0)
            history_wickets.append(float(h["is_wicket"]))
            history_overs.append((h["over"] * 6 + h["ball_in_over"]) / 120.0)
            history_batters.append(self.player_to_idx.get(h["batter"], 0))
            history_bowlers.append(self.player_to_idx.get(h["bowler"], 0))

        # Pad history to fixed length
        pad_len = self.history_len - len(history_runs)
        if pad_len > 0:
            history_runs = [0.0] * pad_len + history_runs
            history_wickets = [0.0] * pad_len + history_wickets
            history_overs = [0.0] * pad_len + history_overs
            history_batters = [0] * pad_len + history_batters
            history_bowlers = [0] * pad_len + history_bowlers

        return {
            "state": state,
            "chase": chase,
            "batter_idx": torch.tensor(batter_idx, dtype=torch.long),
            "bowler_idx": torch.tensor(bowler_idx, dtype=torch.long),
            "non_striker_idx": torch.tensor(non_striker_idx, dtype=torch.long),
            "venue_idx": torch.tensor(venue_idx, dtype=torch.long),
            "batting_team_idx": torch.tensor(batting_team_idx, dtype=torch.long),
            "bowling_team_idx": torch.tensor(bowling_team_idx, dtype=torch.long),
            "history_runs": torch.tensor(history_runs, dtype=torch.float32),
            "history_wickets": torch.tensor(history_wickets, dtype=torch.float32),
            "history_overs": torch.tensor(history_overs, dtype=torch.float32),
            "history_batters": torch.tensor(history_batters, dtype=torch.long),
            "history_bowlers": torch.tensor(history_bowlers, dtype=torch.long),
            "label": torch.tensor(sample.outcome, dtype=torch.long),
        }

    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def num_venues(self) -> int:
        return len(self.venues)

    @property
    def num_teams(self) -> int:
        return len(self.teams)
