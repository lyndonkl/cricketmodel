"""Feature extraction from match state."""

import torch

from .state import MatchState
from .derived import (
    compute_phase,
    compute_pressure_index,
    compute_momentum,
    compute_batsman_setness,
    compute_bowler_threat,
    compute_partnership_stability,
)


class FeatureExtractor:
    """Extract normalized features from match state for model input."""

    # Feature dimensions for each node type
    DIMS = {
        "score_state": 5,
        "chase_state": 4,
        "phase_state": 4,  # One-hot phase + over progress
        "time_pressure": 3,
        "wicket_buffer": 2,
        "batsman_state": 6,
        "bowler_state": 6,
        "partnership": 4,
        "pressure_index": 1,
        "batting_momentum": 1,
        "bowling_momentum": 1,
        "dot_pressure": 2,
    }

    @staticmethod
    def extract_score_state(state: MatchState) -> torch.Tensor:
        """Extract score state features."""
        return torch.tensor([
            state.score / 200.0,
            state.run_rate / 12.0,
            state.balls / 120.0,
            state.over / 20.0,
            state.ball_in_over / 6.0,
        ], dtype=torch.float32)

    @staticmethod
    def extract_chase_state(state: MatchState) -> torch.Tensor:
        """Extract chase-specific features."""
        if state.target is None:
            return torch.zeros(4, dtype=torch.float32)

        runs_needed = state.target - state.score
        balls_remaining = max(120 - state.balls, 1)
        rrr = runs_needed / (balls_remaining / 6)

        return torch.tensor([
            runs_needed / 200.0,
            rrr / 15.0,
            (rrr - state.run_rate) / 10.0,  # Run rate gap
            balls_remaining / 120.0,
        ], dtype=torch.float32)

    @staticmethod
    def extract_phase_state(state: MatchState) -> torch.Tensor:
        """Extract phase features (one-hot + progress)."""
        phase = compute_phase(state.over)
        phase_onehot = [0.0, 0.0, 0.0]
        phase_onehot[phase] = 1.0

        # Progress within phase
        if phase == 0:  # Powerplay
            progress = state.over / 6
        elif phase == 1:  # Middle
            progress = (state.over - 6) / 9
        else:  # Death
            progress = (state.over - 15) / 5

        return torch.tensor(phase_onehot + [progress], dtype=torch.float32)

    @staticmethod
    def extract_time_pressure(state: MatchState) -> torch.Tensor:
        """Extract time pressure features."""
        balls_remaining = max(120 - state.balls, 0)
        overs_remaining = balls_remaining / 6

        return torch.tensor([
            balls_remaining / 120.0,
            1.0 - (balls_remaining / 120.0),  # Urgency
            min(overs_remaining, 5) / 5.0,  # Death overs indicator
        ], dtype=torch.float32)

    @staticmethod
    def extract_wicket_buffer(state: MatchState) -> torch.Tensor:
        """Extract wicket buffer features."""
        wickets_in_hand = 10 - state.wickets
        return torch.tensor([
            wickets_in_hand / 10.0,
            1.0 if wickets_in_hand <= 3 else 0.0,  # Tail exposed
        ], dtype=torch.float32)

    @staticmethod
    def extract_batsman_state(state: MatchState, player: str) -> torch.Tensor:
        """Extract batsman state features."""
        batsman = state.batsman_states.get(player)
        if batsman is None:
            return torch.zeros(6, dtype=torch.float32)

        setness = compute_batsman_setness(batsman.balls)
        return torch.tensor([
            batsman.runs / 100.0,
            batsman.balls / 60.0,
            batsman.strike_rate / 200.0,
            batsman.dot_percentage,
            setness,
            (batsman.fours + batsman.sixes) / 10.0,  # Boundary count
        ], dtype=torch.float32)

    @staticmethod
    def extract_bowler_state(state: MatchState, player: str) -> torch.Tensor:
        """Extract bowler state features."""
        bowler = state.bowler_states.get(player)
        if bowler is None:
            return torch.zeros(6, dtype=torch.float32)

        threat = compute_bowler_threat(
            bowler.economy,
            bowler.wickets,
            bowler.dots,
            bowler.balls,
        )
        return torch.tensor([
            bowler.balls / 24.0,  # Max 4 overs
            bowler.runs / 50.0,
            bowler.wickets / 4.0,
            bowler.economy / 12.0,
            bowler.dot_percentage,
            threat,
        ], dtype=torch.float32)

    @staticmethod
    def extract_partnership(state: MatchState) -> torch.Tensor:
        """Extract partnership features."""
        p = state.partnership
        stability = compute_partnership_stability(p.runs, p.balls, state.score)

        return torch.tensor([
            p.runs / 100.0,
            p.balls / 60.0,
            p.run_rate / 12.0,
            stability,
        ], dtype=torch.float32)

    @staticmethod
    def extract_pressure_index(state: MatchState) -> torch.Tensor:
        """Extract pressure index."""
        pressure = compute_pressure_index(
            state.score,
            state.wickets,
            state.balls,
            state.target,
            state.consecutive_dots,
        )
        return torch.tensor([pressure], dtype=torch.float32)

    @staticmethod
    def extract_batting_momentum(state: MatchState) -> torch.Tensor:
        """Extract batting momentum."""
        momentum = compute_momentum(state.ball_sequence, window=12)
        return torch.tensor([momentum], dtype=torch.float32)

    @staticmethod
    def extract_bowling_momentum(state: MatchState) -> torch.Tensor:
        """Extract bowling momentum (negative of batting momentum)."""
        momentum = compute_momentum(state.ball_sequence, window=12)
        return torch.tensor([-momentum], dtype=torch.float32)

    @staticmethod
    def extract_dot_pressure(state: MatchState) -> torch.Tensor:
        """Extract dot ball pressure features."""
        return torch.tensor([
            state.consecutive_dots / 6.0,
            state.last_boundary_balls_ago / 30.0,
        ], dtype=torch.float32)

    @classmethod
    def extract_all(cls, state: MatchState) -> dict[str, torch.Tensor]:
        """Extract all node features from match state."""
        return {
            "score_state": cls.extract_score_state(state),
            "chase_state": cls.extract_chase_state(state),
            "phase_state": cls.extract_phase_state(state),
            "time_pressure": cls.extract_time_pressure(state),
            "wicket_buffer": cls.extract_wicket_buffer(state),
            "striker_state": cls.extract_batsman_state(state, state.striker),
            "non_striker_state": cls.extract_batsman_state(state, state.non_striker),
            "bowler_state": cls.extract_bowler_state(state, state.bowler),
            "partnership": cls.extract_partnership(state),
            "pressure_index": cls.extract_pressure_index(state),
            "batting_momentum": cls.extract_batting_momentum(state),
            "bowling_momentum": cls.extract_bowling_momentum(state),
            "dot_pressure": cls.extract_dot_pressure(state),
        }
