"""Full cricket prediction model combining hierarchical GAT and Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import EmbeddingManager
from .hierarchical import HierarchicalGAT
from .temporal import TemporalTransformer


class CricketPredictor(nn.Module):
    """
    Full model for cricket ball prediction.

    Combines:
    1. Entity embeddings (players, venues, teams)
    2. Hierarchical GAT for within-ball graph attention (17 nodes, 4 layers)
    3. Temporal Transformer for cross-ball attention (specialized heads)
    4. Fusion layer for final prediction
    """

    def __init__(
        self,
        num_players: int,
        num_venues: int,
        num_teams: int,
        num_outcomes: int = 7,
        hidden_dim: int = 128,
        gat_heads: int = 4,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 24,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_outcomes = num_outcomes
        self.seq_len = seq_len

        # Embedding manager
        self.embeddings = EmbeddingManager(
            num_players=num_players,
            num_venues=num_venues,
            num_teams=num_teams,
            player_dim=64,
            venue_dim=32,
            team_dim=32,
        )

        # Node feature projections to hidden_dim
        self.node_projections = nn.ModuleDict({
            # Global context (Layer 1)
            "venue": nn.Linear(32, hidden_dim),
            "batting_team": nn.Linear(32, hidden_dim),
            "bowling_team": nn.Linear(32, hidden_dim),
            # Match state (Layer 2)
            "score_state": nn.Linear(4, hidden_dim),
            "chase_state": nn.Linear(3, hidden_dim),
            "phase_state": nn.Linear(4, hidden_dim),
            "time_pressure": nn.Linear(3, hidden_dim),
            "wicket_buffer": nn.Linear(2, hidden_dim),
            # Actor (Layer 3)
            "striker_identity": nn.Linear(64, hidden_dim),
            "striker_state": nn.Linear(6, hidden_dim),
            "bowler_identity": nn.Linear(64, hidden_dim),
            "bowler_state": nn.Linear(6, hidden_dim),
            "partnership": nn.Linear(4, hidden_dim),
            # Dynamics (Layer 4)
            "batting_momentum": nn.Linear(1, hidden_dim),
            "bowling_momentum": nn.Linear(1, hidden_dim),
            "pressure_index": nn.Linear(1, hidden_dim),
            "dot_pressure": nn.Linear(2, hidden_dim),
        })

        # Hierarchical GAT (layer-wise attention with conditioning)
        self.gat = HierarchicalGAT(
            hidden_dim=hidden_dim,
            num_heads=gat_heads,
            dropout=dropout,
        )

        # Temporal Transformer (specialized attention heads)
        self.temporal = TemporalTransformer(
            num_players=num_players,
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
            max_seq_len=seq_len,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output head
        self.output_head = nn.Linear(hidden_dim // 2, num_outcomes)

    def _build_node_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build all 17 node features from batch."""
        # Get embeddings
        emb = self.embeddings(
            batter_idx=batch["batter_idx"],
            bowler_idx=batch["bowler_idx"],
            non_striker_idx=batch["non_striker_idx"],
            venue_idx=batch["venue_idx"],
            batting_team_idx=batch["batting_team_idx"],
            bowling_team_idx=batch["bowling_team_idx"],
        )

        nodes = {}

        # Layer 1: Global Context
        nodes["venue"] = self.node_projections["venue"](emb["venue"])
        nodes["batting_team"] = self.node_projections["batting_team"](emb["batting_team"])
        nodes["bowling_team"] = self.node_projections["bowling_team"](emb["bowling_team"])

        # Layer 2: Match State
        nodes["score_state"] = self.node_projections["score_state"](batch["state"])
        nodes["chase_state"] = self.node_projections["chase_state"](batch["chase"])

        # Phase state
        over_progress = batch["state"][:, 2:3]
        phase_onehot = self._get_phase_onehot(over_progress)
        phase_features = torch.cat([phase_onehot, over_progress], dim=-1)
        nodes["phase_state"] = self.node_projections["phase_state"](phase_features)

        # Time pressure
        balls_remaining = 1.0 - batch["state"][:, 2:3]
        time_pressure = torch.cat([
            balls_remaining,
            1.0 - balls_remaining,
            (balls_remaining < 0.25).float(),
        ], dim=-1)
        nodes["time_pressure"] = self.node_projections["time_pressure"](time_pressure)

        # Wicket buffer
        wickets = batch["state"][:, 1:2] * 10
        wicket_buffer = torch.cat([
            1.0 - wickets / 10,
            (wickets > 0.7).float(),
        ], dim=-1)
        nodes["wicket_buffer"] = self.node_projections["wicket_buffer"](wicket_buffer)

        # Layer 3: Actor
        nodes["striker_identity"] = self.node_projections["striker_identity"](emb["batter"])
        nodes["striker_state"] = self.node_projections["striker_state"](
            self._compute_batsman_state(batch)
        )
        nodes["bowler_identity"] = self.node_projections["bowler_identity"](emb["bowler"])
        nodes["bowler_state"] = self.node_projections["bowler_state"](
            self._compute_bowler_state(batch)
        )
        nodes["partnership"] = self.node_projections["partnership"](
            self._compute_partnership(batch)
        )

        # Layer 4: Dynamics
        momentum = self._compute_momentum(batch)
        nodes["batting_momentum"] = self.node_projections["batting_momentum"](momentum)
        nodes["bowling_momentum"] = self.node_projections["bowling_momentum"](-momentum)
        nodes["pressure_index"] = self.node_projections["pressure_index"](
            self._compute_pressure(batch)
        )
        nodes["dot_pressure"] = self.node_projections["dot_pressure"](
            self._compute_dot_pressure(batch)
        )

        return nodes

    def _get_phase_onehot(self, over_progress: torch.Tensor) -> torch.Tensor:
        """Get phase one-hot from over progress."""
        over = over_progress * 20
        batch_size = over_progress.shape[0]
        phase = torch.zeros(batch_size, 3, device=over_progress.device)
        phase[:, 0] = (over < 6).float().squeeze(-1)
        phase[:, 1] = ((over >= 6) & (over < 15)).float().squeeze(-1)
        phase[:, 2] = (over >= 15).float().squeeze(-1)
        return phase

    def _compute_batsman_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute batsman state features from history."""
        history_runs = batch["history_runs"]
        runs = history_runs.sum(dim=-1, keepdim=True) * 6
        balls = (history_runs > -1).float().sum(dim=-1, keepdim=True)
        sr = runs / balls.clamp(min=1) * 100
        dots = (history_runs == 0).float().sum(dim=-1, keepdim=True)
        boundaries = (history_runs >= 4/6).float().sum(dim=-1, keepdim=True)
        setness = (balls / 30).clamp(max=1)
        return torch.cat([
            runs / 100, balls / 60, sr / 200,
            dots / balls.clamp(min=1), setness, boundaries / 10,
        ], dim=-1)

    def _compute_bowler_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute bowler state features from history."""
        history_runs = batch["history_runs"]
        history_wickets = batch["history_wickets"]
        runs = history_runs.sum(dim=-1, keepdim=True) * 6
        balls = (history_runs > -1).float().sum(dim=-1, keepdim=True)
        wickets = history_wickets.sum(dim=-1, keepdim=True)
        dots = (history_runs == 0).float().sum(dim=-1, keepdim=True)
        overs = balls / 6
        economy = runs / overs.clamp(min=0.1)
        threat = 0.5 * (1 - economy / 12) + 0.3 * (wickets / 4) + 0.2 * (dots / balls.clamp(min=1))
        return torch.cat([
            balls / 24, runs / 50, wickets / 4,
            economy / 12, dots / balls.clamp(min=1), threat,
        ], dim=-1)

    def _compute_partnership(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute partnership features."""
        history_runs = batch["history_runs"]
        recent = history_runs[:, -12:]
        partnership_runs = recent.sum(dim=-1, keepdim=True) * 6
        partnership_balls = (recent > -1).float().sum(dim=-1, keepdim=True)
        partnership_rr = partnership_runs / partnership_balls.clamp(min=0.1) * 6
        stability = (partnership_balls / 30).clamp(max=1)
        return torch.cat([
            partnership_runs / 100, partnership_balls / 60,
            partnership_rr / 12, stability,
        ], dim=-1)

    def _compute_momentum(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute batting momentum."""
        recent = batch["history_runs"][:, -12:]
        total_runs = recent.sum(dim=-1, keepdim=True) * 6
        momentum = (total_runs / 48) * 2 - 1
        return momentum.clamp(-1, 1)

    def _compute_pressure(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute pressure index."""
        state = batch["state"]
        chase = batch["chase"]
        wickets = state[:, 1:2]
        balls_progress = state[:, 2:3]
        wicket_pressure = wickets * 0.3
        is_chase = chase[:, 2:3]
        rrr_gap = chase[:, 1:2]
        chase_pressure = is_chase * rrr_gap.clamp(min=0) * 0.4
        stage_pressure = (balls_progress > 0.75).float() * 0.1
        return (wicket_pressure + chase_pressure + stage_pressure).clamp(max=1)

    def _compute_dot_pressure(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute dot ball pressure."""
        history_runs = batch["history_runs"]
        # Count consecutive dots at end
        recent = history_runs[:, -6:]
        consecutive = (recent == 0).float().sum(dim=-1, keepdim=True)
        # Last boundary approximation
        boundaries = (history_runs >= 4/6)
        last_boundary = (boundaries.flip(dims=[1]).cumsum(dim=1) == 0).float().sum(dim=-1, keepdim=True)
        return torch.cat([consecutive / 6, last_boundary / 30], dim=-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dict with all input tensors
            return_attention: Whether to return attention weights

        Returns:
            Dict with logits, probs, and optional attention weights
        """
        # Build hierarchical node features
        nodes = self._build_node_features(batch)

        # Hierarchical GAT forward
        gat_out, gat_attn = self.gat(nodes, return_attention)

        # Temporal Transformer forward
        temporal_out, temporal_attn = self.temporal(
            batch["history_runs"],
            batch["history_wickets"],
            batch["history_overs"],
            batch["history_batters"],
            batch["history_bowlers"],
            return_attention,
        )

        # Fusion
        combined = torch.cat([gat_out, temporal_out], dim=-1)
        fused = self.fusion(combined)

        # Output
        logits = self.output_head(fused)
        probs = F.softmax(logits, dim=-1)

        output = {"logits": logits, "probs": probs}

        if return_attention:
            output["gat_attention"] = gat_attn
            output["temporal_attention"] = temporal_attn

        return output

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Get prediction with probabilities."""
        output = self.forward(batch, return_attention=False)
        output["predicted"] = output["probs"].argmax(dim=-1)
        return output
