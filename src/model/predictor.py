"""Full cricket prediction model combining GAT and Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import EmbeddingManager
from .hierarchical import HierarchicalGAT
from .temporal import TemporalTransformer
from .graph import NUM_NODES, build_edge_index


class CricketPredictor(nn.Module):
    """
    Full model for cricket ball prediction.

    Combines:
    1. Entity embeddings (players, venues, teams)
    2. Hierarchical GAT for within-ball graph attention
    3. Temporal Transformer for cross-ball attention
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
        gat_layers: int = 3,
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
            # Global context
            "venue": nn.Linear(32, hidden_dim),
            "batting_team": nn.Linear(32, hidden_dim),
            "bowling_team": nn.Linear(32, hidden_dim),
            # Match state
            "score_state": nn.Linear(4, hidden_dim),  # state from dataset
            "chase_state": nn.Linear(3, hidden_dim),
            "phase_state": nn.Linear(4, hidden_dim),
            "time_pressure": nn.Linear(3, hidden_dim),
            "wicket_buffer": nn.Linear(2, hidden_dim),
            # Actor - identities are embeddings, states are features
            "striker_identity": nn.Linear(64, hidden_dim),
            "striker_state": nn.Linear(6, hidden_dim),
            "bowler_identity": nn.Linear(64, hidden_dim),
            "bowler_state": nn.Linear(6, hidden_dim),
            "partnership": nn.Linear(4, hidden_dim),
            # Dynamics
            "batting_momentum": nn.Linear(1, hidden_dim),
            "bowling_momentum": nn.Linear(1, hidden_dim),
            "pressure_index": nn.Linear(1, hidden_dim),
            "dot_pressure": nn.Linear(2, hidden_dim),
        })

        # Hierarchical GAT
        self.gat = HierarchicalGAT(
            hidden_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        # Temporal Transformer
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

        # Register edge index buffer
        self.register_buffer("edge_index", build_edge_index())

    def _build_graph_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build node features for graph from batch."""
        batch_size = batch["state"].shape[0]

        # Get embeddings
        emb = self.embeddings(
            batter_idx=batch["batter_idx"],
            bowler_idx=batch["bowler_idx"],
            non_striker_idx=batch["non_striker_idx"],
            venue_idx=batch["venue_idx"],
            batting_team_idx=batch["batting_team_idx"],
            bowling_team_idx=batch["bowling_team_idx"],
        )

        # Build node features for each node type
        # We need to project each to hidden_dim and stack

        nodes = []

        # Global context (Layer 1)
        nodes.append(self.node_projections["venue"](emb["venue"]))
        nodes.append(self.node_projections["batting_team"](emb["batting_team"]))
        nodes.append(self.node_projections["bowling_team"](emb["bowling_team"]))

        # Match state (Layer 2)
        nodes.append(self.node_projections["score_state"](batch["state"]))
        nodes.append(self.node_projections["chase_state"](batch["chase"]))

        # Phase state - derive from state
        over_progress = batch["state"][:, 2:3]  # balls/120
        phase_onehot = self._get_phase_onehot(over_progress)
        phase_features = torch.cat([phase_onehot, over_progress], dim=-1)
        nodes.append(self.node_projections["phase_state"](phase_features))

        # Time pressure
        balls_remaining = 1.0 - batch["state"][:, 2:3]
        time_pressure = torch.cat([
            balls_remaining,
            1.0 - balls_remaining,  # Urgency
            (balls_remaining < 0.25).float(),  # Death overs indicator
        ], dim=-1)
        nodes.append(self.node_projections["time_pressure"](time_pressure))

        # Wicket buffer
        wickets = batch["state"][:, 1:2] * 10  # Denormalize
        wicket_buffer = torch.cat([
            1.0 - wickets / 10,
            (wickets > 0.7).float(),  # Tail exposed
        ], dim=-1)
        nodes.append(self.node_projections["wicket_buffer"](wicket_buffer))

        # Actor layer (Layer 3)
        nodes.append(self.node_projections["striker_identity"](emb["batter"]))
        # Striker state - simplified from history
        striker_state = self._compute_batsman_state(batch)
        nodes.append(self.node_projections["striker_state"](striker_state))

        nodes.append(self.node_projections["bowler_identity"](emb["bowler"]))
        bowler_state = self._compute_bowler_state(batch)
        nodes.append(self.node_projections["bowler_state"](bowler_state))

        # Partnership - simplified
        partnership = self._compute_partnership(batch)
        nodes.append(self.node_projections["partnership"](partnership))

        # Dynamics layer (Layer 4)
        momentum = self._compute_momentum(batch)
        nodes.append(self.node_projections["batting_momentum"](momentum))
        nodes.append(self.node_projections["bowling_momentum"](-momentum))

        pressure = self._compute_pressure(batch)
        nodes.append(self.node_projections["pressure_index"](pressure))

        dot_pressure = self._compute_dot_pressure(batch)
        nodes.append(self.node_projections["dot_pressure"](dot_pressure))

        # Stack nodes: [batch, num_nodes, hidden_dim]
        x = torch.stack(nodes, dim=1)

        return x

    def _get_phase_onehot(self, over_progress: torch.Tensor) -> torch.Tensor:
        """Get phase one-hot from over progress."""
        batch_size = over_progress.shape[0]
        over = over_progress * 20  # Convert back to over number

        phase = torch.zeros(batch_size, 3, device=over_progress.device)
        phase[:, 0] = (over < 6).float().squeeze(-1)  # Powerplay
        phase[:, 1] = ((over >= 6) & (over < 15)).float().squeeze(-1)  # Middle
        phase[:, 2] = (over >= 15).float().squeeze(-1)  # Death

        return phase

    def _compute_batsman_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute batsman state features from history."""
        # Simplified: aggregate from ball history
        history_runs = batch["history_runs"]  # [batch, seq]
        history_wickets = batch["history_wickets"]

        # Compute stats for current batsman
        runs = history_runs.sum(dim=-1, keepdim=True) * 6  # Denormalize
        balls = (history_runs > -1).float().sum(dim=-1, keepdim=True)  # Non-padding
        sr = runs / balls.clamp(min=1) * 100
        dots = (history_runs == 0).float().sum(dim=-1, keepdim=True)
        boundaries = (history_runs >= 4/6).float().sum(dim=-1, keepdim=True)

        # Setness approximation
        setness = (balls / 30).clamp(max=1)

        return torch.cat([
            runs / 100,
            balls / 60,
            sr / 200,
            dots / balls.clamp(min=1),
            setness,
            boundaries / 10,
        ], dim=-1)

    def _compute_bowler_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute bowler state features from history."""
        history_runs = batch["history_runs"]
        history_wickets = batch["history_wickets"]

        # Aggregate bowler stats
        runs = history_runs.sum(dim=-1, keepdim=True) * 6
        balls = (history_runs > -1).float().sum(dim=-1, keepdim=True)
        wickets = history_wickets.sum(dim=-1, keepdim=True)
        dots = (history_runs == 0).float().sum(dim=-1, keepdim=True)

        overs = balls / 6
        economy = runs / overs.clamp(min=0.1)

        threat = 0.5 * (1 - economy / 12) + 0.3 * (wickets / 4) + 0.2 * (dots / balls.clamp(min=1))

        return torch.cat([
            balls / 24,
            runs / 50,
            wickets / 4,
            economy / 12,
            dots / balls.clamp(min=1),
            threat,
        ], dim=-1)

    def _compute_partnership(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute partnership features."""
        # Simplified partnership computation
        history_runs = batch["history_runs"]

        # Recent runs as partnership proxy
        recent = history_runs[:, -12:]
        partnership_runs = recent.sum(dim=-1, keepdim=True) * 6
        partnership_balls = (recent > -1).float().sum(dim=-1, keepdim=True)
        partnership_rr = partnership_runs / partnership_balls.clamp(min=0.1) * 6

        stability = (partnership_balls / 30).clamp(max=1)

        return torch.cat([
            partnership_runs / 100,
            partnership_balls / 60,
            partnership_rr / 12,
            stability,
        ], dim=-1)

    def _compute_momentum(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute batting momentum."""
        history_runs = batch["history_runs"]
        recent = history_runs[:, -12:]

        total_runs = recent.sum(dim=-1, keepdim=True) * 6
        max_runs = 12 * 4  # 4 per ball aggressive

        momentum = (total_runs / max_runs) * 2 - 1
        return momentum.clamp(-1, 1)

    def _compute_pressure(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute pressure index."""
        state = batch["state"]
        chase = batch["chase"]

        wickets = state[:, 1:2]
        balls_progress = state[:, 2:3]

        # Wicket pressure
        wicket_pressure = wickets * 0.3

        # Chase pressure (if applicable)
        is_chase = chase[:, 2:3]
        rrr_gap = chase[:, 1:2]  # Already normalized RRR
        chase_pressure = is_chase * rrr_gap.clamp(min=0) * 0.4

        # Stage pressure
        stage_pressure = (balls_progress > 0.75).float() * 0.1

        return (wicket_pressure + chase_pressure + stage_pressure).clamp(max=1)

    def _compute_dot_pressure(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute dot ball pressure."""
        history_runs = batch["history_runs"]

        # Count consecutive dots at end
        consecutive = torch.zeros(history_runs.shape[0], 1, device=history_runs.device)
        for i in range(history_runs.shape[1] - 1, -1, -1):
            is_dot = (history_runs[:, i] == 0).float()
            consecutive = consecutive + is_dot
            # Break on non-dot (simplified: just count recent)
            if i < history_runs.shape[1] - 6:
                break

        # Last boundary
        boundaries = (history_runs >= 4/6)
        last_boundary = torch.zeros(history_runs.shape[0], 1, device=history_runs.device)
        for i in range(history_runs.shape[1] - 1, -1, -1):
            mask = boundaries[:, i] & (last_boundary == 0).squeeze(-1)
            last_boundary[mask] = history_runs.shape[1] - i

        return torch.cat([
            consecutive / 6,
            last_boundary / 30,
        ], dim=-1)

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
            Dict with:
                - logits: [batch, num_outcomes]
                - probs: [batch, num_outcomes]
                - gat_attention: Optional attention from GAT
                - temporal_attention: Optional attention from Transformer
        """
        batch_size = batch["state"].shape[0]

        # Build graph features
        graph_x = self._build_graph_features(batch)  # [batch, 17, hidden]

        # Reshape for GAT: [batch * 17, hidden]
        graph_x_flat = graph_x.view(-1, self.hidden_dim)

        # Build batched edge index
        edge_index_batch = self._batch_edge_index(batch_size)

        # Batch tensor for pooling
        batch_idx = torch.arange(batch_size, device=graph_x.device)
        batch_idx = batch_idx.repeat_interleave(NUM_NODES)

        # GAT forward
        gat_out, gat_attn = self.gat(
            graph_x_flat,
            edge_index_batch,
            batch_idx,
            return_attention,
        )  # [batch, hidden]

        # Temporal Transformer forward
        temporal_out, temporal_attn = self.temporal(
            batch["history_runs"],
            batch["history_wickets"],
            batch["history_overs"],
            batch["history_batters"],
            batch["history_bowlers"],
            return_attention,
        )  # [batch, hidden]

        # Fusion
        combined = torch.cat([gat_out, temporal_out], dim=-1)
        fused = self.fusion(combined)

        # Output
        logits = self.output_head(fused)
        probs = F.softmax(logits, dim=-1)

        output = {
            "logits": logits,
            "probs": probs,
        }

        if return_attention:
            output["gat_attention"] = gat_attn
            output["temporal_attention"] = temporal_attn

        return output

    def _batch_edge_index(self, batch_size: int) -> torch.Tensor:
        """Create batched edge index."""
        edge_index = self.edge_index
        edge_list = []

        for i in range(batch_size):
            offset = i * NUM_NODES
            edge_list.append(edge_index + offset)

        return torch.cat(edge_list, dim=1)

    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Get prediction with probabilities."""
        output = self.forward(batch, return_attention=False)
        output["predicted"] = output["probs"].argmax(dim=-1)
        return output
