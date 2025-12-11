"""Hierarchical Graph Attention Network with layer-wise attention.

Implements the architecture from 03-hierarchical-attention.md:
- Global Context Layer (venue, team, match importance)
- Match State Layer (score, chase, phase, time, wicket)
- Actor Layer (batsman identity/state, bowler identity/state, partnership)
- Dynamics Layer (batting momentum, bowling momentum, pressure, dot pressure)

Each layer attends to itself AND receives conditioning from the layer above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GlobalContextAttention(nn.Module):
    """
    Layer 1: Aggregate venue, team, and match importance.

    Outputs h_global which conditions all lower layers.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        venue: torch.Tensor,
        batting_team: torch.Tensor,
        bowling_team: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            venue: [batch, hidden_dim]
            batting_team: [batch, hidden_dim]
            bowling_team: [batch, hidden_dim]

        Returns:
            h_global: [batch, hidden_dim]
            attn_weights: [batch, 1, 3] attention over global nodes
        """
        batch_size = venue.shape[0]

        # Stack global nodes: [batch, 3, hidden_dim]
        global_nodes = torch.stack([venue, batting_team, bowling_team], dim=1)

        # Query with learned query token
        query = self.global_query.expand(batch_size, -1, -1)
        h_global, attn_weights = self.attention(query, global_nodes, global_nodes)

        h_global = self.norm(h_global.squeeze(1))
        return h_global, attn_weights.squeeze(1)


class MatchStateAttention(nn.Module):
    """
    Layer 2: Match state nodes attend to each other and global context.

    Receives h_global as conditioning.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        # Self-attention within layer
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        # Cross-attention to global context
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output query for pooling
        self.state_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        score_state: torch.Tensor,
        chase_state: torch.Tensor,
        phase_state: torch.Tensor,
        time_pressure: torch.Tensor,
        wicket_buffer: torch.Tensor,
        h_global: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            h_state: [batch, hidden_dim]
            attn_dict: attention weights for interpretability
        """
        batch_size = score_state.shape[0]

        # Stack state nodes: [batch, 5, hidden_dim]
        state_nodes = torch.stack([
            score_state, chase_state, phase_state, time_pressure, wicket_buffer
        ], dim=1)

        # Self-attention
        h_self, self_attn = self.self_attn(state_nodes, state_nodes, state_nodes)
        h_self = self.norm1(h_self + state_nodes)

        # Cross-attention to global context
        h_global_kv = h_global.unsqueeze(1)  # [batch, 1, hidden]
        h_cross, cross_attn = self.cross_attn(h_self, h_global_kv, h_global_kv)
        h_cross = self.norm2(h_cross + h_self)

        # Pool to single representation using query
        query = self.state_query.expand(batch_size, -1, -1)
        h_state, pool_attn = self.self_attn(query, h_cross, h_cross)
        h_state = h_state.squeeze(1)

        return h_state, {
            "self_attn": self_attn,
            "cross_attn": cross_attn,
            "pool_attn": pool_attn.squeeze(1),
        }


class ActorLayerGAT(nn.Module):
    """
    Layer 3: Actor nodes with graph attention.

    Uses GAT for matchup modeling, conditioned on match state.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATv2Conv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Actor edge index (fixed connectivity)
        # 0: striker_identity, 1: striker_state, 2: bowler_identity,
        # 3: bowler_state, 4: partnership
        edges = [
            [0, 1], [1, 0],  # striker identity <-> state
            [2, 3], [3, 2],  # bowler identity <-> state
            [0, 2], [2, 0],  # matchup: striker <-> bowler
            [1, 4], [4, 1],  # striker state <-> partnership
            [3, 4], [4, 3],  # bowler state <-> partnership (indirect)
        ]
        self.register_buffer(
            "edge_index",
            torch.tensor(edges, dtype=torch.long).t().contiguous()
        )

    def forward(
        self,
        striker_identity: torch.Tensor,
        striker_state: torch.Tensor,
        bowler_identity: torch.Tensor,
        bowler_state: torch.Tensor,
        partnership: torch.Tensor,
        h_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            h_actor: [batch, hidden_dim]
            attn_weights: GAT attention weights
        """
        batch_size = striker_identity.shape[0]

        # Condition on match state
        context = self.context_proj(h_state)

        # Stack actor nodes and add context: [batch, 5, hidden]
        actor_nodes = torch.stack([
            striker_identity, striker_state, bowler_identity,
            bowler_state, partnership
        ], dim=1)
        actor_nodes = actor_nodes + context.unsqueeze(1)

        # Process each batch item through GAT
        outputs = []
        all_attn = []

        for i in range(batch_size):
            x = actor_nodes[i]  # [5, hidden]
            out, (_, attn) = self.gat(
                x, self.edge_index, return_attention_weights=True
            )
            outputs.append(out.mean(dim=0))  # Pool nodes
            all_attn.append(attn)

        h_actor = torch.stack(outputs)
        h_actor = self.norm(h_actor)

        return h_actor, all_attn[0] if all_attn else None


class DynamicsAttention(nn.Module):
    """
    Layer 4: Dynamics nodes attend to each other, conditioned on actors.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.actor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        batting_momentum: torch.Tensor,
        bowling_momentum: torch.Tensor,
        pressure_index: torch.Tensor,
        dot_pressure: torch.Tensor,
        h_actor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_dynamics: [batch, hidden_dim]
            attn_weights: [batch, 4, 4]
        """
        # Stack dynamics nodes: [batch, 4, hidden]
        dynamics_nodes = torch.stack([
            batting_momentum, bowling_momentum, pressure_index, dot_pressure
        ], dim=1)

        # Condition on actor representation
        actor_context = self.actor_proj(h_actor).unsqueeze(1)
        dynamics_conditioned = dynamics_nodes + actor_context

        # Self-attention
        h_dyn, attn_weights = self.attention(
            dynamics_conditioned, dynamics_conditioned, dynamics_conditioned
        )

        h_dynamics = self.norm(h_dyn.mean(dim=1))
        return h_dynamics, attn_weights


class HierarchicalGAT(nn.Module):
    """
    Full hierarchical attention following docs architecture.

    Information flows: Global -> State -> Actor -> Dynamics
    Each layer is conditioned on the layer above.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer modules
        self.global_attn = GlobalContextAttention(hidden_dim, num_heads)
        self.state_attn = MatchStateAttention(hidden_dim, num_heads)
        self.actor_attn = ActorLayerGAT(hidden_dim, num_heads, dropout)
        self.dynamics_attn = DynamicsAttention(hidden_dim, num_heads)

        # Fusion: combine all layer outputs
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Layer importance (learnable)
        self.layer_importance = nn.Parameter(torch.ones(4) / 4)

    def forward(
        self,
        nodes: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Args:
            nodes: Dict with all 17 node features, each [batch, hidden_dim]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, hidden_dim]
            attention_dict: Layer-wise attention if requested
        """
        # Layer 1: Global Context
        h_global, global_attn = self.global_attn(
            nodes["venue"],
            nodes["batting_team"],
            nodes["bowling_team"],
        )

        # Layer 2: Match State (conditioned on global)
        h_state, state_attn = self.state_attn(
            nodes["score_state"],
            nodes["chase_state"],
            nodes["phase_state"],
            nodes["time_pressure"],
            nodes["wicket_buffer"],
            h_global,
        )

        # Layer 3: Actor (conditioned on state)
        h_actor, actor_attn = self.actor_attn(
            nodes["striker_identity"],
            nodes["striker_state"],
            nodes["bowler_identity"],
            nodes["bowler_state"],
            nodes["partnership"],
            h_state,
        )

        # Layer 4: Dynamics (conditioned on actor)
        h_dynamics, dynamics_attn = self.dynamics_attn(
            nodes["batting_momentum"],
            nodes["bowling_momentum"],
            nodes["pressure_index"],
            nodes["dot_pressure"],
            h_actor,
        )

        # Fusion with learned layer importance
        layer_weights = F.softmax(self.layer_importance, dim=0)
        h_weighted = (
            layer_weights[0] * h_global +
            layer_weights[1] * h_state +
            layer_weights[2] * h_actor +
            layer_weights[3] * h_dynamics
        )

        # Also concatenate for richer representation
        h_concat = torch.cat([h_global, h_state, h_actor, h_dynamics], dim=-1)
        output = self.fusion(h_concat) + h_weighted

        if return_attention:
            attention_dict = {
                "layer_importance": {
                    "global": layer_weights[0].item(),
                    "match_state": layer_weights[1].item(),
                    "actor": layer_weights[2].item(),
                    "dynamics": layer_weights[3].item(),
                },
                "global": {
                    "venue": global_attn[0, 0].item() if global_attn.dim() > 1 else 0,
                    "batting_team": global_attn[0, 1].item() if global_attn.dim() > 1 else 0,
                    "bowling_team": global_attn[0, 2].item() if global_attn.dim() > 1 else 0,
                },
                "match_state": state_attn,
                "actor": actor_attn,
                "dynamics": dynamics_attn,
            }
            return output, attention_dict

        return output, None
