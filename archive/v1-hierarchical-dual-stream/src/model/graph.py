"""Graph structure for hierarchical attention.

17 semantic nodes across 4 layers:

Layer 1 - Global Context (3 nodes):
    0: venue
    1: batting_team
    2: bowling_team

Layer 2 - Match State (5 nodes):
    3: score_state
    4: chase_state
    5: phase_state
    6: time_pressure
    7: wicket_buffer

Layer 3 - Actor (5 nodes):
    8: striker_identity
    9: striker_state
    10: bowler_identity
    11: bowler_state
    12: partnership

Layer 4 - Dynamics (4 nodes):
    13: batting_momentum
    14: bowling_momentum
    15: pressure_index
    16: dot_pressure
"""

from dataclasses import dataclass
from enum import IntEnum

import torch


class NodeType(IntEnum):
    """Node indices in the graph."""

    # Global context (Layer 1)
    VENUE = 0
    BATTING_TEAM = 1
    BOWLING_TEAM = 2

    # Match state (Layer 2)
    SCORE_STATE = 3
    CHASE_STATE = 4
    PHASE_STATE = 5
    TIME_PRESSURE = 6
    WICKET_BUFFER = 7

    # Actor (Layer 3)
    STRIKER_IDENTITY = 8
    STRIKER_STATE = 9
    BOWLER_IDENTITY = 10
    BOWLER_STATE = 11
    PARTNERSHIP = 12

    # Dynamics (Layer 4)
    BATTING_MOMENTUM = 13
    BOWLING_MOMENTUM = 14
    PRESSURE_INDEX = 15
    DOT_PRESSURE = 16


NUM_NODES = 17

# Node dimensions (must match feature extraction)
NODE_DIMS = {
    NodeType.VENUE: 32,
    NodeType.BATTING_TEAM: 32,
    NodeType.BOWLING_TEAM: 32,
    NodeType.SCORE_STATE: 5,
    NodeType.CHASE_STATE: 4,
    NodeType.PHASE_STATE: 4,
    NodeType.TIME_PRESSURE: 3,
    NodeType.WICKET_BUFFER: 2,
    NodeType.STRIKER_IDENTITY: 64,
    NodeType.STRIKER_STATE: 6,
    NodeType.BOWLER_IDENTITY: 64,
    NodeType.BOWLER_STATE: 6,
    NodeType.PARTNERSHIP: 4,
    NodeType.BATTING_MOMENTUM: 1,
    NodeType.BOWLING_MOMENTUM: 1,
    NodeType.PRESSURE_INDEX: 1,
    NodeType.DOT_PRESSURE: 2,
}


def build_edge_index() -> torch.Tensor:
    """
    Build edge index for the graph.

    Edges encode hierarchical attention flow:
    - Global -> Match State
    - Match State -> Actor
    - Actor -> Dynamics
    - Within-layer connections
    """
    edges = []

    # Global -> Match State (all global nodes connect to all match state)
    global_nodes = [NodeType.VENUE, NodeType.BATTING_TEAM, NodeType.BOWLING_TEAM]
    state_nodes = [
        NodeType.SCORE_STATE, NodeType.CHASE_STATE, NodeType.PHASE_STATE,
        NodeType.TIME_PRESSURE, NodeType.WICKET_BUFFER
    ]
    for g in global_nodes:
        for s in state_nodes:
            edges.append([g, s])

    # Match State -> Actor
    actor_nodes = [
        NodeType.STRIKER_IDENTITY, NodeType.STRIKER_STATE,
        NodeType.BOWLER_IDENTITY, NodeType.BOWLER_STATE, NodeType.PARTNERSHIP
    ]
    for s in state_nodes:
        for a in actor_nodes:
            edges.append([s, a])

    # Actor -> Dynamics
    dynamics_nodes = [
        NodeType.BATTING_MOMENTUM, NodeType.BOWLING_MOMENTUM,
        NodeType.PRESSURE_INDEX, NodeType.DOT_PRESSURE
    ]
    for a in actor_nodes:
        for d in dynamics_nodes:
            edges.append([a, d])

    # Within-layer connections (Actor layer)
    # Striker identity <-> Striker state
    edges.append([NodeType.STRIKER_IDENTITY, NodeType.STRIKER_STATE])
    edges.append([NodeType.STRIKER_STATE, NodeType.STRIKER_IDENTITY])

    # Bowler identity <-> Bowler state
    edges.append([NodeType.BOWLER_IDENTITY, NodeType.BOWLER_STATE])
    edges.append([NodeType.BOWLER_STATE, NodeType.BOWLER_IDENTITY])

    # Partnership connects to both batsmen
    edges.append([NodeType.STRIKER_STATE, NodeType.PARTNERSHIP])
    edges.append([NodeType.PARTNERSHIP, NodeType.STRIKER_STATE])

    # Add reverse edges for bidirectional flow
    reverse_edges = [[e[1], e[0]] for e in edges]
    all_edges = edges + reverse_edges

    # Remove duplicates
    all_edges = list(set(tuple(e) for e in all_edges))

    return torch.tensor(all_edges, dtype=torch.long).t().contiguous()


@dataclass
class BallGraph:
    """Graph representation for a single ball prediction."""

    x: torch.Tensor  # Node features [num_nodes, hidden_dim]
    edge_index: torch.Tensor  # Edge connectivity [2, num_edges]

    @classmethod
    def from_features(
        cls,
        features: dict[str, torch.Tensor],
        hidden_dim: int = 64,
    ) -> "BallGraph":
        """
        Build graph from extracted features.

        Args:
            features: Dict with keys matching NODE_DIMS
            hidden_dim: Target hidden dimension (features projected to this)
        """
        # Project all features to hidden_dim
        node_features = []

        # Layer 1 - Global (already embeddings)
        node_features.append(features["venue"])  # [32] -> pad to hidden_dim
        node_features.append(features["batting_team"])
        node_features.append(features["bowling_team"])

        # Layer 2 - Match State
        node_features.append(features["score_state"])
        node_features.append(features["chase_state"])
        node_features.append(features["phase_state"])
        node_features.append(features["time_pressure"])
        node_features.append(features["wicket_buffer"])

        # Layer 3 - Actor
        node_features.append(features["striker_identity"])
        node_features.append(features["striker_state"])
        node_features.append(features["bowler_identity"])
        node_features.append(features["bowler_state"])
        node_features.append(features["partnership"])

        # Layer 4 - Dynamics
        node_features.append(features["batting_momentum"])
        node_features.append(features["bowling_momentum"])
        node_features.append(features["pressure_index"])
        node_features.append(features["dot_pressure"])

        # Pad/project all to hidden_dim
        x_list = []
        for i, feat in enumerate(node_features):
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            # Pad to hidden_dim
            if feat.shape[-1] < hidden_dim:
                padding = torch.zeros(feat.shape[0], hidden_dim - feat.shape[-1])
                feat = torch.cat([feat, padding], dim=-1)
            elif feat.shape[-1] > hidden_dim:
                feat = feat[..., :hidden_dim]
            x_list.append(feat)

        x = torch.stack([f.squeeze(0) for f in x_list])  # [17, hidden_dim]

        return cls(x=x, edge_index=build_edge_index())


def batch_graphs(graphs: list[BallGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch multiple graphs for efficient processing.

    Returns:
        x: [batch_size * num_nodes, hidden_dim]
        edge_index: [2, batch_size * num_edges] with offset node indices
        batch: [batch_size * num_nodes] indicating which graph each node belongs to
    """
    x_list = []
    edge_list = []
    batch_list = []

    node_offset = 0
    for i, g in enumerate(graphs):
        x_list.append(g.x)
        edge_list.append(g.edge_index + node_offset)
        batch_list.append(torch.full((g.x.shape[0],), i, dtype=torch.long))
        node_offset += g.x.shape[0]

    return (
        torch.cat(x_list, dim=0),
        torch.cat(edge_list, dim=1),
        torch.cat(batch_list, dim=0),
    )
