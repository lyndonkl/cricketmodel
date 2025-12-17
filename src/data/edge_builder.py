"""
Edge Builder for Cricket Heterogeneous Graph

Defines graph structure and builds edges for the unified heterogeneous graph.
All edge indices are returned as PyTorch tensors.
"""

import torch
from typing import Dict, List, Tuple, Set
from collections import defaultdict


# =============================================================================
# NODE TYPE DEFINITIONS
# =============================================================================

# All node types in the graph (21 total for context + N balls + 1 query)
NODE_TYPES = [
    # Entity nodes (6 types) - includes non-striker
    'venue',
    'batting_team',
    'bowling_team',
    'striker_identity',
    'nonstriker_identity',
    'bowler_identity',
    # State nodes (5 types)
    'score_state',
    'chase_state',
    'phase_state',
    'time_pressure',
    'wicket_buffer',
    # Actor state nodes (4 types) - includes non-striker state
    'striker_state',
    'nonstriker_state',
    'bowler_state',
    'partnership',
    # Dynamics nodes (4 types)
    'batting_momentum',
    'bowling_momentum',
    'pressure_index',
    'dot_pressure',
    # Ball nodes (N nodes of same type)
    'ball',
    # Query node
    'query',
]

# Group nodes by layer for hierarchical edges
LAYER_NODES = {
    'global': ['venue', 'batting_team', 'bowling_team'],
    'state': ['score_state', 'chase_state', 'phase_state', 'time_pressure', 'wicket_buffer'],
    'actor': [
        'striker_identity', 'striker_state',
        'nonstriker_identity', 'nonstriker_state',
        'bowler_identity', 'bowler_state',
        'partnership'
    ],
    'dynamics': ['batting_momentum', 'bowling_momentum', 'pressure_index', 'dot_pressure'],
}


# =============================================================================
# EDGE TYPE DEFINITIONS
# =============================================================================

# Edge type format: (source_type, relation, target_type)
EdgeType = Tuple[str, str, str]

# Hierarchical edges: top-down conditioning between layers
HIERARCHICAL_EDGES: List[Tuple[str, str]] = [
    # Global -> State (all-to-all)
    *[(g, s) for g in LAYER_NODES['global'] for s in LAYER_NODES['state']],
    # State -> Actor (all-to-all)
    *[(s, a) for s in LAYER_NODES['state'] for a in LAYER_NODES['actor']],
    # Actor -> Dynamics (all-to-all)
    *[(a, d) for a in LAYER_NODES['actor'] for d in LAYER_NODES['dynamics']],
]

# Intra-layer edges: interactions within the same layer
INTRA_LAYER_GLOBAL = [
    ('venue', 'batting_team'),
    ('venue', 'bowling_team'),
    ('batting_team', 'bowling_team'),
]

INTRA_LAYER_STATE = [
    ('score_state', 'chase_state'),
    ('score_state', 'phase_state'),
    ('score_state', 'time_pressure'),
    ('score_state', 'wicket_buffer'),
    ('chase_state', 'time_pressure'),
    ('phase_state', 'time_pressure'),
    ('time_pressure', 'wicket_buffer'),
]

# Actor matchup edges (semantically important)
INTRA_LAYER_ACTOR = [
    # Identity to state connections
    ('striker_identity', 'striker_state'),
    ('nonstriker_identity', 'nonstriker_state'),
    ('bowler_identity', 'bowler_state'),
    # THE KEY MATCHUP: striker vs bowler
    ('striker_identity', 'bowler_identity'),
    # Non-striker matchups (for run-out risk, strike rotation)
    ('nonstriker_identity', 'bowler_identity'),
    ('striker_identity', 'nonstriker_identity'),
    # Partnership connections - both batsmen contribute
    ('striker_state', 'partnership'),
    ('nonstriker_state', 'partnership'),
    ('bowler_state', 'partnership'),
    ('striker_identity', 'partnership'),
    ('nonstriker_identity', 'partnership'),
]

INTRA_LAYER_DYNAMICS = [
    ('batting_momentum', 'bowling_momentum'),
    ('batting_momentum', 'pressure_index'),
    ('bowling_momentum', 'pressure_index'),
    ('pressure_index', 'dot_pressure'),
]


# =============================================================================
# EDGE BUILDING FUNCTIONS
# =============================================================================

def build_hierarchical_edges() -> Dict[EdgeType, torch.Tensor]:
    """
    Build hierarchical conditioning edges between layers.

    Returns all-to-all edges between consecutive layers:
    - Global -> State
    - State -> Actor
    - Actor -> Dynamics

    Each context node type has exactly 1 node, so edges are simple.

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    for src_type, tgt_type in HIERARCHICAL_EDGES:
        # Since each type has exactly 1 node, edge is [0] -> [0]
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_type = (src_type, 'conditions', tgt_type)
        edges[edge_type] = edge_index

    return edges


def build_intra_layer_edges() -> Dict[EdgeType, torch.Tensor]:
    """
    Build within-layer interaction edges.

    Creates bidirectional edges between related nodes in the same layer.

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    # Global layer
    for src, tgt in INTRA_LAYER_GLOBAL:
        # Bidirectional: add both directions
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(src, 'relates_to', tgt)] = edge_index
        edges[(tgt, 'relates_to', src)] = edge_index.clone()

    # State layer
    for src, tgt in INTRA_LAYER_STATE:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(src, 'relates_to', tgt)] = edge_index
        edges[(tgt, 'relates_to', src)] = edge_index.clone()

    # Actor layer (use 'matchup' relation for semantic clarity)
    for src, tgt in INTRA_LAYER_ACTOR:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(src, 'matchup', tgt)] = edge_index
        edges[(tgt, 'matchup', src)] = edge_index.clone()

    # Dynamics layer
    for src, tgt in INTRA_LAYER_DYNAMICS:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(src, 'relates_to', tgt)] = edge_index
        edges[(tgt, 'relates_to', src)] = edge_index.clone()

    return edges


def build_temporal_edges(
    num_balls: int,
    bowler_ids: List[int],
    batsman_ids: List[int]
) -> Dict[EdgeType, torch.Tensor]:
    """
    Build temporal edges between ball nodes.

    Creates four types of temporal edges:
    - precedes: sequential ordering (ball i -> ball i+1)
    - same_bowler: connects balls by same bowler
    - same_batsman: connects balls faced by same batsman
    - same_matchup: connects balls with same bowler-batsman pair (key for matchup learning)

    Args:
        num_balls: Number of historical balls
        bowler_ids: List of bowler IDs for each ball
        batsman_ids: List of batsman IDs for each ball

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    if num_balls == 0:
        # Return empty edges for empty history
        edges[('ball', 'precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_bowler', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_batsman', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_matchup', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        return edges

    # 1. Precedes edges: sequential ordering
    if num_balls > 1:
        src = list(range(num_balls - 1))
        tgt = list(range(1, num_balls))
        edges[('ball', 'precedes', 'ball')] = torch.tensor([src, tgt], dtype=torch.long)
    else:
        edges[('ball', 'precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    # 2. Same bowler edges: connect balls by same bowler
    bowler_to_balls = defaultdict(list)
    for ball_idx, bowler_id in enumerate(bowler_ids):
        bowler_to_balls[bowler_id].append(ball_idx)

    same_bowler_src = []
    same_bowler_tgt = []
    for balls in bowler_to_balls.values():
        if len(balls) > 1:
            # Create clique edges (bidirectional)
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    same_bowler_src.extend([balls[i], balls[j]])
                    same_bowler_tgt.extend([balls[j], balls[i]])

    if same_bowler_src:
        edges[('ball', 'same_bowler', 'ball')] = torch.tensor(
            [same_bowler_src, same_bowler_tgt], dtype=torch.long
        )
    else:
        edges[('ball', 'same_bowler', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    # 3. Same batsman edges: connect balls faced by same batsman
    batsman_to_balls = defaultdict(list)
    for ball_idx, batsman_id in enumerate(batsman_ids):
        batsman_to_balls[batsman_id].append(ball_idx)

    same_batsman_src = []
    same_batsman_tgt = []
    for balls in batsman_to_balls.values():
        if len(balls) > 1:
            # Create clique edges (bidirectional)
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    same_batsman_src.extend([balls[i], balls[j]])
                    same_batsman_tgt.extend([balls[j], balls[i]])

    if same_batsman_src:
        edges[('ball', 'same_batsman', 'ball')] = torch.tensor(
            [same_batsman_src, same_batsman_tgt], dtype=torch.long
        )
    else:
        edges[('ball', 'same_batsman', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    # 4. Same matchup edges: connect balls with same bowler-batsman pair
    # This is THE key predictor - specific matchup history
    matchup_to_balls = defaultdict(list)
    for ball_idx, (bowler_id, batsman_id) in enumerate(zip(bowler_ids, batsman_ids)):
        matchup_key = (bowler_id, batsman_id)
        matchup_to_balls[matchup_key].append(ball_idx)

    same_matchup_src = []
    same_matchup_tgt = []
    for balls in matchup_to_balls.values():
        if len(balls) > 1:
            # Create clique edges (bidirectional)
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    same_matchup_src.extend([balls[i], balls[j]])
                    same_matchup_tgt.extend([balls[j], balls[i]])

    if same_matchup_src:
        edges[('ball', 'same_matchup', 'ball')] = torch.tensor(
            [same_matchup_src, same_matchup_tgt], dtype=torch.long
        )
    else:
        edges[('ball', 'same_matchup', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    return edges


def build_cross_domain_edges(
    num_balls: int,
    recent_k: int = 12
) -> Dict[EdgeType, torch.Tensor]:
    """
    Build edges connecting ball nodes to context nodes.

    Creates edges:
    - ball -> striker_identity (faced_by)
    - ball -> nonstriker_identity (partnered_by)
    - ball -> bowler_identity (bowled_by)
    - dynamics <- recent balls (informs)

    Args:
        num_balls: Number of historical balls
        recent_k: Number of recent balls for dynamics connection

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    if num_balls == 0:
        # Return empty edges for empty history
        edges[('ball', 'faced_by', 'striker_identity')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'partnered_by', 'nonstriker_identity')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'bowled_by', 'bowler_identity')] = torch.zeros((2, 0), dtype=torch.long)
        for dynamics_node in LAYER_NODES['dynamics']:
            edges[('ball', 'informs', dynamics_node)] = torch.zeros((2, 0), dtype=torch.long)
        return edges

    # All balls connect to current striker/nonstriker/bowler identity
    # (Note: these are the CURRENT players, not historical)
    ball_indices = list(range(num_balls))
    target_indices = [0] * num_balls  # All point to single identity node

    edges[('ball', 'faced_by', 'striker_identity')] = torch.tensor(
        [ball_indices, target_indices], dtype=torch.long
    )
    edges[('ball', 'partnered_by', 'nonstriker_identity')] = torch.tensor(
        [ball_indices, target_indices], dtype=torch.long
    )
    edges[('ball', 'bowled_by', 'bowler_identity')] = torch.tensor(
        [ball_indices, target_indices], dtype=torch.long
    )

    # Recent balls inform dynamics
    recent_balls = list(range(max(0, num_balls - recent_k), num_balls))
    dynamics_target = [0] * len(recent_balls)

    for dynamics_node in LAYER_NODES['dynamics']:
        edges[('ball', 'informs', dynamics_node)] = torch.tensor(
            [recent_balls, dynamics_target], dtype=torch.long
        )

    return edges


def build_query_edges(num_balls: int) -> Dict[EdgeType, torch.Tensor]:
    """
    Build edges from all context nodes to the query node.

    The query node aggregates information from everything for prediction.
    Information flows FROM other nodes TO query (for message passing).

    Args:
        num_balls: Number of historical balls

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    # Context nodes -> query (each type has 1 node, query has 1 node)
    context_types = (
        LAYER_NODES['global'] +
        LAYER_NODES['state'] +
        LAYER_NODES['actor'] +
        LAYER_NODES['dynamics']
    )

    for node_type in context_types:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(node_type, 'attends', 'query')] = edge_index

    # Balls -> query
    if num_balls > 0:
        ball_indices = list(range(num_balls))
        query_indices = [0] * num_balls
        edges[('ball', 'attends', 'query')] = torch.tensor(
            [ball_indices, query_indices], dtype=torch.long
        )
    else:
        edges[('ball', 'attends', 'query')] = torch.zeros((2, 0), dtype=torch.long)

    return edges


def build_all_edges(
    num_balls: int,
    bowler_ids: List[int],
    batsman_ids: List[int],
    recent_k: int = 12
) -> Dict[EdgeType, torch.Tensor]:
    """
    Build all edges for the heterogeneous graph.

    Combines:
    - Hierarchical conditioning edges
    - Intra-layer interaction edges
    - Temporal ball edges
    - Cross-domain edges
    - Query aggregation edges

    Args:
        num_balls: Number of historical balls
        bowler_ids: Bowler ID for each ball
        batsman_ids: Batsman ID for each ball
        recent_k: Number of recent balls for dynamics

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    all_edges = {}

    # Static edges (don't depend on ball count)
    all_edges.update(build_hierarchical_edges())
    all_edges.update(build_intra_layer_edges())

    # Dynamic edges (depend on ball history)
    all_edges.update(build_temporal_edges(num_balls, bowler_ids, batsman_ids))
    all_edges.update(build_cross_domain_edges(num_balls, recent_k))
    all_edges.update(build_query_edges(num_balls))

    return all_edges


def get_all_edge_types() -> List[EdgeType]:
    """
    Get list of all possible edge types in the graph.

    Returns:
        List of (src_type, relation, tgt_type) tuples
    """
    edge_types = set()

    # Hierarchical
    for src, tgt in HIERARCHICAL_EDGES:
        edge_types.add((src, 'conditions', tgt))

    # Intra-layer global
    for src, tgt in INTRA_LAYER_GLOBAL:
        edge_types.add((src, 'relates_to', tgt))
        edge_types.add((tgt, 'relates_to', src))

    # Intra-layer state
    for src, tgt in INTRA_LAYER_STATE:
        edge_types.add((src, 'relates_to', tgt))
        edge_types.add((tgt, 'relates_to', src))

    # Intra-layer actor (matchup)
    for src, tgt in INTRA_LAYER_ACTOR:
        edge_types.add((src, 'matchup', tgt))
        edge_types.add((tgt, 'matchup', src))

    # Intra-layer dynamics
    for src, tgt in INTRA_LAYER_DYNAMICS:
        edge_types.add((src, 'relates_to', tgt))
        edge_types.add((tgt, 'relates_to', src))

    # Temporal
    edge_types.add(('ball', 'precedes', 'ball'))
    edge_types.add(('ball', 'same_bowler', 'ball'))
    edge_types.add(('ball', 'same_batsman', 'ball'))
    edge_types.add(('ball', 'same_matchup', 'ball'))

    # Cross-domain
    edge_types.add(('ball', 'faced_by', 'striker_identity'))
    edge_types.add(('ball', 'partnered_by', 'nonstriker_identity'))
    edge_types.add(('ball', 'bowled_by', 'bowler_identity'))
    for dynamics_node in LAYER_NODES['dynamics']:
        edge_types.add(('ball', 'informs', dynamics_node))

    # Query edges
    context_types = (
        LAYER_NODES['global'] +
        LAYER_NODES['state'] +
        LAYER_NODES['actor'] +
        LAYER_NODES['dynamics']
    )
    for node_type in context_types:
        edge_types.add((node_type, 'attends', 'query'))
    edge_types.add(('ball', 'attends', 'query'))

    return sorted(list(edge_types))


def count_edges(edge_dict: Dict[EdgeType, torch.Tensor]) -> int:
    """Count total number of edges in edge dictionary."""
    return sum(e.shape[1] for e in edge_dict.values())
