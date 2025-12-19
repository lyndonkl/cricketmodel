"""
Edge Builder for Cricket Heterogeneous Graph

Defines graph structure and builds edges for the unified heterogeneous graph.
All edge indices are returned as PyTorch tensors.
"""

import torch
from typing import Dict, List, Tuple, Set, Optional
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


def build_same_over_edges(
    num_balls: int,
    ball_overs: List[int]
) -> Dict[EdgeType, torch.Tensor]:
    """
    Build CAUSAL edges connecting balls within the same over.

    Over boundaries are significant discontinuities in cricket:
    - New bowler starts
    - Batsmen swap ends
    - Fielding positions reset

    Within an over, there's strong local coherence:
    - Same bowler rhythm
    - Same batsman facing
    - Consistent field placement

    IMPORTANT: Edges are CAUSAL (older -> newer only) to prevent train-test
    distribution shift. During training, bidirectional edges would allow
    future-to-past information flow, but at inference only historical balls
    exist. Causal edges ensure consistent behavior.

    Args:
        num_balls: Number of historical balls
        ball_overs: List of over numbers for each ball (0-indexed)

    Returns:
        Dict mapping edge type to edge_index tensor [2, num_edges]
    """
    edges = {}

    if num_balls == 0:
        edges[('ball', 'same_over', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        return edges

    # Group balls by over
    over_to_balls = defaultdict(list)
    for ball_idx, over_num in enumerate(ball_overs):
        over_to_balls[over_num].append(ball_idx)

    same_over_src = []
    same_over_tgt = []

    for balls in over_to_balls.values():
        if len(balls) > 1:
            # Sort to ensure temporal order within over
            sorted_balls = sorted(balls)
            # Create CAUSAL edges only (older -> newer)
            for i in range(len(sorted_balls)):
                for j in range(i + 1, len(sorted_balls)):
                    # Only older -> newer direction (causal)
                    same_over_src.append(sorted_balls[i])
                    same_over_tgt.append(sorted_balls[j])

    if same_over_src:
        edges[('ball', 'same_over', 'ball')] = torch.tensor(
            [same_over_src, same_over_tgt], dtype=torch.long
        )
    else:
        edges[('ball', 'same_over', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    return edges


def build_temporal_edges(
    num_balls: int,
    bowler_ids: List[int],
    batsman_ids: List[int],
    ball_overs: Optional[List[int]] = None,
    max_temporal_distance: int = 120
) -> Tuple[Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    """
    Build temporal edges between ball nodes.

    Creates MULTI-SCALE temporal edges to capture different temporal patterns:
    - recent_precedes: Last 6 balls (within-over, immediate context)
    - medium_precedes: 7-18 balls ago (2-over window, momentum context)
    - distant_precedes: 19+ balls ago (historical patterns, sparse connections)

    Plus set-based edges:
    - same_bowler: connects balls by same bowler
    - same_batsman: connects balls faced by same batsman
    - same_matchup: connects balls with same bowler-batsman pair (CAUSAL: older -> newer)

    Multi-scale temporal edges allow the model to:
    - Weight recent transitions more heavily (fast decay)
    - Capture medium-range momentum patterns (medium decay)
    - Access historical context when relevant (slow decay)

    Args:
        num_balls: Number of historical balls
        bowler_ids: List of bowler IDs for each ball
        batsman_ids: List of batsman IDs for each ball
        max_temporal_distance: Maximum temporal distance for normalization (T20 = 120 balls)

    Returns:
        Tuple of:
        - Dict mapping edge type to edge_index tensor [2, num_edges]
        - Dict mapping edge type to edge_attr tensor [num_edges, edge_dim]
    """
    edges = {}
    edge_attrs = {}

    if num_balls == 0:
        # Return empty edges for empty history
        edges[('ball', 'recent_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'medium_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'distant_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_bowler', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_batsman', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_matchup', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edges[('ball', 'same_over', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        # Empty edge attributes
        edge_attrs[('ball', 'recent_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)
        edge_attrs[('ball', 'medium_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)
        edge_attrs[('ball', 'distant_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)
        edge_attrs[('ball', 'same_bowler', 'ball')] = torch.zeros((0, 1), dtype=torch.float)
        edge_attrs[('ball', 'same_batsman', 'ball')] = torch.zeros((0, 1), dtype=torch.float)
        return edges, edge_attrs

    # 1. Multi-scale precedes edges
    # Each scale captures different temporal patterns in cricket:
    # - Recent (6 balls): Current over context, immediate pressure
    # - Medium (12 balls): 2-over momentum window
    # - Distant (18+ balls): Historical patterns, phase transitions

    recent_src, recent_tgt, recent_dist = [], [], []
    medium_src, medium_tgt, medium_dist = [], [], []
    distant_src, distant_tgt, distant_dist = [], [], []

    for i in range(num_balls):
        for j in range(i + 1, num_balls):
            gap = j - i
            # Temporal distance feature: normalized by window size
            if gap <= 6:
                # Recent: within current over context
                recent_src.append(i)
                recent_tgt.append(j)
                recent_dist.append(gap / 6.0)
            elif gap <= 18:
                # Medium: 2-over momentum window
                medium_src.append(i)
                medium_tgt.append(j)
                medium_dist.append((gap - 6) / 12.0)
            else:
                # Distant: sparse connections every 6 balls for efficiency
                if gap % 6 == 0:
                    distant_src.append(i)
                    distant_tgt.append(j)
                    distant_dist.append((gap - 18) / max_temporal_distance)

    # Recent precedes
    if recent_src:
        edges[('ball', 'recent_precedes', 'ball')] = torch.tensor([recent_src, recent_tgt], dtype=torch.long)
        edge_attrs[('ball', 'recent_precedes', 'ball')] = torch.tensor([[d] for d in recent_dist], dtype=torch.float)
    else:
        edges[('ball', 'recent_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edge_attrs[('ball', 'recent_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)

    # Medium precedes
    if medium_src:
        edges[('ball', 'medium_precedes', 'ball')] = torch.tensor([medium_src, medium_tgt], dtype=torch.long)
        edge_attrs[('ball', 'medium_precedes', 'ball')] = torch.tensor([[d] for d in medium_dist], dtype=torch.float)
    else:
        edges[('ball', 'medium_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edge_attrs[('ball', 'medium_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)

    # Distant precedes
    if distant_src:
        edges[('ball', 'distant_precedes', 'ball')] = torch.tensor([distant_src, distant_tgt], dtype=torch.long)
        edge_attrs[('ball', 'distant_precedes', 'ball')] = torch.tensor([[d] for d in distant_dist], dtype=torch.float)
    else:
        edges[('ball', 'distant_precedes', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edge_attrs[('ball', 'distant_precedes', 'ball')] = torch.zeros((0, 1), dtype=torch.float)

    # 2. Same bowler edges: connect balls by same bowler WITH TEMPORAL DECAY
    # Edge attribute encodes temporal distance (normalized) for attention weighting
    # This allows the model to learn that recent balls in a spell matter more
    bowler_to_balls = defaultdict(list)
    for ball_idx, bowler_id in enumerate(bowler_ids):
        bowler_to_balls[bowler_id].append(ball_idx)

    same_bowler_src = []
    same_bowler_tgt = []
    same_bowler_dist = []  # Temporal distance for decay
    spell_window = 24.0  # ~4 overs - typical spell window for normalization

    for balls in bowler_to_balls.values():
        if len(balls) > 1:
            # Create clique edges (bidirectional) with temporal distance
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    temporal_dist = abs(balls[j] - balls[i]) / spell_window
                    # Forward edge
                    same_bowler_src.append(balls[i])
                    same_bowler_tgt.append(balls[j])
                    same_bowler_dist.append(temporal_dist)
                    # Backward edge (same distance)
                    same_bowler_src.append(balls[j])
                    same_bowler_tgt.append(balls[i])
                    same_bowler_dist.append(temporal_dist)

    if same_bowler_src:
        edges[('ball', 'same_bowler', 'ball')] = torch.tensor(
            [same_bowler_src, same_bowler_tgt], dtype=torch.long
        )
        edge_attrs[('ball', 'same_bowler', 'ball')] = torch.tensor(
            [[d] for d in same_bowler_dist], dtype=torch.float
        )
    else:
        edges[('ball', 'same_bowler', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edge_attrs[('ball', 'same_bowler', 'ball')] = torch.zeros((0, 1), dtype=torch.float)

    # 3. Same batsman edges: connect balls faced by same batsman WITH TEMPORAL DECAY
    # Similar decay for batsman form - recent balls reflect current form
    batsman_to_balls = defaultdict(list)
    for ball_idx, batsman_id in enumerate(batsman_ids):
        batsman_to_balls[batsman_id].append(ball_idx)

    same_batsman_src = []
    same_batsman_tgt = []
    same_batsman_dist = []  # Temporal distance for decay
    innings_window = 60.0  # ~10 overs - batsman form window

    for balls in batsman_to_balls.values():
        if len(balls) > 1:
            # Create clique edges (bidirectional) with temporal distance
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    temporal_dist = abs(balls[j] - balls[i]) / innings_window
                    # Forward edge
                    same_batsman_src.append(balls[i])
                    same_batsman_tgt.append(balls[j])
                    same_batsman_dist.append(temporal_dist)
                    # Backward edge (same distance)
                    same_batsman_src.append(balls[j])
                    same_batsman_tgt.append(balls[i])
                    same_batsman_dist.append(temporal_dist)

    if same_batsman_src:
        edges[('ball', 'same_batsman', 'ball')] = torch.tensor(
            [same_batsman_src, same_batsman_tgt], dtype=torch.long
        )
        edge_attrs[('ball', 'same_batsman', 'ball')] = torch.tensor(
            [[d] for d in same_batsman_dist], dtype=torch.float
        )
    else:
        edges[('ball', 'same_batsman', 'ball')] = torch.zeros((2, 0), dtype=torch.long)
        edge_attrs[('ball', 'same_batsman', 'ball')] = torch.zeros((0, 1), dtype=torch.float)

    # 4. Same matchup edges: connect balls with same bowler-batsman pair
    # This is THE key predictor - specific matchup history
    # CRITICAL: Use CAUSAL edges only (older -> newer) to prevent train-test distribution shift
    # During training, bidirectional edges would allow future-to-past information flow,
    # but at inference only historical balls exist. Causal edges ensure consistent behavior.
    matchup_to_balls = defaultdict(list)
    for ball_idx, (bowler_id, batsman_id) in enumerate(zip(bowler_ids, batsman_ids)):
        matchup_key = (bowler_id, batsman_id)
        matchup_to_balls[matchup_key].append(ball_idx)

    same_matchup_src = []
    same_matchup_tgt = []
    for balls in matchup_to_balls.values():
        if len(balls) > 1:
            # Sort to ensure temporal order, then create CAUSAL edges (older -> newer only)
            sorted_balls = sorted(balls)
            for i in range(len(sorted_balls)):
                for j in range(i + 1, len(sorted_balls)):
                    # Only older -> newer direction (causal)
                    same_matchup_src.append(sorted_balls[i])
                    same_matchup_tgt.append(sorted_balls[j])

    if same_matchup_src:
        edges[('ball', 'same_matchup', 'ball')] = torch.tensor(
            [same_matchup_src, same_matchup_tgt], dtype=torch.long
        )
    else:
        edges[('ball', 'same_matchup', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    # 5. Same over edges: connect balls within the same over
    # Over boundaries are significant discontinuities (new bowler, batsmen swap)
    if ball_overs is not None:
        same_over_edges = build_same_over_edges(num_balls, ball_overs)
        edges.update(same_over_edges)
    else:
        edges[('ball', 'same_over', 'ball')] = torch.zeros((2, 0), dtype=torch.long)

    return edges, edge_attrs


def build_cross_domain_edges(
    num_balls: int,
    historical_batsman_ids: List[int],
    historical_bowler_ids: List[int],
    historical_nonstriker_ids: List[int],
    current_striker_id: int,
    current_bowler_id: int,
    current_nonstriker_id: int,
    recent_k: int = 12
) -> Dict[EdgeType, torch.Tensor]:
    """
    Build edges connecting ball nodes to context nodes.

    Creates SEMANTICALLY CORRECT edges that connect each historical ball
    to the players who actually faced/bowled/partnered that ball, filtered
    to only include edges to CURRENT players (for relevance).

    Edge types:
    - ball -> striker_identity (faced_by): Only balls faced by CURRENT striker
    - ball -> nonstriker_identity (partnered_by): Only balls where CURRENT non-striker was partner
    - ball -> bowler_identity (bowled_by): Only balls bowled by CURRENT bowler
    - dynamics <- recent balls (informs): Recent balls inform momentum/pressure

    This respects the Z2 symmetry of striker/non-striker swapping after odd runs.

    Args:
        num_balls: Number of historical balls
        historical_batsman_ids: Player ID of batsman who faced each historical ball
        historical_bowler_ids: Player ID of bowler who bowled each historical ball
        historical_nonstriker_ids: Player ID of non-striker for each historical ball
        current_striker_id: ID of current striker
        current_bowler_id: ID of current bowler
        current_nonstriker_id: ID of current non-striker
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

    # faced_by: Connect balls actually faced by the CURRENT striker
    # This gives the model the current striker's historical performance
    striker_faced_balls = [
        i for i, batsman_id in enumerate(historical_batsman_ids)
        if batsman_id == current_striker_id
    ]
    if striker_faced_balls:
        edges[('ball', 'faced_by', 'striker_identity')] = torch.tensor(
            [striker_faced_balls, [0] * len(striker_faced_balls)], dtype=torch.long
        )
    else:
        edges[('ball', 'faced_by', 'striker_identity')] = torch.zeros((2, 0), dtype=torch.long)

    # partnered_by: Connect balls where CURRENT non-striker was at non-striker end
    # OR balls that the current non-striker actually faced (they were striker then)
    nonstriker_partner_balls = [
        i for i, (ns_id, bat_id) in enumerate(zip(historical_nonstriker_ids, historical_batsman_ids))
        if ns_id == current_nonstriker_id or bat_id == current_nonstriker_id
    ]
    if nonstriker_partner_balls:
        edges[('ball', 'partnered_by', 'nonstriker_identity')] = torch.tensor(
            [nonstriker_partner_balls, [0] * len(nonstriker_partner_balls)], dtype=torch.long
        )
    else:
        edges[('ball', 'partnered_by', 'nonstriker_identity')] = torch.zeros((2, 0), dtype=torch.long)

    # bowled_by: Connect balls actually bowled by the CURRENT bowler
    # This gives the model the current bowler's spell performance
    bowler_bowled_balls = [
        i for i, bowler_id in enumerate(historical_bowler_ids)
        if bowler_id == current_bowler_id
    ]
    if bowler_bowled_balls:
        edges[('ball', 'bowled_by', 'bowler_identity')] = torch.tensor(
            [bowler_bowled_balls, [0] * len(bowler_bowled_balls)], dtype=torch.long
        )
    else:
        edges[('ball', 'bowled_by', 'bowler_identity')] = torch.zeros((2, 0), dtype=torch.long)

    # Recent balls inform dynamics (unchanged - all recent balls matter for momentum)
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

    Additionally, dynamics nodes have explicit 'drives' edges to query to ensure
    momentum and pressure signals directly influence predictions. This is critical
    because momentum is a forward-looking predictor (positive momentum predicts
    more positive outcomes).

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

    # Dynamics nodes also have explicit 'drives' edges to query
    # This ensures momentum/pressure directly influence predictions
    # The 'drives' relation is separate from 'attends' to allow different
    # convolution operators (momentum should have strong influence on prediction)
    for dynamics_node in LAYER_NODES['dynamics']:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edges[(dynamics_node, 'drives', 'query')] = edge_index

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
    nonstriker_ids: List[int],
    current_striker_id: int,
    current_bowler_id: int,
    current_nonstriker_id: int,
    ball_overs: Optional[List[int]] = None,
    recent_k: int = 12
) -> Tuple[Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    """
    Build all edges for the heterogeneous graph.

    Combines:
    - Hierarchical conditioning edges
    - Intra-layer interaction edges
    - Temporal ball edges (with edge attributes for temporal distance)
    - Same-over edges (structural over boundary encoding)
    - Cross-domain edges (with correct player attribution)
    - Query aggregation edges

    Args:
        num_balls: Number of historical balls
        bowler_ids: Bowler ID for each historical ball (who bowled)
        batsman_ids: Batsman ID for each historical ball (who faced)
        nonstriker_ids: Non-striker ID for each historical ball
        current_striker_id: ID of current striker (for cross-domain edges)
        current_bowler_id: ID of current bowler (for cross-domain edges)
        current_nonstriker_id: ID of current non-striker (for cross-domain edges)
        ball_overs: Over number for each historical ball (for same_over edges)
        recent_k: Number of recent balls for dynamics

    Returns:
        Tuple of:
        - Dict mapping edge type to edge_index tensor [2, num_edges]
        - Dict mapping edge type to edge_attr tensor [num_edges, edge_dim] (for edges with attributes)
    """
    all_edges = {}
    all_edge_attrs = {}

    # Static edges (don't depend on ball count)
    all_edges.update(build_hierarchical_edges())
    all_edges.update(build_intra_layer_edges())

    # Dynamic edges (depend on ball history)
    # Temporal edges now return both edge_index and edge_attr
    # Includes same_over edges if ball_overs is provided
    temporal_edges, temporal_edge_attrs = build_temporal_edges(
        num_balls, bowler_ids, batsman_ids, ball_overs
    )
    all_edges.update(temporal_edges)
    all_edge_attrs.update(temporal_edge_attrs)

    all_edges.update(build_cross_domain_edges(
        num_balls=num_balls,
        historical_batsman_ids=batsman_ids,
        historical_bowler_ids=bowler_ids,
        historical_nonstriker_ids=nonstriker_ids,
        current_striker_id=current_striker_id,
        current_bowler_id=current_bowler_id,
        current_nonstriker_id=current_nonstriker_id,
        recent_k=recent_k
    ))
    all_edges.update(build_query_edges(num_balls))

    return all_edges, all_edge_attrs


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

    # Temporal - Multi-scale for different temporal patterns
    edge_types.add(('ball', 'recent_precedes', 'ball'))   # Last 6 balls (within-over)
    edge_types.add(('ball', 'medium_precedes', 'ball'))   # 7-18 balls (momentum window)
    edge_types.add(('ball', 'distant_precedes', 'ball'))  # 19+ balls (historical, sparse)
    edge_types.add(('ball', 'same_bowler', 'ball'))
    edge_types.add(('ball', 'same_batsman', 'ball'))
    edge_types.add(('ball', 'same_matchup', 'ball'))
    edge_types.add(('ball', 'same_over', 'ball'))  # Within-over structural edges

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

    # Dynamics -> query 'drives' edges (momentum directly influences prediction)
    for dynamics_node in LAYER_NODES['dynamics']:
        edge_types.add((dynamics_node, 'drives', 'query'))

    return sorted(list(edge_types))


def count_edges(edge_dict: Dict[EdgeType, torch.Tensor]) -> int:
    """Count total number of edges in edge dictionary."""
    return sum(e.shape[1] for e in edge_dict.values())
