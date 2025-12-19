"""
HeteroData Builder for Cricket Ball Prediction

Constructs PyTorch Geometric HeteroData objects from match data.
Each HeteroData represents the complete state before a single ball.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple

from .entity_mapper import EntityMapper
from .feature_utils import (
    prepare_deliveries,
    compute_score_state,
    compute_chase_state,
    compute_phase_state,
    compute_time_pressure,
    compute_wicket_buffer,
    compute_striker_state,
    compute_nonstriker_state,
    compute_bowler_state,
    compute_partnership,
    compute_dynamics,
    compute_ball_features,
    outcome_to_class,
    BALL_FEATURE_DIM,
)
from .edge_builder import build_all_edges


def create_hetero_data(
    match_data: Dict,
    innings_idx: int,
    ball_idx: int,
    entity_mapper: EntityMapper
) -> HeteroData:
    """
    Create a HeteroData object for predicting ball at ball_idx.

    This is the core function that constructs the unified heterogeneous graph.

    Args:
        match_data: Full match JSON data
        innings_idx: Which innings (0 or 1)
        ball_idx: Which ball to predict (0-indexed in flattened deliveries)
        entity_mapper: Entity name to ID mapper

    Returns:
        HeteroData object with all node features, edges, and label
    """
    data = HeteroData()

    # Extract info
    info = match_data['info']
    innings = match_data['innings'][innings_idx]
    innings_num = innings_idx + 1

    # Extract super over flag (tie-breaker has unique dynamics)
    is_super_over = innings.get('super_over', False)

    # Extract gender for women's cricket indicator
    gender = info.get('gender', 'male')
    is_womens = gender == 'female'

    # Flatten deliveries
    all_deliveries = prepare_deliveries(innings)

    # Split into history (context) and target
    history = all_deliveries[:ball_idx]
    target_ball = all_deliveries[ball_idx]

    # Get target for chase calculations
    target_score = None
    if innings_num == 2 and 'target' in innings:
        target_score = innings['target'].get('runs')

    # Current striker/bowler/non-striker from target ball
    striker = target_ball['batter']
    bowler = target_ball['bowler']
    non_striker = target_ball.get('non_striker', striker)

    # Count runs and wickets from history
    total_runs = sum(d['runs']['total'] for d in history)
    total_wickets = sum(len(d.get('wickets', [])) for d in history)
    total_balls = len(history)

    # =========================================================================
    # 1. ENTITY NODES (Global layer)
    # =========================================================================

    # Venue
    venue_name = info.get('venue', 'Unknown')
    venue_id = entity_mapper.get_venue_id(venue_name)
    data['venue'].x = torch.tensor([[venue_id]], dtype=torch.long)

    # Teams
    teams = info.get('teams', ['Unknown', 'Unknown'])
    batting_team = innings.get('team', teams[0])
    bowling_team = teams[1] if teams[0] == batting_team else teams[0]

    batting_team_id = entity_mapper.get_team_id(batting_team)
    bowling_team_id = entity_mapper.get_team_id(bowling_team)

    data['batting_team'].x = torch.tensor([[batting_team_id]], dtype=torch.long)
    data['bowling_team'].x = torch.tensor([[bowling_team_id]], dtype=torch.long)

    # Player identities with hierarchical fallback info (team_id, role_id)
    striker_id, striker_team_id, striker_role_id = entity_mapper.get_player_hierarchy(striker)
    nonstriker_id, nonstriker_team_id, nonstriker_role_id = entity_mapper.get_player_hierarchy(non_striker)
    bowler_id, bowler_team_id, bowler_role_id = entity_mapper.get_player_hierarchy(bowler)

    data['striker_identity'].x = torch.tensor([[striker_id]], dtype=torch.long)
    data['striker_identity'].team_id = torch.tensor([[striker_team_id]], dtype=torch.long)
    data['striker_identity'].role_id = torch.tensor([[striker_role_id]], dtype=torch.long)

    data['nonstriker_identity'].x = torch.tensor([[nonstriker_id]], dtype=torch.long)
    data['nonstriker_identity'].team_id = torch.tensor([[nonstriker_team_id]], dtype=torch.long)
    data['nonstriker_identity'].role_id = torch.tensor([[nonstriker_role_id]], dtype=torch.long)

    data['bowler_identity'].x = torch.tensor([[bowler_id]], dtype=torch.long)
    data['bowler_identity'].team_id = torch.tensor([[bowler_team_id]], dtype=torch.long)
    data['bowler_identity'].role_id = torch.tensor([[bowler_role_id]], dtype=torch.long)

    # =========================================================================
    # 2. STATE NODES
    # =========================================================================

    # Score state (includes women's cricket indicator)
    score_features = compute_score_state(history, innings_num, is_womens)
    data['score_state'].x = torch.tensor([score_features], dtype=torch.float)

    # Chase state
    chase_features = compute_chase_state(history, target_score, innings_num)
    data['chase_state'].x = torch.tensor([chase_features], dtype=torch.float)

    # Phase state (includes super over flag)
    phase_features = compute_phase_state(total_balls, is_super_over)
    data['phase_state'].x = torch.tensor([phase_features], dtype=torch.float)

    # Time pressure
    time_features = compute_time_pressure(total_balls, target_score, total_runs, innings_num)
    data['time_pressure'].x = torch.tensor([time_features], dtype=torch.float)

    # Wicket buffer
    wicket_features = compute_wicket_buffer(total_wickets)
    data['wicket_buffer'].x = torch.tensor([wicket_features], dtype=torch.float)

    # =========================================================================
    # 3. ACTOR STATE NODES
    # =========================================================================

    # Striker state
    striker_features = compute_striker_state(history, striker)
    data['striker_state'].x = torch.tensor([striker_features], dtype=torch.float)

    # Non-striker state
    nonstriker_features = compute_nonstriker_state(history, non_striker)
    data['nonstriker_state'].x = torch.tensor([nonstriker_features], dtype=torch.float)

    # Bowler state (with bowling type from P1.2)
    bowling_type = entity_mapper.get_player_bowling_type(bowler)
    bowler_features = compute_bowler_state(history, bowler, bowling_type)
    data['bowler_state'].x = torch.tensor([bowler_features], dtype=torch.float)

    # Partnership
    partnership_features = compute_partnership(history, striker, non_striker)
    data['partnership'].x = torch.tensor([partnership_features], dtype=torch.float)

    # =========================================================================
    # 4. DYNAMICS NODES
    # =========================================================================

    dynamics = compute_dynamics(history, target_score, innings_num)

    data['batting_momentum'].x = torch.tensor(
        [dynamics['batting_momentum']], dtype=torch.float
    )
    data['bowling_momentum'].x = torch.tensor(
        [dynamics['bowling_momentum']], dtype=torch.float
    )
    data['pressure_index'].x = torch.tensor(
        [dynamics['pressure_index']], dtype=torch.float
    )
    data['dot_pressure'].x = torch.tensor(
        [dynamics['dot_pressure']], dtype=torch.float
    )

    # =========================================================================
    # 5. BALL NODES
    # =========================================================================

    num_balls = len(history)

    if num_balls > 0:
        # Ball features
        ball_features = [compute_ball_features(d) for d in history]
        data['ball'].x = torch.tensor(ball_features, dtype=torch.float)

        # Ball player IDs (for embedding lookup in model)
        # These track WHO actually faced/bowled/partnered each historical ball
        ball_bowler_ids = [entity_mapper.get_player_id(d['bowler']) for d in history]
        ball_batsman_ids = [entity_mapper.get_player_id(d['batter']) for d in history]
        ball_nonstriker_ids = [
            entity_mapper.get_player_id(d.get('non_striker', d['batter']))
            for d in history
        ]

        data['ball'].bowler_ids = torch.tensor(ball_bowler_ids, dtype=torch.long)
        data['ball'].batsman_ids = torch.tensor(ball_batsman_ids, dtype=torch.long)
        data['ball'].nonstriker_ids = torch.tensor(ball_nonstriker_ids, dtype=torch.long)
    else:
        # Empty ball nodes for first ball prediction
        data['ball'].x = torch.zeros((0, BALL_FEATURE_DIM), dtype=torch.float)
        data['ball'].bowler_ids = torch.zeros((0,), dtype=torch.long)
        data['ball'].batsman_ids = torch.zeros((0,), dtype=torch.long)
        data['ball'].nonstriker_ids = torch.zeros((0,), dtype=torch.long)

    # =========================================================================
    # 6. QUERY NODE
    # =========================================================================

    # Query node has learned embedding, we just initialize placeholder
    data['query'].x = torch.zeros((1, 1), dtype=torch.float)

    # =========================================================================
    # 7. BUILD EDGES
    # =========================================================================

    # Get ball player IDs and over numbers for temporal and cross-domain edge building
    if num_balls > 0:
        bowler_ids_list = data['ball'].bowler_ids.tolist()
        batsman_ids_list = data['ball'].batsman_ids.tolist()
        nonstriker_ids_list = data['ball'].nonstriker_ids.tolist()
        # Extract over numbers for same_over edges
        ball_overs_list = [d['_over'] for d in history]
    else:
        bowler_ids_list = []
        batsman_ids_list = []
        nonstriker_ids_list = []
        ball_overs_list = []

    # Build all edges with correct player attribution
    # This ensures cross-domain edges connect to the ACTUAL players
    # who faced/bowled each historical ball (respecting Z2 swap symmetry)
    edge_dict, edge_attr_dict = build_all_edges(
        num_balls=num_balls,
        bowler_ids=bowler_ids_list,
        batsman_ids=batsman_ids_list,
        nonstriker_ids=nonstriker_ids_list,
        current_striker_id=striker_id,
        current_bowler_id=bowler_id,
        current_nonstriker_id=nonstriker_id,
        ball_overs=ball_overs_list
    )

    # Add edges to HeteroData
    for edge_type, edge_index in edge_dict.items():
        data[edge_type].edge_index = edge_index

    # Add edge attributes where available (e.g., temporal distance for precedes edges)
    for edge_type, edge_attr in edge_attr_dict.items():
        data[edge_type].edge_attr = edge_attr

    # =========================================================================
    # 8. LABEL
    # =========================================================================

    data.y = torch.tensor([outcome_to_class(target_ball)], dtype=torch.long)

    # Store metadata for debugging and conditional routing
    data.match_id = info.get('match_type_number', 0)
    data.innings_num = innings_num
    data.ball_idx = ball_idx
    data.num_balls = num_balls

    # Is this a chase (2nd innings with target)? Used for innings-conditional prediction heads
    data.is_chase = torch.tensor([innings_num == 2 and target_score is not None], dtype=torch.bool)

    return data


def create_samples_from_innings(
    match_data: Dict,
    innings_idx: int,
    entity_mapper: EntityMapper,
    min_history: int = 1
) -> List[HeteroData]:
    """
    Create HeteroData samples for all predictable balls in an innings.

    Args:
        match_data: Full match JSON data
        innings_idx: Which innings (0 or 1)
        entity_mapper: Entity name to ID mapper
        min_history: Minimum balls of history required (default 1)

    Returns:
        List of HeteroData objects, one per ball
    """
    samples = []

    innings = match_data['innings'][innings_idx]
    all_deliveries = prepare_deliveries(innings)

    # Create sample for each ball (starting from min_history to ensure some context)
    for ball_idx in range(min_history, len(all_deliveries)):
        try:
            sample = create_hetero_data(
                match_data=match_data,
                innings_idx=innings_idx,
                ball_idx=ball_idx,
                entity_mapper=entity_mapper
            )
            samples.append(sample)
        except Exception as e:
            # Skip problematic balls
            print(f"Warning: Failed to create sample for innings {innings_idx}, "
                  f"ball {ball_idx}: {e}")

    return samples


def create_samples_from_match(
    match_data: Dict,
    entity_mapper: EntityMapper,
    min_history: int = 1
) -> List[HeteroData]:
    """
    Create HeteroData samples for all predictable balls in a match.

    Args:
        match_data: Full match JSON data
        entity_mapper: Entity name to ID mapper
        min_history: Minimum balls of history required

    Returns:
        List of HeteroData objects for entire match
    """
    samples = []

    for innings_idx in range(len(match_data.get('innings', []))):
        innings_samples = create_samples_from_innings(
            match_data=match_data,
            innings_idx=innings_idx,
            entity_mapper=entity_mapper,
            min_history=min_history
        )
        samples.extend(innings_samples)

    return samples


def get_node_feature_dims() -> Dict[str, int]:
    """
    Get input feature dimensions for each node type.

    Returns:
        Dict mapping node type to input dimension
    """
    return {
        # Entity nodes (ID to be embedded)
        'venue': 1,
        'batting_team': 1,
        'bowling_team': 1,
        'striker_identity': 1,
        'nonstriker_identity': 1,
        'bowler_identity': 1,
        # State nodes
        'score_state': 5,  # runs, wickets, balls, innings_indicator, is_womens_cricket
        'chase_state': 7,  # Enhanced with RRR details: runs_needed, rrr, is_chase, rrr_norm, difficulty, balls_rem, wickets_rem
        'phase_state': 6,  # is_powerplay, is_middle, is_death, over_progress, is_first_ball, is_super_over
        'time_pressure': 3,
        'wicket_buffer': 2,
        # Actor state nodes
        # striker_state: 8 features (7 base + balls_since_on_strike for cold restart)
        # nonstriker_state: 8 features (7 base + balls_since_as_nonstriker for Z2 symmetry)
        'striker_state': 8,  # +1 for is_debut_ball, +1 for balls_since_on_strike
        'nonstriker_state': 8,  # Z2 symmetric with striker: +1 for balls_since_as_nonstriker
        'bowler_state': 8,  # +2 for is_pace, is_spin bowling type indicators (P1.2)
        'partnership': 4,
        # Dynamics nodes
        'batting_momentum': 1,
        'bowling_momentum': 1,
        'pressure_index': 1,
        'dot_pressure': 5,  # consecutive_dots, balls_since_boundary, balls_since_wicket, pressure_accumulated, pressure_trend
        # Ball nodes (17 features + embeddings added in model):
        # - 5 basic: runs, is_wicket, over, ball_in_over, is_boundary
        # - 4 extras: is_wide, is_noball, is_bye, is_legbye
        # - 6 wicket types: bowled, caught, lbw, run_out, stumped, other
        # - 2 run-out attribution: striker_run_out, nonstriker_run_out
        'ball': BALL_FEATURE_DIM,
        # Query node (learned embedding)
        'query': 1,
    }


def validate_hetero_data(data: HeteroData) -> bool:
    """
    Validate that a HeteroData object is correctly formed.

    Args:
        data: HeteroData object to validate

    Returns:
        True if valid, raises AssertionError otherwise
    """
    expected_dims = get_node_feature_dims()

    # Check all node types exist with correct dimensions
    for node_type, expected_dim in expected_dims.items():
        assert node_type in data.node_types, f"Missing node type: {node_type}"

        if node_type == 'ball':
            # Ball can have 0 or more nodes
            if data[node_type].x.shape[0] > 0:
                assert data[node_type].x.shape[1] == expected_dim, \
                    f"Ball feature dim mismatch: {data[node_type].x.shape[1]} vs {expected_dim}"
        else:
            assert data[node_type].x.shape == (1, expected_dim), \
                f"{node_type} shape mismatch: {data[node_type].x.shape} vs (1, {expected_dim})"

    # Check label exists
    assert hasattr(data, 'y'), "Missing label"
    assert data.y.shape == (1,), f"Label shape mismatch: {data.y.shape}"
    assert 0 <= data.y.item() <= 6, f"Invalid label: {data.y.item()}"

    return True
