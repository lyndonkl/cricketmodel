#!/usr/bin/env python3
"""
Validate feature dimensions are consistent across data loader and model.

This script checks that:
1. get_node_feature_dims() returns expected dimensions
2. Feature computation functions return correct number of features
3. ModelConfig.condition_dim matches phase_dim + chase_dim + resource_dim
4. Encoder dimensions match data loader output

Run from project root: python scripts/validate_dimensions.py
"""

import sys
import os
from pathlib import Path

# Add project root to path and set up src as a proper package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import from src as a package (preserves relative imports within src)
import src.data.feature_utils as feature_utils_module
import src.data.hetero_data_builder as hetero_data_builder_module
import src.model.hetero_gnn as hetero_gnn_module
import src.model.encoders as encoders_module


def validate_feature_utils():
    """Validate feature computation functions return correct dimensions."""
    print("\n=== Validating feature_utils.py ===")

    # Use pre-imported module
    compute_score_state = feature_utils_module.compute_score_state
    compute_chase_state = feature_utils_module.compute_chase_state
    compute_phase_state = feature_utils_module.compute_phase_state
    compute_time_pressure = feature_utils_module.compute_time_pressure
    compute_wicket_buffer = feature_utils_module.compute_wicket_buffer
    compute_striker_state = feature_utils_module.compute_striker_state
    compute_nonstriker_state = feature_utils_module.compute_nonstriker_state
    compute_bowler_state = feature_utils_module.compute_bowler_state
    compute_partnership = feature_utils_module.compute_partnership
    compute_dynamics = feature_utils_module.compute_dynamics
    compute_ball_features = feature_utils_module.compute_ball_features
    BALL_FEATURE_DIM = feature_utils_module.BALL_FEATURE_DIM

    # Test with sample data
    sample_deliveries = [
        {
            'batter': 'Player A',
            'bowler': 'Bowler X',
            'non_striker': 'Player B',
            'runs': {'batter': 1, 'total': 1, 'extras': 0},
            '_over': 0,
            '_ball_in_over': 0,
        },
        {
            'batter': 'Player B',
            'bowler': 'Bowler X',
            'non_striker': 'Player A',
            'runs': {'batter': 0, 'total': 0, 'extras': 0},
            '_over': 0,
            '_ball_in_over': 1,
        },
    ]

    checks = []

    # score_state: 5 features
    result = compute_score_state(sample_deliveries, 1, False)
    checks.append(("score_state", len(result), 5))

    # chase_state: 7 features
    result = compute_chase_state(sample_deliveries, 150, 2)
    checks.append(("chase_state", len(result), 7))

    # phase_state: 6 features
    result = compute_phase_state(12, False)
    checks.append(("phase_state", len(result), 6))

    # time_pressure: 3 features
    result = compute_time_pressure(12, 150, 50, 2)
    checks.append(("time_pressure", len(result), 3))

    # wicket_buffer: 2 features
    result = compute_wicket_buffer(3)
    checks.append(("wicket_buffer", len(result), 2))

    # striker_state: 8 features
    result = compute_striker_state(sample_deliveries, 'Player A')
    checks.append(("striker_state", len(result), 8))

    # nonstriker_state: 8 features (Z2 symmetric with striker)
    result = compute_nonstriker_state(sample_deliveries, 'Player B')
    checks.append(("nonstriker_state", len(result), 8))

    # bowler_state: 6 features
    result = compute_bowler_state(sample_deliveries, 'Bowler X')
    checks.append(("bowler_state", len(result), 6))

    # partnership: 4 features
    result = compute_partnership(sample_deliveries, 'Player A', 'Player B')
    checks.append(("partnership", len(result), 4))

    # dynamics: dict with specific dimensions
    result = compute_dynamics(sample_deliveries, None, 1)
    checks.append(("batting_momentum", len(result['batting_momentum']), 1))
    checks.append(("bowling_momentum", len(result['bowling_momentum']), 1))
    checks.append(("pressure_index", len(result['pressure_index']), 1))
    checks.append(("dot_pressure", len(result['dot_pressure']), 3))

    # ball features: 18 features
    result = compute_ball_features(sample_deliveries[0])
    checks.append(("ball_features", len(result), 18))
    checks.append(("BALL_FEATURE_DIM", BALL_FEATURE_DIM, 18))

    # Print results
    all_passed = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"  {name}: {actual} (expected {expected}) [{status}]")

    return all_passed


def validate_hetero_data_builder():
    """Validate get_node_feature_dims() returns expected dimensions."""
    print("\n=== Validating hetero_data_builder.py ===")

    # Use pre-imported module
    get_node_feature_dims = hetero_data_builder_module.get_node_feature_dims
    dims = get_node_feature_dims()

    expected = {
        'venue': 1,
        'batting_team': 1,
        'bowling_team': 1,
        'striker_identity': 1,
        'nonstriker_identity': 1,
        'bowler_identity': 1,
        'score_state': 5,
        'chase_state': 7,
        'phase_state': 6,
        'time_pressure': 3,
        'wicket_buffer': 2,
        'striker_state': 8,
        'nonstriker_state': 8,
        'bowler_state': 6,
        'partnership': 4,
        'batting_momentum': 1,
        'bowling_momentum': 1,
        'pressure_index': 1,
        'dot_pressure': 3,
        'ball': 18,
        'query': 1,
    }

    all_passed = True
    for node_type, expected_dim in expected.items():
        actual_dim = dims.get(node_type)
        status = "PASS" if actual_dim == expected_dim else "FAIL"
        if actual_dim != expected_dim:
            all_passed = False
        print(f"  {node_type}: {actual_dim} (expected {expected_dim}) [{status}]")

    return all_passed


def validate_model_config():
    """Validate ModelConfig dimensions."""
    print("\n=== Validating hetero_gnn.py ModelConfig ===")

    # Use pre-imported module
    ModelConfig = hetero_gnn_module.ModelConfig
    config = ModelConfig(num_venues=1, num_teams=1, num_players=1)

    checks = [
        ("phase_dim", config.phase_dim, 6),
        ("chase_dim", config.chase_dim, 7),
        ("resource_dim", config.resource_dim, 2),
        ("condition_dim", config.condition_dim, 15),
    ]

    # Verify condition_dim = phase_dim + chase_dim + resource_dim
    expected_condition = config.phase_dim + config.chase_dim + config.resource_dim
    checks.append(("condition_dim = phase + chase + resource", config.condition_dim, expected_condition))

    all_passed = True
    for name, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_passed = False
        print(f"  {name}: {actual} (expected {expected}) [{status}]")

    return all_passed


def validate_encoders():
    """Validate encoder dimensions match data loader output."""
    print("\n=== Validating encoders.py ===")

    import torch
    # Use pre-imported module
    NodeEncoderDict = encoders_module.NodeEncoderDict

    # Create encoder with minimal config
    encoder = NodeEncoderDict(
        num_venues=10,
        num_teams=10,
        num_players=100,
        venue_embed_dim=32,
        team_embed_dim=32,
        player_embed_dim=64,
        role_embed_dim=16,
        hidden_dim=128,
        dropout=0.1,
        use_hierarchical_player=True,
    )

    # Check feature encoder dimensions
    expected_dims = {
        'score_state': 5,
        'chase_state': 7,
        'phase_state': 6,
        'time_pressure': 3,
        'wicket_buffer': 2,
        'striker_state': 8,
        'nonstriker_state': 8,
        'bowler_state': 6,
        'partnership': 4,
        'batting_momentum': 1,
        'bowling_momentum': 1,
        'pressure_index': 1,
        'dot_pressure': 3,
    }

    all_passed = True
    for name, expected_dim in expected_dims.items():
        fe = encoder.feature_encoders[name]
        # FeatureEncoder uses self.encoder which is nn.Sequential, first layer is Linear
        actual_dim = fe.encoder[0].in_features
        status = "PASS" if actual_dim == expected_dim else "FAIL"
        if actual_dim != expected_dim:
            all_passed = False
        print(f"  {name}: {actual_dim} (expected {expected_dim}) [{status}]")

    # Check ball encoder
    # Ball encoder takes 18 features + 2 * player_embed_dim (64) = 18 + 128 = 146
    ball_input_dim = encoder.ball_encoder.projection[0].in_features
    expected_ball_input = 18 + 2 * 64  # 146
    status = "PASS" if ball_input_dim == expected_ball_input else "FAIL"
    if ball_input_dim != expected_ball_input:
        all_passed = False
    print(f"  ball_encoder input: {ball_input_dim} (expected {expected_ball_input}) [{status}]")

    return all_passed


def main():
    print("=" * 60)
    print("Cricket Model Dimension Validation")
    print("=" * 60)

    results = []

    results.append(("feature_utils.py", validate_feature_utils()))
    results.append(("hetero_data_builder.py", validate_hetero_data_builder()))
    results.append(("hetero_gnn.py ModelConfig", validate_model_config()))
    results.append(("encoders.py", validate_encoders()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name}: {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
