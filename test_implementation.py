#!/usr/bin/env python3
"""
Test script to validate the V2 implementation.

Run this to ensure all components can be imported and basic
functionality works before full training.
"""

import json
import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.data import (
            EntityMapper,
            compute_score_state,
            compute_ball_features,
            outcome_to_class,
            NODE_TYPES,
            LAYER_NODES,
            build_all_edges,
            create_hetero_data,
            get_node_feature_dims,
        )
        print("  ✓ Data modules imported")
    except ImportError as e:
        print(f"  ✗ Data import failed: {e}")
        return False

    try:
        from src.model import (
            EntityEncoder,
            FeatureEncoder,
            BallEncoder,
            QueryEncoder,
            NodeEncoderDict,
            build_hetero_conv,
            CricketHeteroGNN,
            ModelConfig,
        )
        print("  ✓ Model modules imported")
    except ImportError as e:
        print(f"  ✗ Model import failed: {e}")
        return False

    try:
        from src.training import (
            Trainer,
            TrainingConfig,
            compute_metrics,
        )
        print("  ✓ Training modules imported")
    except ImportError as e:
        print(f"  ✗ Training import failed: {e}")
        return False

    try:
        from src.config import Config, get_device, set_seed
        print("  ✓ Config module imported")
    except ImportError as e:
        print(f"  ✗ Config import failed: {e}")
        return False

    return True


def test_entity_mapper():
    """Test EntityMapper functionality."""
    print("\nTesting EntityMapper...")

    from src.data import EntityMapper

    mapper = EntityMapper()

    # Test venue
    venue_id = mapper.get_venue_id("MCG", create=True)
    assert venue_id == 1, f"Expected 1, got {venue_id}"
    assert mapper.get_venue_id("MCG") == 1
    assert mapper.get_venue_id("Unknown") == 0  # Unknown returns 0
    print("  ✓ Venue mapping works")

    # Test team
    team_id = mapper.get_team_id("Australia", create=True)
    assert team_id == 1
    print("  ✓ Team mapping works")

    # Test player
    player_id = mapper.get_player_id("V Kohli", create=True)
    assert player_id == 1
    print("  ✓ Player mapping works")

    # Test counts
    assert mapper.num_venues == 1
    assert mapper.num_teams == 1
    assert mapper.num_players == 1
    print("  ✓ Counts correct")

    return True


def test_feature_computation():
    """Test feature computation functions."""
    print("\nTesting feature computation...")

    from src.data import (
        compute_score_state,
        compute_chase_state,
        compute_phase_state,
        compute_ball_features,
        outcome_to_class,
    )

    # Test score state
    score = compute_score_state([], 1)
    assert len(score) == 4, f"Expected 4 features, got {len(score)}"
    print("  ✓ compute_score_state works")

    # Test chase state
    chase = compute_chase_state([], None, 1)
    assert len(chase) == 3
    print("  ✓ compute_chase_state works")

    # Test phase state
    phase = compute_phase_state(30)  # 30 balls = over 5
    assert len(phase) == 4
    assert phase[0] == 1.0  # Still in powerplay
    print("  ✓ compute_phase_state works")

    # Test ball features
    sample_delivery = {
        'runs': {'total': 4, 'batter': 4},
        '_over': 5,
        '_ball_in_over': 3,
    }
    ball_feat = compute_ball_features(sample_delivery)
    assert len(ball_feat) == 5
    print("  ✓ compute_ball_features works")

    # Test outcome classification
    dot_delivery = {'runs': {'total': 0, 'batter': 0}}
    assert outcome_to_class(dot_delivery) == 0

    wicket_delivery = {'runs': {'total': 0, 'batter': 0}, 'wickets': [{}]}
    assert outcome_to_class(wicket_delivery) == 6

    six_delivery = {'runs': {'total': 6, 'batter': 6}}
    assert outcome_to_class(six_delivery) == 5
    print("  ✓ outcome_to_class works")

    return True


def test_edge_building():
    """Test edge building functions."""
    print("\nTesting edge building...")

    from src.data import build_all_edges, NODE_TYPES

    # Build edges with no history
    edges = build_all_edges(num_balls=0, bowler_ids=[], batsman_ids=[])
    assert isinstance(edges, dict)
    print(f"  ✓ Built {len(edges)} edge types with 0 balls")

    # Build edges with some history
    bowler_ids = [1, 1, 2, 2, 1]  # 5 balls, 2 bowlers
    batsman_ids = [1, 1, 1, 2, 2]  # 2 batsmen
    edges = build_all_edges(num_balls=5, bowler_ids=bowler_ids, batsman_ids=batsman_ids)

    # Check temporal edges exist
    assert ('ball', 'precedes', 'ball') in edges
    precedes_edges = edges[('ball', 'precedes', 'ball')]
    assert precedes_edges.shape[1] == 4  # 4 sequential edges for 5 balls
    print("  ✓ Temporal edges correct")

    # Check same_bowler edges exist
    same_bowler = edges[('ball', 'same_bowler', 'ball')]
    assert same_bowler.shape[1] > 0
    print("  ✓ Same bowler edges created")

    return True


def test_hetero_data_creation():
    """Test HeteroData creation from match data."""
    print("\nTesting HeteroData creation...")

    from src.data import EntityMapper, create_hetero_data, validate_hetero_data

    # Load a sample match
    match_files = list(Path("data/t20s_male_json").glob("*.json"))
    if len(match_files) == 0:
        print("  ⚠ No match files found, skipping HeteroData test")
        return True

    with open(match_files[0], 'r') as f:
        match_data = json.load(f)

    # Build entity mapper
    mapper = EntityMapper()
    mapper.build_from_matches([match_files[0]])
    print(f"  Built mapper: {mapper}")

    # Create HeteroData for ball 10 of innings 0
    data = create_hetero_data(
        match_data=match_data,
        innings_idx=0,
        ball_idx=10,
        entity_mapper=mapper
    )

    # Validate
    validate_hetero_data(data)
    print(f"  ✓ Created valid HeteroData with {data.num_balls} ball nodes")

    # Check structure
    assert 'venue' in data.node_types
    assert 'query' in data.node_types
    assert 'ball' in data.node_types
    assert hasattr(data, 'y')
    print("  ✓ HeteroData structure correct")

    return True


def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")

    import torch
    from src.model import CricketHeteroGNN, ModelConfig, count_parameters

    config = ModelConfig(
        num_venues=100,
        num_teams=20,
        num_players=500,
        hidden_dim=64,  # Small for testing
        num_layers=2,
        num_heads=4,
    )

    model = CricketHeteroGNN(config)
    num_params = count_parameters(model)
    print(f"  ✓ Created model with {num_params:,} parameters")

    return True


def test_model_forward():
    """Test model forward pass with real data."""
    print("\nTesting model forward pass...")

    import torch
    from src.data import EntityMapper, create_hetero_data
    from src.model import CricketHeteroGNN, ModelConfig

    # Load sample match
    match_files = list(Path("data/t20s_male_json").glob("*.json"))
    if len(match_files) == 0:
        print("  ⚠ No match files found, skipping forward pass test")
        return True

    with open(match_files[0], 'r') as f:
        match_data = json.load(f)

    # Build mapper
    mapper = EntityMapper()
    mapper.build_from_matches([match_files[0]])

    # Create sample data
    data = create_hetero_data(
        match_data=match_data,
        innings_idx=0,
        ball_idx=10,
        entity_mapper=mapper
    )

    # Create model
    config = ModelConfig(
        num_venues=mapper.num_venues,
        num_teams=mapper.num_teams,
        num_players=mapper.num_players,
        hidden_dim=32,  # Small for testing
        num_layers=2,
        num_heads=2,
    )
    model = CricketHeteroGNN(config)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(data)

    assert logits.shape == (1, 7), f"Expected (1, 7), got {logits.shape}"
    print(f"  ✓ Forward pass successful, output shape: {logits.shape}")

    # Check predictions are valid
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    print("  ✓ Output probabilities sum to 1")

    return True


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")

    from src.training import compute_metrics, print_classification_report
    import numpy as np

    # Sample predictions
    labels = [0, 1, 2, 0, 1, 4, 6, 0, 1, 2]
    preds = [0, 1, 1, 0, 1, 4, 0, 1, 1, 2]
    probs = np.eye(7)[preds]  # One-hot as probabilities

    metrics = compute_metrics(labels, preds, probs)

    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert 'confusion_matrix' in metrics
    print(f"  ✓ Accuracy: {metrics['accuracy']:.3f}")
    print(f"  ✓ Macro F1: {metrics['f1_macro']:.3f}")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cricket Model V2 Implementation Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("EntityMapper", test_entity_mapper),
        ("Feature Computation", test_feature_computation),
        ("Edge Building", test_edge_building),
        ("HeteroData Creation", test_hetero_data_creation),
        ("Model Creation", test_model_creation),
        ("Model Forward", test_model_forward),
        ("Metrics", test_metrics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
