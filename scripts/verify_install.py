#!/usr/bin/env python
"""Verify that the cricketmodel environment is correctly installed."""

import os
import sys

# Ensure we're in the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, project_root)


def main():
    print("=== Verifying Cricket Model Installation ===\n")
    errors = []

    # Check Python version
    print(f"Python: {sys.version}")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ required")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("  MPS (Apple Silicon GPU): Available")
        elif torch.cuda.is_available():
            print(f"  CUDA: Available ({torch.cuda.get_device_name(0)})")
        else:
            print("  GPU: Not available (CPU only)")
    except ImportError as e:
        errors.append(f"PyTorch import failed: {e}")

    # Check PyTorch Geometric
    try:
        import torch_geometric
        print(f"PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch Geometric import failed: {e}")

    # Check NumPy version
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
        if np.__version__.startswith("2"):
            errors.append("NumPy 2.x detected - may cause issues with PyTorch")
    except ImportError as e:
        errors.append(f"NumPy import failed: {e}")

    # Check other dependencies
    deps = ["pandas", "sklearn", "tqdm", "yaml", "matplotlib", "seaborn"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            print(f"{dep}: {version}")
        except ImportError:
            errors.append(f"{dep} not installed")

    # Check project imports
    print("\n--- Project Imports ---")
    try:
        from src.data.feature_utils import BALL_FEATURE_DIM
        print(f"BALL_FEATURE_DIM: {BALL_FEATURE_DIM}")
    except ImportError as e:
        errors.append(f"feature_utils import failed: {e}")

    try:
        from src.data.edge_builder import NODE_TYPES, get_all_edge_types
        print(f"NODE_TYPES: {len(NODE_TYPES)}")
        print(f"Edge types: {len(get_all_edge_types())}")
    except ImportError as e:
        errors.append(f"edge_builder import failed: {e}")

    try:
        from src.model.hetero_gnn import CricketHeteroGNN, ModelConfig
        config = ModelConfig(
            num_venues=100,
            num_teams=20,
            num_players=1000,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            num_classes=7,
        )
        model = CricketHeteroGNN(config)
        params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {params:,}")
    except Exception as e:
        errors.append(f"Model creation failed: {e}")

    # Summary
    print("\n" + "=" * 40)
    if errors:
        print("ERRORS FOUND:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
