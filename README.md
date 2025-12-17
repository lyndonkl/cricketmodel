# Cricket Ball Prediction Model

Predict T20 cricket ball outcomes using a **unified heterogeneous graph neural network**. The V2 architecture models the entire match context as a single graph with typed nodes and edges, enabling full innings history through sparse O(n) edges instead of O(n²) attention.

## Architecture

```
Match Context Graph (21 node types, 16 edge types)
├── Global Layer:    venue, batting_team, bowling_team
├── State Layer:     score_state, chase_state, phase_state, time_pressure, wicket_buffer
├── Actor Layer:     striker_identity, striker_state, nonstriker_identity, nonstriker_state,
│                    bowler_identity, bowler_state, partnership
├── Dynamics Layer:  batting_momentum, bowling_momentum, pressure_index, dot_pressure
├── Ball Nodes:      Full innings history (N balls, 15 features each)
└── Query Node:      Aggregates information for prediction
```

### Key Features

- **Unified Graph**: All match context in one heterogeneous graph
- **Full History**: O(n) sparse edges enable complete innings history (vs V1's 24-ball window)
- **Non-Striker Modeling**: Partnership dynamics and strike rotation patterns
- **Rich Ball Features**: 15 features including extras and wicket types (bowled, caught, lbw, run out, stumped)
- **Typed Edges**: Different convolution operators per relationship type (GATv2Conv, TransformerConv, SAGEConv)
- **Temporal Edge Features**: Precedes edges include temporal distance for recency-weighted attention
- **Correct Player Attribution**: Cross-domain edges respect Z2 symmetry of striker/non-striker swap
- **Interpretable**: Attention weights show which context influences predictions

### Edge Types

| Category | Edges | Purpose |
|----------|-------|---------|
| **Hierarchical** | global→state→actor→dynamics | Top-down conditioning |
| **Intra-layer** | Within-layer interactions | E.g., striker↔bowler↔nonstriker matchup |
| **Temporal** | precedes, same_bowler, same_batsman, same_matchup | Ball history structure |
| **Cross-domain** | faced_by, partnered_by, bowled_by, informs | Connect history to context |
| **Query** | all→query | Aggregate for prediction |

## Quick Start

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- macOS, Linux, or Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/lyndonkl/cricketmodel.git
cd cricketmodel

# Run setup script (creates conda environment)
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate environment
conda activate cricketmodel

# Verify installation
python scripts/verify_install.py
```

### Manual Installation

If you prefer manual setup:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate
conda activate cricketmodel

# Verify
python scripts/verify_install.py
```

### Data Setup

Download T20 data from [Cricsheet](https://cricsheet.org/downloads/):

```bash
mkdir -p data
cd data
curl -O https://cricsheet.org/downloads/t20s_male_json.zip
unzip t20s_male_json.zip
cd ..
```

The data should be in `data/t20s_male_json/*.json`.

## Training

```bash
# Basic training
python train.py

# Custom settings
python train.py \
    --epochs 100 \
    --batch-size 64 \
    --hidden-dim 128 \
    --num-layers 3 \
    --lr 1e-3 \
    --device auto

# Test a trained model
python train.py --test-only
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | data/t20s_male_json | Raw JSON match files |
| `--batch-size` | 64 | Training batch size |
| `--hidden-dim` | 128 | Hidden dimension |
| `--num-layers` | 3 | Message passing layers |
| `--num-heads` | 4 | Attention heads |
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--patience` | 10 | Early stopping patience |
| `--device` | auto | Device: auto, cuda, mps, cpu |
| `--test-only` | False | Only evaluate existing model |

## Project Structure

```
cricketmodel/
├── src/
│   ├── data/
│   │   ├── entity_mapper.py       # Venue/team/player ID mapping
│   │   ├── feature_utils.py       # Feature computation for all node types
│   │   ├── edge_builder.py        # Graph structure and edge construction
│   │   ├── hetero_data_builder.py # PyG HeteroData creation
│   │   └── dataset.py             # CricketDataset class
│   ├── model/
│   │   ├── encoders.py            # Node encoders (entity, feature, ball, query)
│   │   ├── conv_builder.py        # HeteroConv with typed convolutions
│   │   └── hetero_gnn.py          # Main CricketHeteroGNN model
│   ├── training/
│   │   ├── trainer.py             # Training loop with early stopping
│   │   └── metrics.py             # Multi-class evaluation metrics
│   └── config.py                  # Configuration dataclasses
├── scripts/
│   ├── setup.sh                   # Automated environment setup
│   └── verify_install.py          # Installation verification
├── train.py                       # Main training script
├── environment.yml                # Conda environment specification
├── pyproject.toml                 # Package configuration
└── notes/
    └── architecture/
        └── v2-unified-heterograph/  # Design documentation
```

## Output Classes

The model predicts 7 outcome classes:

| Class | Outcome | Typical % |
|-------|---------|-----------|
| 0 | Dot | ~40% |
| 1 | Single | ~30% |
| 2 | Two | ~8% |
| 3 | Three | ~2% |
| 4 | Four | ~12% |
| 5 | Six | ~5% |
| 6 | Wicket | ~3% |

Class weights are automatically computed to handle imbalance.

## Documentation

Detailed design documentation in `notes/architecture/v2-unified-heterograph/`:

- `01-overview.md` - High-level architecture
- `02-node-types.md` - 21 node type specifications
- `03-edge-types.md` - 16 edge type specifications
- `04-model-architecture.md` - HeteroGNN implementation details
- `05-data-pipeline.md` - Dataset and DataLoader design
- `06-training.md` - Training procedure

## Troubleshooting

### Environment Issues

```bash
# Remove and recreate environment
conda env remove -n cricketmodel
./scripts/setup.sh
```

### Memory Issues

For large datasets, reduce batch size or use fewer message passing layers:

```bash
python train.py --batch-size 32 --num-layers 2
```

### Apple Silicon (M1/M2/M3)

The environment auto-detects MPS. Verify with:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Dependencies

Core dependencies managed via conda for reproducibility:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | Required |
| PyTorch | >=2.1,<2.3 | MPS/CUDA support |
| PyTorch Geometric | >=2.4 | Installed via pip (no conda pkg for ARM) |
| NumPy | >=1.24,<2 | Pinned for PyTorch compatibility |

See `environment.yml` for full specification.

## License

MIT
