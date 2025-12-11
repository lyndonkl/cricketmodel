# Cricket Ball Prediction Model

Predict T20 cricket ball outcomes using hierarchical graph attention and temporal transformers. Designed for **interpretable attention** that an LLM can observe and translate into real-time match insights.

## Architecture

```
Input (per ball) → Hierarchical GAT (17 nodes) → Temporal Transformer → Prediction
                          ↓                              ↓
                   Within-ball attention         Cross-ball attention
                   (what factors matter)         (which past balls matter)
```

### Hierarchical Graph Structure

17 semantic nodes across 4 layers with learned attention:

| Layer | Nodes | Purpose |
|-------|-------|---------|
| **Global** | Venue, Batting Team, Bowling Team | Match-level context |
| **State** | Score, Chase, Phase, Time, Wickets | Current situation |
| **Actor** | Striker ID/State, Bowler ID/State, Partnership | Player dynamics |
| **Dynamics** | Batting Momentum, Bowling Momentum, Pressure, Dot Pressure | Recent trends |

### Temporal Transformer

Specialized attention heads for cricket-specific patterns:
- **Recency head**: Recent balls weighted higher
- **Same-bowler head**: Pattern from current bowler's deliveries
- **Same-batsman head**: Current batsman's recent form
- **Learned heads**: Free to discover other patterns

## Installation

```bash
# Clone and install
git clone https://github.com/lyndonkl/cricketmodel.git
cd cricketmodel
pip install -e .

# Download Cricsheet T20 data
# Place JSON files in data/t20s_male_json/
```

**Requirements**: Python 3.10+, PyTorch 2.1+, PyTorch Geometric 2.4+

## Training

### Single GPU / Apple Silicon (MPS)

```bash
python train.py --device mps --epochs 50 --batch-size 64
```

### Multi-GPU with DDP

```bash
torchrun --nproc_per_node=2 train.py --distributed --epochs 50
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | mps | Device: mps, cuda, cpu |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--distributed` | False | Enable DDP |

## Project Structure

```
src/
├── data/           # Cricsheet JSON loading, PyTorch Dataset
├── features/       # Match state, derived features (pressure, momentum)
├── embeddings/     # Player/venue embeddings with cold-start handling
├── model/
│   ├── hierarchical.py   # 4-layer GAT with attention extraction
│   ├── temporal.py       # Transformer with specialized heads
│   └── predictor.py      # Full model combining both
└── training/       # Trainer with MPS/DDP support
```

## Interpretability

The model outputs attention weights at multiple levels:

```python
output = model(batch, return_attention=True)
print(output["gat_attention"]["layer_importance"])
# {'global': 0.12, 'match_state': 0.38, 'actor': 0.28, 'dynamics': 0.22}
```

This enables an LLM to generate insights like:
> "The model predicts a single with 28% confidence. Focus is heavily on the chase equation (42%) and current pressure (38%). The model looked at this bowler's previous deliveries where 2 boundaries were scored."

## Cold-Start Handling

Unknown players/venues are handled via feature-based embedding generation:

```python
# The model never sees player IDs - only embedding vectors
# generated from stats. New players get embeddings from:
# 1. Career stats (if available)
# 2. Role prototypes (opener_aggressive, death_pace, etc.)
# 3. Country/position inference
```

See `notes/implementation/08-cold-start-embeddings.md` for details.

## Documentation

Detailed design docs in `notes/implementation/`:
- `02-graph-structure.md` - 17-node semantic graph design
- `03-hierarchical-attention.md` - Layer-wise attention with conditioning
- `04-temporal-attention.md` - Specialized attention heads
- `07-live-data-contract.md` - API for live predictions

## License

MIT
