#!/bin/bash
# Joint HP search: model architecture + all hyperparameters together
# Avoids confounding problem of phased search
# Usage: ./scripts/run_joint_search.sh

set -e

# Change to project root (parent of scripts directory)
cd "$(dirname "$0")/.."
echo "Running from: $(pwd)"

NUM_GPUS=6
STAGGER_DELAY=15          # seconds between each GPU start to avoid SQLite race conditions
DATA_FRACTION=0.02        # 2% of data for fast iteration
BATCH_SIZE=1024           # Larger batch for faster epochs (4090 has 24GB VRAM)
TRIALS_PER_GPU=15         # 90 total trials across 6 GPUs (~2 hour search)
EPOCHS=10

# Set file descriptor limit for PyTorch multiprocessing
ulimit -n 65535

# Clean up previous studies
rm -f optuna_studies.db
rm -rf checkpoints/optuna/*
echo "Cleaned up previous Optuna studies and checkpoints"

# Shared study name for coordination across GPUs
STUDY_NAME="cricket_gnn_joint_$(date +%Y%m%d_%H%M%S)"

echo "=== Running Joint HP Search with $NUM_GPUS GPUs ==="
echo "Study name: $STUDY_NAME"
echo "Total trials: $((NUM_GPUS * TRIALS_PER_GPU))"
echo ""

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    echo "Starting GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
        --phase full_with_model \
        --n-trials $TRIALS_PER_GPU \
        --epochs $EPOCHS \
        --data-fraction $DATA_FRACTION \
        --batch-size $BATCH_SIZE \
        --study-name "$STUDY_NAME" \
        --wandb --device cuda --n-jobs 1 &
    sleep $STAGGER_DELAY
done
wait

echo "=== Joint HP Search Complete! ==="

# Find and display best params
BEST_PARAMS=$(ls -t checkpoints/optuna/cricket_gnn_joint_*/best_params.json 2>/dev/null | head -1)
if [ -z "$BEST_PARAMS" ]; then
    echo "ERROR: No best_params.json found!"
    exit 1
fi

echo ""
echo "Best params: $BEST_PARAMS"
cat "$BEST_PARAMS"

# Train final model with all 6 GPUs using DDP
echo ""
echo "=== Training final model with torchrun (6 GPUs) ==="
torchrun --standalone --nproc_per_node=6 train.py \
    --config "$BEST_PARAMS" \
    --epochs 100 \
    --batch-size 1024 \
    --wandb \
    --wandb-project cricket-gnn-final

echo "=== Done! ==="
