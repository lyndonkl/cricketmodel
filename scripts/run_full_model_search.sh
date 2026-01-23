#!/bin/bash
# Hyperparameter search for CricketHeteroGNNFull model only
# Uses wandb + optuna for tracking and optimization
# Usage: ./scripts/run_full_model_search.sh

set -e

# Change to project root (parent of scripts directory)
cd "$(dirname "$0")/.."
echo "Running from: $(pwd)"

# Configuration - adjust these for your environment
NUM_GPUS=${NUM_GPUS:-6}           # Number of GPUs (override with env var)
STAGGER_DELAY=15                   # seconds between each GPU start
DATA_FRACTION=0.02                 # 2% of data for fast iteration
BATCH_SIZE=1024                    # Larger batch for faster epochs
TRIALS_PER_GPU=15                  # Total trials = NUM_GPUS * TRIALS_PER_GPU
EPOCHS=10
DEVICE=${DEVICE:-cuda}             # Device: cuda, cpu, or mps

# Set file descriptor limit for PyTorch multiprocessing
ulimit -n 65535 2>/dev/null || true

# Clean up previous studies
rm -f optuna_full_model.db
rm -rf checkpoints/optuna/cricket_gnn_full_model_*
echo "Cleaned up previous Optuna studies and checkpoints"

# Create base params file to fix model_class to CricketHeteroGNNFull
BASE_PARAMS_FILE=$(mktemp)
cat > "$BASE_PARAMS_FILE" << 'EOF'
{
    "best_params": {
        "model_class": "CricketHeteroGNNFull"
    }
}
EOF
echo "Created base params file: $BASE_PARAMS_FILE"

# Shared study name for coordination across GPUs
STUDY_NAME="cricket_gnn_full_model_$(date +%Y%m%d_%H%M%S)"

echo "=== Running Full Model HP Search ==="
echo "Study name: $STUDY_NAME"
echo "Device: $DEVICE"
echo "Model: CricketHeteroGNNFull (fixed)"

if [ "$DEVICE" = "cuda" ]; then
    echo "GPUs: $NUM_GPUS"
    echo "Total trials: $((NUM_GPUS * TRIALS_PER_GPU))"
    echo ""

    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        echo "Starting GPU $gpu..."
        CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
            --phase full_model_only \
            --n-trials $TRIALS_PER_GPU \
            --epochs $EPOCHS \
            --data-fraction $DATA_FRACTION \
            --batch-size $BATCH_SIZE \
            --study-name "$STUDY_NAME" \
            --storage "sqlite:///optuna_full_model.db" \
            --best-params "$BASE_PARAMS_FILE" \
            --wandb --device cuda --n-jobs 1 &
        sleep $STAGGER_DELAY
    done
    wait
else
    # Single device mode (CPU or MPS)
    TOTAL_TRIALS=$((NUM_GPUS * TRIALS_PER_GPU))
    echo "Total trials: $TOTAL_TRIALS"
    echo ""

    python scripts/hp_search.py \
        --phase full_model_only \
        --n-trials $TOTAL_TRIALS \
        --epochs $EPOCHS \
        --data-fraction $DATA_FRACTION \
        --batch-size $BATCH_SIZE \
        --study-name "$STUDY_NAME" \
        --storage "sqlite:///optuna_full_model.db" \
        --best-params "$BASE_PARAMS_FILE" \
        --wandb --device $DEVICE --n-jobs 1
fi

# Cleanup temp file
rm -f "$BASE_PARAMS_FILE"

echo "=== Full Model HP Search Complete! ==="

# Find and display best params
BEST_PARAMS=$(ls -t checkpoints/optuna/cricket_gnn_full_model_*/best_params.json 2>/dev/null | head -1)
if [ -z "$BEST_PARAMS" ]; then
    echo "ERROR: No best_params.json found!"
    exit 1
fi

echo ""
echo "Best params: $BEST_PARAMS"
cat "$BEST_PARAMS"

echo ""
echo "=== Done! ==="
echo ""
echo "To train the final model with these params, run:"
echo "  python train.py --config \"$BEST_PARAMS\" --epochs 100 --wandb"
