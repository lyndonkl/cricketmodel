#!/bin/bash
# Run all HP search phases with 4 GPUs, then train final model
# Usage: ./scripts/run_all_phases.sh (can run from any directory)
set -e

# Change to project root (parent of scripts directory)
cd "$(dirname "$0")/.."
echo "Running from: $(pwd)"

NUM_GPUS=4
STAGGER_DELAY=15  # seconds between each GPU start to avoid SQLite race conditions
DATA_FRACTION=0.02  # Use 2% of data for faster trials (use 1.0 for full dataset)
BATCH_SIZE=256  # Larger batch size for faster epochs (default was 64)

# Set file descriptor limit for PyTorch multiprocessing
ulimit -n 65535

# Clean up previous Optuna database
rm -f optuna_studies.db

run_phase() {
    local phase=$1
    local trials_per_gpu=$2
    local epochs=$3
    local best_params=$4

    # Shared study name so all GPUs coordinate trials
    local study_name="cricket_gnn_${phase}_$(date +%Y%m%d_%H%M%S)"

    echo "=== Running $phase with $NUM_GPUS GPUs (staggered start) ==="
    echo "Study name: $study_name"

    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        echo "Starting GPU $gpu..."
        if [ -z "$best_params" ]; then
            CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
                --phase $phase --n-trials $trials_per_gpu --epochs $epochs \
                --data-fraction $DATA_FRACTION \
                --batch-size $BATCH_SIZE \
                --study-name "$study_name" \
                --wandb --device cuda --n-jobs 1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu python scripts/hp_search.py \
                --phase $phase --n-trials $trials_per_gpu --epochs $epochs \
                --data-fraction $DATA_FRACTION \
                --batch-size $BATCH_SIZE \
                --study-name "$study_name" \
                --best-params "$best_params" \
                --wandb --device cuda --n-jobs 1 &
        fi
        sleep $STAGGER_DELAY
    done
    wait
    echo "=== $phase complete! ==="
}

# Phase 1: 12 trials (3 per GPU)
run_phase "phase1_coarse" 3 25 ""
PHASE1_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase1_coarse_*/best_params.json | head -1)
echo "Phase 1 best: $PHASE1_BEST"

# Phase 2: 12 trials (3 per GPU)
run_phase "phase2_architecture" 3 25 "$PHASE1_BEST"
PHASE2_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase2_architecture_*/best_params.json | head -1)
echo "Phase 2 best: $PHASE2_BEST"

# Phase 3: 16 trials (4 per GPU)
run_phase "phase3_training" 4 30 "$PHASE2_BEST"
PHASE3_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase3_training_*/best_params.json | head -1)
echo "Phase 3 best: $PHASE3_BEST"

# Phase 4: 12 trials (3 per GPU)
run_phase "phase4_loss" 3 30 "$PHASE3_BEST"
PHASE4_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase4_loss_*/best_params.json | head -1)
echo "Phase 4 best: $PHASE4_BEST"

echo ""
echo "=== All HP search phases complete! ==="
echo "Final best params: $PHASE4_BEST"
cat "$PHASE4_BEST"

# Train final model with all 4 GPUs using DDP
echo ""
echo "=== Training final model with torchrun (4 GPUs) ==="
torchrun --standalone --nproc_per_node=4 train.py \
    --config "$PHASE4_BEST" \
    --epochs 100 \
    --wandb \
    --wandb-project cricket-gnn-final

echo "=== Done! ==="
