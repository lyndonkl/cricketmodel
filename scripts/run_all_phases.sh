#!/bin/bash
# Run all HP search phases with 4 GPUs, then train final model
# Usage: ./scripts/run_all_phases.sh [--resume] (can run from any directory)
#   --resume: Skip phases that already have best_params.json, pick up where left off
set -e

# Change to project root (parent of scripts directory)
cd "$(dirname "$0")/.."
echo "Running from: $(pwd)"

NUM_GPUS=4
STAGGER_DELAY=15  # seconds between each GPU start to avoid SQLite race conditions
DATA_FRACTION=0.02  # Use 2% of data for faster trials (use 1.0 for full dataset)
BATCH_SIZE=256  # Larger batch size for faster epochs (default was 64)

# Check for --resume flag
RESUME=false
if [ "$1" == "--resume" ]; then
    RESUME=true
    echo "Resume mode: will skip phases with existing results"
fi

# Set file descriptor limit for PyTorch multiprocessing
ulimit -n 65535

# Only clean up if not resuming
if [ "$RESUME" = false ]; then
    rm -f optuna_studies.db
    rm -rf checkpoints/optuna/*
    echo "Cleaned up previous Optuna studies and checkpoints"
else
    echo "Keeping existing checkpoints for resume"
fi

# Function to find existing best params for a phase
find_best_params() {
    local phase_pattern=$1
    local best_file=$(ls -t checkpoints/optuna/cricket_gnn_${phase_pattern}_*/best_params.json 2>/dev/null | head -1)
    echo "$best_file"
}

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

# Phase 1: Model variants FIRST - 12 trials (3 per GPU)
# Compare different model architectures with default params to find the best architecture
PHASE1_BEST=$(find_best_params "model_variants")
if [ "$RESUME" = true ] && [ -n "$PHASE1_BEST" ]; then
    echo "=== Skipping model_variants (found existing: $PHASE1_BEST) ==="
else
    run_phase "model_variants" 3 10 ""
    PHASE1_BEST=$(ls -t checkpoints/optuna/cricket_gnn_model_variants_*/best_params.json | head -1)
fi
echo "Phase 1 (model_variants) best: $PHASE1_BEST"

# Phase 2: Coarse search - 12 trials (3 per GPU)
# Now tune hidden_dim and lr for the winning model architecture
PHASE2_BEST=$(find_best_params "phase1_coarse")
if [ "$RESUME" = true ] && [ -n "$PHASE2_BEST" ]; then
    echo "=== Skipping phase1_coarse (found existing: $PHASE2_BEST) ==="
else
    run_phase "phase1_coarse" 3 10 "$PHASE1_BEST"
    PHASE2_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase1_coarse_*/best_params.json | head -1)
fi
echo "Phase 2 (coarse) best: $PHASE2_BEST"

# Phase 3: Architecture tuning - 12 trials (3 per GPU)
PHASE3_BEST=$(find_best_params "phase2_architecture")
if [ "$RESUME" = true ] && [ -n "$PHASE3_BEST" ]; then
    echo "=== Skipping phase2_architecture (found existing: $PHASE3_BEST) ==="
else
    run_phase "phase2_architecture" 3 10 "$PHASE2_BEST"
    PHASE3_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase2_architecture_*/best_params.json | head -1)
fi
echo "Phase 3 (architecture) best: $PHASE3_BEST"

# Phase 4: Training dynamics - 16 trials (4 per GPU)
PHASE4_BEST=$(find_best_params "phase3_training")
if [ "$RESUME" = true ] && [ -n "$PHASE4_BEST" ]; then
    echo "=== Skipping phase3_training (found existing: $PHASE4_BEST) ==="
else
    run_phase "phase3_training" 4 10 "$PHASE3_BEST"
    PHASE4_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase3_training_*/best_params.json | head -1)
fi
echo "Phase 4 (training) best: $PHASE4_BEST"

# Phase 5: Loss function - 12 trials (3 per GPU)
PHASE5_BEST=$(find_best_params "phase4_loss")
if [ "$RESUME" = true ] && [ -n "$PHASE5_BEST" ]; then
    echo "=== Skipping phase4_loss (found existing: $PHASE5_BEST) ==="
else
    run_phase "phase4_loss" 3 10 "$PHASE4_BEST"
    PHASE5_BEST=$(ls -t checkpoints/optuna/cricket_gnn_phase4_loss_*/best_params.json | head -1)
fi
echo "Phase 5 (loss) best: $PHASE5_BEST"

echo ""
echo "=== All HP search phases complete! ==="
echo "Final best params: $PHASE5_BEST"
cat "$PHASE5_BEST"

# Train final model with all 4 GPUs using DDP
echo ""
echo "=== Training final model with torchrun (4 GPUs) ==="
torchrun --standalone --nproc_per_node=4 train.py \
    --config "$PHASE5_BEST" \
    --epochs 100 \
    --wandb \
    --wandb-project cricket-gnn-final

echo "=== Done! ==="
