#!/bin/bash
# Training wrapper script that sets required environment variables
# before launching torchrun for distributed training.
#
# Usage:
#   ./scripts/train.sh                    # Default: 4 workers, 100 epochs
#   ./scripts/train.sh --nproc 14         # Use 14 CPU cores
#   ./scripts/train.sh --nproc 14 --wandb # With WandB logging
#   ./scripts/train.sh --help             # Show all options
#
# Note: DDP with torchrun automatically uses CPU on non-CUDA systems (Gloo backend).
#       For single-device MPS training, run directly: python train.py --wandb
#
# Environment variables set:
#   KMP_DUPLICATE_LIB_OK=TRUE  - Fixes OpenMP duplicate library error on macOS
#                                when mixing conda packages with pip-installed PyTorch
#   PYTORCH_ENABLE_MPS_FALLBACK=1 - Falls back to CPU for MPS-unsupported ops
#                                   (e.g., scatter_reduce in PyTorch Geometric)

set -e

# Required environment variables for macOS with conda + pip PyTorch
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Default values
NPROC=4
TRAIN_ARGS=""

# Parse wrapper-specific arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC="$2"
            shift 2
            ;;
        --help|-h)
            echo "Training wrapper script"
            echo ""
            echo "Wrapper options:"
            echo "  --nproc N    Number of processes for torchrun (default: 4)"
            echo ""
            echo "All other arguments are passed to train.py:"
            python train.py --help
            exit 0
            ;;
        *)
            TRAIN_ARGS="$TRAIN_ARGS $1"
            shift
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Environment variables set:"
echo "  KMP_DUPLICATE_LIB_OK=$KMP_DUPLICATE_LIB_OK"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=$PYTORCH_ENABLE_MPS_FALLBACK"
echo ""
echo "Running: torchrun --nproc_per_node=$NPROC train.py$TRAIN_ARGS"
echo ""

exec torchrun --nproc_per_node="$NPROC" train.py $TRAIN_ARGS
