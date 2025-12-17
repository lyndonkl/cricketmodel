#!/bin/bash
# Setup script for cricketmodel environment
# Usage: ./scripts/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Cricket Model Environment Setup ==="
echo "Project directory: $PROJECT_DIR"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "  macOS: brew install --cask miniconda"
    echo "  Or download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^cricketmodel "; then
    echo "Removing existing cricketmodel environment..."
    conda env remove -n cricketmodel -y
fi

# Create environment
echo "Creating conda environment from environment.yml..."
cd "$PROJECT_DIR"
conda env create -f environment.yml

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  conda activate cricketmodel"
echo ""
echo "To verify installation:"
echo "  python scripts/verify_install.py"
echo ""
echo "To train the model:"
echo "  python train.py"
