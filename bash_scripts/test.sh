#!/bin/bash

# =============================================================================
# Script Name: test.sh
# Description: Runs the HIDISC testing script across a range of checkpoints.
# Usage: ./bash_scripts/test.sh [DATASET_NAME] [START_CHECKPOINT] [END_CHECKPOINT] [STEP]
# Example: ./bash_scripts/test.sh Office_Home 0 50 5
# =============================================================================

# --- Configuration ---

# Default values
DEFAULT_DATASET_NAME="Office_Home"
DEFAULT_START=0
DEFAULT_END=51
DEFAULT_STEP=5

# Get arguments or use defaults
DATASET_NAME="${1:-$DEFAULT_DATASET_NAME}"
START="${2:-$DEFAULT_START}"
END="${3:-$DEFAULT_END}"
STEP="${4:-$DEFAULT_STEP}"

# Determine the project root directory (assuming script is in bash_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/test.py"

# --- Environment Setup ---

# Export thread limits to avoid conflicts/overhead
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# --- Validation ---

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

echo "========================================================"
echo "Starting Testing Pipeline"
echo "Dataset:    $DATASET_NAME"
echo "Script:     $PYTHON_SCRIPT"
echo "Checkpoints: $START to $END (step $STEP)"
echo "========================================================"

# --- Main Loop ---

for (( checkpoint=START; checkpoint<=END; checkpoint+=STEP )); do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing checkpoint $checkpoint..."

    # Run the Python script
    if python3 "$PYTHON_SCRIPT" \
        --checkpoint "$checkpoint" \
        --dataset_name "$DATASET_NAME"; then
        
        echo "Testing completed successfully for checkpoint $checkpoint."
    else
        echo "Error occurred while testing checkpoint $checkpoint."
        # Optional: exit 1 # Uncomment to stop on first failure
    fi
    
    echo "--------------------------------------------------------"
done

echo "All requested checkpoints processed."