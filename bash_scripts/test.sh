#!/bin/bash
# Activate the Python environment
source "path/to/your/miniconda3/etc/profile.d/conda.sh"  # Update this path to your conda installation
conda activate dgcd
# Define the paths for the script and logs
SCRIPT_PATH="path to your scripts folder"  # Update this with the actual path to your Python script
DATASET_NAME="Office_Home"  # Update with the desired dataset name (e.g., OfficeHome, PACS, Domain_Net)

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Loop through checkpoints at intervals of 1 from 0 to 9
for checkpoint in {0..51..5}; do
    # Print a message indicating the checkpoint being processed
    echo "Processing checkpoint $checkpoint"

    # Run the Python script with the current checkpoint and dataset name, save the output to a log file
    python3 $SCRIPT_PATH \
        --checkpoint $checkpoint \
        --dataset_name $DATASET_NAME

    # Print a message indicating completion of the current checkpoint
    echo "Testing completed for checkpoint $checkpoint."
done

# Deactivate environment if it was activated
# deactivate