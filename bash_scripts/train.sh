#!/bin/bash
# -------------------------------------------------------------------
# HIDISC Training Launcher (train.sh)
# -------------------------------------------------------------------
# This script automates the training process for the HIDISC framework
# across multiple source domains for a specified dataset (e.g., Office-Home).
#
# It iterates through each domain in the dataset, treating it as the
# source domain for Domain Generalization (DG) training, while the
# other domains serve as unseen targets (handled internally by the
# data loader in train.py).
#
# Usage:
#   bash bash_scripts/train.sh
#
# Note: Ensure the conda environment is active before running.
# -------------------------------------------------------------------

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Activate the Python environment
# source "path_to_your_conda/miniconda3/etc/profile.d/conda.sh"
# conda activate dgcd

# Navigate to the script's directory
mkdir -p ../output

# -------------------------------------------------------------------
# Domain Configuration
# -------------------------------------------------------------------
# Uncomment the appropriate list for the dataset you wish to train on.
# The script will loop through these domains one by one.

# List of domains in the Office Home dataset
declare -a domains=("Real_world" "Clipart" "Product") # "Real_world" "Clipart" "Product")
# List of domains in the PACS dataset
# declare -a domains=("art_painting" "sketch" "cartoon" "photo")
# List of domains in the Domain Net dataset
# declare -a domains=("sketch" "clipart" "painting") # "clipart") # "sketch") # "sketch" "painting" "clipart")

# -------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------
# Loop through each domain and execute the training script
for domain in "${domains[@]}"
do
    echo "Starting training for the domain: $domain"
    
    # Run the Python training script (train.py) with HIDISC hyperparameters.
    # Key arguments:
    #   --dataset_name:       Target dataset (e.g., Office_Home, PACS)
    #   --source_domain_name: Current source domain from the loop
    #   --do_hyperbolic:      Enable hyperbolic geometry (True for HIDISC)
    #   --c:                  Initial curvature value (learnable)
    #   --penalty:            Penalty weight for Busemann loss
    #
    # Output is redirected to a log file in the output/ directory.
    python scripts/train.py \
        --task_epochs 50 \
        --task_lr 0.01 \
        --batch_size 128 \
        --n_views 2 \
        --image_size 224 \
        --dataset_name "Office_Home" \
        --source_domain_name "$domain" \
        --transform "imagenet" \
        --c 0.04 \
        --do_hyperbolic "True" \
        --prototype_dim 32 \
        --penalty 0.75 \
        --device_id 0 > /users/student/pg/pg23/vaibhav.rathore/HIDISC/output/HIDISC_${domain}_log.out 2>&1
echo "Training completed for the domain: $domain"
done

echo "All domain trainings are completed."
