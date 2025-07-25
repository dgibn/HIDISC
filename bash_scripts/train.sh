#!/bin/bash

# Script to run the episodic-training process for Domain Generalization
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Activate the Python environment
source "path_to_your_conda/miniconda3/etc/profile.d/conda.sh"
conda activate dgcd

# Navigate to the script's directory
cd "path_to_your_scripts_directory"

# List of domains in the Office Home dataset
# declare -a domains=("Art" "Real_world" "Clipart" "Product")
# List of domains in the Office_Home dataset
# declare -a domains=("art_painting" "sketch") # "art_painting" "sketch") # "cartoon") # "photo" "sketch" "cartoon") # "sketch") # "art_painting" "sketch" "cartoon" "photo")
# List of domains in the Domain Net dataset
declare -a domains=("sketch" "clipart" "painting") # "clipart") # "sketch") # "sketch" "painting" "clipart")
# Loop through each domain and execute the training script
for domain in "${domains[@]}"
do
    echo "Starting training for the domain: $domain"
    # Run the Python script with parameters for the current domain
    python all_in.py \
        --task_epochs 51 \
        --task_lr 0.01 \
        --batch_size 128 \
        --n_views 2 \
        --image_size 224 \
        --dataset_name "Domain_Net" \
        --source_domain_name "$domain" \
        --transform "imagenet" \
        --c 0.04 \
        --do_hyperbolic "True" \
        --prototype_dim 32 \
        --penalty 0.75 \
        --device_id 1 > /DGCD_Hyp_Bus_all_in_${domain}_log_0.01.out 2>&1 &
    echo "Training completed for the domain: $domain"
done

echo "All domain trainings are completed."
