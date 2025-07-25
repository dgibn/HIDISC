import torch
import os,sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import torchvision
from project_utils.my_utils import *
# from project_utils.loss import *
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import cv2
from data.augmentations import get_transform
from project_utils.data_setup import *
import warnings
import math
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str)
parser.add_argument('--dataset_name',required=True,type=str)
args = parser.parse_args()
print(args)
import random
import numpy as np

def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Setting this to False may slow down training but ensures deterministic results

# Set the seed
set_seed(20)  # You can change the seed value as needed

# Define number of classes for each dataset
num_classes_mapping = {
    'Office_Home': 40,
    'PACS': 4,
    'Domain_Net': 250
}
total_classes_mapping = {
    'Office_Home': 65,
    'PACS': 7,
    'Domain_Net': 345
}

# Domains and their specific model checkpoints
checkpoint=args.checkpoint
c=0.05
radius=2.3

source_domains_mapping = {
    'Office_Home': {
        "Art": f"checkpoints/{args.dataset_name}/Art/Intermediate_Art_hyper_all_in_trained_model_alphadmax0.75_{args.checkpoint}.pkl",
        "Clipart": f"checkpoints/{args.dataset_name}/Clipart/Intermediate_Clipart_hyper_all_in_trained_model_alphadmax0.75_{args.checkpoint}.pkl",
        "Product": f"checkpoints/{args.dataset_name}/Product/Intermediate_Product_hyper_all_in_trained_model_alphadmax0.75_{args.checkpoint}.pkl",
        "Real_world": f"checkpoints/{args.dataset_name}/Real_world/Intermediate_Real_world_hyper_all_in_trained_model_alphadmax0.75_{args.checkpoint}.pkl"
    },
    'PACS': {
        "art_painting": f"checkpoints/{args.dataset_name}/art_painting/Intermediate_art_painting_hyper_all_in_trained_model_{args.checkpoint}.pkl",
        "photo": f"checkpoints/{args.dataset_name}/photo/Intermediate_photo_hyper_all_in_trained_model_{args.checkpoint}.pkl",
        "cartoon": f"checkpoints/{args.dataset_name}/cartoon/Intermediate_cartoon_hyper_all_in_trained_model_{args.checkpoint}.pkl",
        "sketch": f"checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_hyper_all_in_trained_model_{args.checkpoint}.pkl"
    },
    'Domain_Net': {
        "clipart": f"checkpoints/{args.dataset_name}/clipart/Intermediate_clipart_hyper_all_in_trained_model_{args.checkpoint}.pkl",
        "sketch": f"checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_hyper_all_in_trained_model_{args.checkpoint}.pkl",
        "painting": f"checkpoints/{args.dataset_name}/painting/Intermediate_painting_hyper_all_in_trained_model_{args.checkpoint}.pkl",
    }
}

# Set source domains based on the dataset_name
if args.dataset_name in source_domains_mapping:
    source_domains = source_domains_mapping[args.dataset_name]
else:
    raise ValueError(f"Dataset '{args.dataset_name}' is not supported.")

device = "cuda:5" if torch.cuda.is_available() else "cpu"
# Load the model
global_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
# global_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',pretrained=True)
global_model.head = nn.Identity()
print("Model loaded successfully")
# Initialize transformation pipeline
train_transform, test_transform = get_transform(transform_type="imagenet", image_size=224)
train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)
results=[]
BATCH_SIZE = 128
# List of all domains for each dataset
all_domains_mapping = {
    'Office_Home': ["Art", "Clipart", "Product", "Real_world"],
    'PACS': ["art_painting", "photo", "cartoon", "sketch"],
    'Domain_Net': ["clipart", "infograph", "quickdraw", "real","painting","sketch"]
}

# Set all_domains based on dataset_name
if args.dataset_name in all_domains_mapping:
    all_domains = all_domains_mapping[args.dataset_name]
else:
    raise ValueError(f"Dataset '{args.dataset_name}' is not supported.")

selected_classes=create_list(source_domain=os.path.join(f"{args.dataset_name}", all_domains[0]), num_classes=num_classes_mapping[args.dataset_name])
print("Length of Selected Classes=",len(selected_classes))
print("Length of not selected Classes=",total_classes_mapping[args.dataset_name]-num_classes_mapping[args.dataset_name])

for train_domain, model_file in source_domains.items():
    adapted_model = load_model(model_file)
    adapted_model=adapted_model.to(device)
    print(f"Model loaded successfully for {train_domain}")
    for domain in all_domains:
        if domain!=train_domain:
            folder_path = f"Episode_all_{args.dataset_name}/{domain}"
            target_Dataloader = create_ViT_test_dataloaders(
                target_domain=os.path.join(f"{args.dataset_name}", domain),
                csv_dir_path=folder_path,
                batch_size=BATCH_SIZE,
                transform=train_transform,
                selected_classes=selected_classes,
                split = num_classes_mapping[args.dataset_name]
            )
            try:
                with torch.no_grad():
                    print(f'Testing on {domain}_Domain dataset...')
                    all_acc_test, old_acc_test, new_acc_test = test_kmeans_cdad(
                        device=device,
                        model=adapted_model,
                        test_loader=target_Dataloader,
                        epoch=1,
                        save_name='Test ACC',
                        num_train_classes=num_classes_mapping[args.dataset_name],
                        num_unlabelled_classes=total_classes_mapping[args.dataset_name]-num_classes_mapping[args.dataset_name]
                    )
                print(f'{domain}_Test Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f}  | New {new_acc_test:.4f}')
                results.append({
                    "Checkpoint":checkpoint,
                    "Source Domain": train_domain,
                    "Target Domain": domain,
                    "All": all_acc_test*100,
                    "Old": old_acc_test*100,
                    "New": new_acc_test*100
                })
               
            except Exception as e:
                print(f"Failed to test on {domain}: {e}")

# Define the path where the results will be saved
save_path = f"Results/{args.dataset_name}/DGCD_hyper_all_in_alphadmax0.75.xlsx"

# Create the directory if it does not exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Create a DataFrame from the results
new_results_df = pd.DataFrame(results)

# Check if the file already exists
if os.path.exists(save_path):
    # Read the existing data
    existing_df = pd.read_excel(save_path)
    # Append new results
    updated_df = pd.concat([existing_df, new_results_df], ignore_index=True)
else:
    updated_df = new_results_df

# Write the combined DataFrame to Excel, without the index
updated_df.to_excel(save_path, index=False)
