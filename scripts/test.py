import argparse
import math
import os
import random
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add parent directory to path to allow imports from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.augmentations import get_transform
from project_utils.data_setup import ContrastiveLearningViewGenerator, create_test_dataloaders
from project_utils.my_utils import create_list, load_model, set_seed, test_kmeans_cdad

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_environment(seed: int = 20) -> str:
    """
    Sets the random seed and determines the device to use.
    
    Args:
        seed (int): Random seed.
        
    Returns:
        str: Device string ('cuda:0' or 'cpu').
    """
    set_seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def get_dataset_config(dataset_name: str, checkpoint: str) -> Tuple[Dict, Dict, Dict, List]:
    """
    Returns configuration mappings for the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        checkpoint (str): Checkpoint identifier.
        
    Returns:
        Tuple containing:
            - num_classes_mapping (Dict): Number of classes per dataset.
            - total_classes_mapping (Dict): Total classes per dataset.
            - source_domains (Dict): Mapping of domains to checkpoint paths.
            - all_domains (List): List of all domains for the dataset.
    """
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
    
    all_domains_mapping = {
        'Office_Home': ["Art", "Clipart", "Product", "Real_world"],
        'PACS': ["art_painting", "photo", "cartoon", "sketch"],
        'Domain_Net': ["clipart", "infograph", "quickdraw", "real", "painting", "sketch"]
    }

    if dataset_name not in all_domains_mapping:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    all_domains = all_domains_mapping[dataset_name]

    # Construct source domains mapping dynamically based on dataset and checkpoint
    source_domains = {}
    if dataset_name == 'Office_Home':
        source_domains = {
            "Art": f"checkpoints/{dataset_name}/Art/Intermediate_Art__trained_model_{checkpoint}.pkl",
            "Clipart": f"checkpoints/{dataset_name}/Clipart/Intermediate_Clipart__trained_model_{checkpoint}.pkl",
            "Product": f"checkpoints/{dataset_name}/Product/Intermediate_Product__trained_model_{checkpoint}.pkl",
            "Real_world": f"checkpoints/{dataset_name}/Real_world/Intermediate_Real_world__trained_model_{checkpoint}.pkl"
        }
    elif dataset_name == 'PACS':
        source_domains = {
            "art_painting": f"checkpoints/{dataset_name}/art_painting/Intermediate_art_painting__trained_model_{checkpoint}.pkl",
            "photo": f"checkpoints/{dataset_name}/photo/Intermediate_photo__trained_model_{checkpoint}.pkl",
            "cartoon": f"checkpoints/{dataset_name}/cartoon/Intermediate_cartoon__trained_model_{checkpoint}.pkl",
            "sketch": f"checkpoints/{dataset_name}/sketch/Intermediate_sketch__trained_model_{checkpoint}.pkl"
        }
    elif dataset_name == 'Domain_Net':
        source_domains = {
            "clipart": f"checkpoints/{dataset_name}/clipart/Intermediate_clipart__trained_model_{checkpoint}.pkl",
            "sketch": f"checkpoints/{dataset_name}/sketch/Intermediate_sketch__trained_model_{checkpoint}.pkl",
            "painting": f"checkpoints/{dataset_name}/painting/Intermediate_painting__trained_model_{checkpoint}.pkl",
        }

    return num_classes_mapping, total_classes_mapping, source_domains, all_domains


def run_testing(args, device, train_transform, num_classes_mapping, total_classes_mapping, source_domains, all_domains):
    """
    Runs the testing loop across domains.
    
    Args:
        args: Parsed arguments.
        device: Device to run on.
        train_transform: Transformations.
        num_classes_mapping: Mapping of number of classes.
        total_classes_mapping: Mapping of total classes.
        source_domains: Source domains and checkpoints.
        all_domains: List of all domains.
        
    Returns:
        List[Dict]: Results of the testing.
    """
    results = []
    BATCH_SIZE = 128
    
    # Determine selected classes based on the first domain
    first_domain_path = os.path.join(f"{args.dataset_name}", all_domains[0])
    selected_classes = create_list(source_domain=first_domain_path, num_classes=num_classes_mapping[args.dataset_name])
    
    print(f"Length of Selected Classes: {len(selected_classes)}")
    print(f"Length of not selected Classes: {total_classes_mapping[args.dataset_name] - num_classes_mapping[args.dataset_name]}")

    for train_domain, model_file in source_domains.items():
        # Check if model file exists
        if not os.path.exists(model_file):
             # Try checking relative to current working directory if the path in script was relative
             # The script uses f"checkpoints/..." which is relative.
             # If the script is run from root, it should be fine.
             pass

        try:
            print(f"Loading model for {train_domain} from {model_file}")
            adapted_model = load_model(model_file)
            adapted_model = adapted_model.to(device)
            print(f"Model loaded successfully for {train_domain}")
        except Exception as e:
            print(f"Failed to load model for {train_domain}: {e}")
            continue
        
        for domain in all_domains:
            if domain != train_domain:
                folder_path = f"HIDISC/Episode_all_{args.dataset_name}/{domain}"
                target_domain_path = os.path.join(f"{args.dataset_name}", domain)
                
                target_dataloader = create_test_dataloaders(
                    target_domain=target_domain_path,
                    csv_dir_path=folder_path,
                    batch_size=BATCH_SIZE,
                    transform=train_transform,
                    selected_classes=selected_classes,
                    split=num_classes_mapping[args.dataset_name]
                )
                
                try:
                    with torch.no_grad():
                        print(f'Testing on {domain}_Domain dataset...')
                        all_acc_test, old_acc_test, new_acc_test = test_kmeans_cdad(
                            device=device,
                            model=adapted_model,
                            test_loader=target_dataloader,
                            epoch=1,
                            save_name='Test ACC',
                            num_train_classes=num_classes_mapping[args.dataset_name],
                            num_unlabelled_classes=total_classes_mapping[args.dataset_name] - num_classes_mapping[args.dataset_name]
                        )
                    
                    print(f'{domain}_Test Accuracies: All {all_acc_test:.4f} | Old {old_acc_test:.4f} | New {new_acc_test:.4f}')
                    results.append({
                        "Checkpoint": args.checkpoint,
                        "Source Domain": train_domain,
                        "Target Domain": domain,
                        "All": all_acc_test * 100,
                        "Old": old_acc_test * 100,
                        "New": new_acc_test * 100
                    })
                
                except Exception as e:
                    print(f"Failed to test on {domain}: {e}")
                    # import traceback
                    # traceback.print_exc()

    return results


def save_results(results: List[Dict], dataset_name: str):
    """
    Saves the results to an Excel file.
    
    Args:
        results (List[Dict]): List of result dictionaries.
        dataset_name (str): Name of the dataset.
    """
    if not results:
        print("No results to save.")
        return

    save_path = f"HIDISC/Results/{dataset_name}/HIDISC_Results.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    new_results_df = pd.DataFrame(results)

    if os.path.exists(save_path):
        existing_df = pd.read_excel(save_path)
        updated_df = pd.concat([existing_df, new_results_df], ignore_index=True)
    else:
        updated_df = new_results_df

    updated_df.to_excel(save_path, index=False)
    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test HIDISC model on various domains.")
    parser.add_argument('--checkpoint', required=True, type=str, help="Checkpoint identifier")
    parser.add_argument('--dataset_name', required=True, type=str, help="Name of the dataset (e.g., Office_Home, PACS)")
    args = parser.parse_args()

    device = setup_environment(seed=20)
    
    try:
        num_classes_mapping, total_classes_mapping, source_domains, all_domains = get_dataset_config(args.dataset_name, args.checkpoint)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Initialize transformation pipeline
    # Note: The original script loads a global model here but doesn't use it. It has been removed in this refactor.
    
    train_transform, _ = get_transform(transform_type="imagenet", image_size=224)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)

    results = run_testing(args, device, train_transform, num_classes_mapping, total_classes_mapping, source_domains, all_domains)
    
    save_results(results, args.dataset_name)


if __name__ == "__main__":
    main()
