import torch
import os,sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pandas as pd
from project_utils.my_utils import *
from project_utils.loss import *
from data.augmentations import get_transform
from project_utils.data_setup import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse

from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tsne(tsne_dir: str, domain_name: str, X: np.ndarray, labels: np.ndarray, episode: int, class_names: dict) -> None:
    # Run t-SNE on the data
    tsne = TSNE(n_components=2, perplexity=30,learning_rate='auto', init='random', random_state=42)
    x0 = tsne.fit_transform(X)

    # Extract the two components
    x = x0[:, 0]
    y = x0[:, 1]

    # Create a color map based on the unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    # colors = plt.cm.plasma(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(8, 6))  # Larger figure size for clarity
    # Scatter plot using the colors corresponding to labels
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(x[indices], y[indices], c=colors[i].reshape(1, -1), label=class_names.get(label, f"Class {label}"),s=20)

    # Add legend and title
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

    # Adjust the layout to make space for the legend
    plt.subplots_adjust(right=0.75)  # Adjust the right space to make room for the legend
    plt.title(f"t-SNE visualization")

    # Prepare the output directory and file
    domain_folder = os.path.join(tsne_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)
    image_path = os.path.join(domain_folder, f"DGCD_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}.png")
    plt.savefig(image_path, dpi=300)
    plt.close()


from umap import UMAP  # Make sure to `pip install umap-learn`

def plot_umap(tsne_dir: str, domain_name: str, X: np.ndarray, labels: np.ndarray, episode: int, class_names: dict) -> None:
    # Run UMAP on the data
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    x0 = umap.fit_transform(X)

    # Extract the two components
    x = x0[:, 0]
    y = x0[:, 1]

    # Create a color map based on the unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 6))  # Larger figure size for clarity
    # Scatter plot using the colors corresponding to labels
    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(x[indices], y[indices], c=colors[i].reshape(1, -1),
                    label=class_names.get(label, f"Class {label}"), s=20)

    # Add legend and title
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")
    plt.subplots_adjust(right=0.75)
    plt.title(f"UMAP visualization")

    # Prepare the output directory and file
    domain_folder = os.path.join(tsne_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)
    image_path = os.path.join(domain_folder, f"DGCD_Umap_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}_UMAP.png")
    plt.savefig(image_path, dpi=300)
    plt.close()


def plot_umap_poincare(tsne_dir: str, domain_name: str, X: np.ndarray, labels: np.ndarray, episode: int, class_names: dict) -> None:
    from umap import UMAP
    import matplotlib.patches as patches

    # Run UMAP
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    x0 = umap.fit_transform(X)

    # Normalize points inside the Poincaré disk
    norm = np.linalg.norm(x0, axis=1, keepdims=True)
    epsilon = 1e-6
    normalized = x0 / (np.maximum(norm, 1.0 + epsilon))  # Project any out-of-bound points to boundary

    x = normalized[:, 0]
    y = normalized[:, 1]

    # Prepare plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Draw Poincaré disk (unit circle)
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='none', linewidth=1)
    ax.add_patch(circle)

    # Plot colored embeddings
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        ax.scatter(x[indices], y[indices], c=colors[i].reshape(1, -1), label=class_names.get(label, f"Class {label}"), s=15)

    # Format plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.axis('off')
    plt.title("UMAP Projection in Poincaré Ball")

    # Save
    domain_folder = os.path.join(tsne_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)
    image_path = os.path.join(domain_folder, f"DGCD_UMAP_Poincare_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=str)
parser.add_argument('--dataset_name',required=True,type=str)
args = parser.parse_args()
print(args)

# Define number of classes for each dataset
num_classes_mapping = {
    'Office_Home': 40,
    'PACS': 4,
    'Domain_Net': 250
}
# Domains and their specific model checkpoints
checkpoint=args.checkpoint
c=0.03
# Define source domains for each dataset
# Define source domains for each dataset
source_domains_mapping = {
    'Office_Home': {
        "Art": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Art/Intermediate_Art_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Clipart": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Clipart/Intermediate_Clipart_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Product": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Product/Intermediate_Product_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Real_world": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Real_world/Intermediate_Real_world_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl"
    },
    'PACS': {
        "art_painting": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/art_painting/Intermediate_art_painting_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "photo": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/photo/Intermediate_photo_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "cartoon": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/cartoon/Intermediate_cartoon_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "sketch": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_hyper_bus_manual_trained_model_{args.checkpoint}.pkl"
    },
    'Domain_Net': {
        "clipart": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/clipart/Intermediate_clipart_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "sketch": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "painting": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/painting/Intermediate_painting_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
    }
}

proj_source_domains_mapping = {
    'Office_Home': {
        "Art": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Art/Intermediate_Art_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Clipart": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Clipart/Intermediate_Clipart_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Product": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Product/Intermediate_Product_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl",
        "Real_world": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/Real_world/Intermediate_Real_world_hyper_dbus_cut_trained_model_c0.03_{args.checkpoint}.pkl"
    },
    'PACS': {
        "art_painting": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/art_painting/Projection_Head_art_painting_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "photo": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/photo/Projection_Head_photo_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "cartoon": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/cartoon/Projection_Head_cartoon_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "sketch": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/sketch/Projection_Head_sketch_hyper_bus_manual_trained_model_{args.checkpoint}.pkl"
    },
    'Domain_Net': {
        "clipart": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/clipart/Intermediate_clipart_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "sketch": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "painting": f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/checkpoints/{args.dataset_name}/painting/Intermediate_painting_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
    }
}

# Set source domains based on the dataset_name
if args.dataset_name in source_domains_mapping:
    source_domains = source_domains_mapping[args.dataset_name]
    proj_domains= proj_source_domains_mapping[args.dataset_name]
else:
    raise ValueError(f"Dataset '{args.dataset_name}' is not supported.")

device = "cuda:4" if torch.cuda.is_available() else "cpu"
# Load the model
global_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
global_model.head = nn.Identity()
print("Model loaded successfully")
# Define the Projection Head for the INFO-NCE loss
projection_head = DINOHead(in_dim=768, out_dim=32, nlayers=3)
projection_head = projection_head.to(device)
# Initialize transformation pipeline
train_transform, test_transform = get_transform(transform_type="imagenet", image_size=224)
train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=2)
results=[]
# List of all domains for each dataset
all_domains_mapping = {
    'Office_Home': ["Art", "Clipart", "Product", "Real_world"],
    'PACS': ["photo","art_painting", "cartoon", "sketch"],
    'Domain_Net': ["clipart", "infograph", "quickdraw", "real","painting","sketch"]
}
PACS_class_names = {
    0: "Giraffe",
    1: "House",
    2: "Guitar",
    3: "Horse",
    4: "Person",
    5: "Elephant",
    6: "Dog"
}

OfficeHome_class_names = {
    0: "Alarm Clock", 1: "Backpack", 2: "Batteries", 3: "Bed", 4: "Bike",
    5: "Bottle", 6: "Bucket", 7: "Calculator", 8: "Calendar", 9: "Candles",
    10: "Chair", 11: "Clipboards", 12: "Computer", 13: "Couch", 14: "Curtains",
    15: "Desk Lamp", 16: "Drill", 17: "Eraser", 18: "Exit Sign", 19: "Fan",
    20: "File Cabinet", 21: "Flipflops", 22: "Flowers", 23: "Folder", 24: "Fork",
    25: "Glasses", 26: "Hammer", 27: "Helmet", 28: "Kettle", 29: "Keyboard",
    30: "Knives", 31: "Lamp", 32: "Laptop", 33: "Marker", 34: "Monitor",
    35: "Mop", 36: "Mouse", 37: "Mug", 38: "Notebook", 39: "Oven"
}

# Set all_domains based on dataset_name
if args.dataset_name in all_domains_mapping:
    all_domains = all_domains_mapping[args.dataset_name]
else:
    raise ValueError(f"Dataset '{args.dataset_name}' is not supported.")

if args.dataset_name == "PACS":
    class_names = PACS_class_names
elif args.dataset_name == "Office_Home":
    class_names = OfficeHome_class_names
else:
    class_names = {i: f"Class {i}" for i in range(num_classes_mapping[args.dataset_name])}


selected_classes=create_list(source_domain=os.path.join(f"/users/student/Datasets/vaibhav/{args.dataset_name}", all_domains[0]), num_classes=num_classes_mapping[args.dataset_name])

for train_domain, model_file in source_domains.items():
    adapted_model = load_model(model_file)
    adapted_model=adapted_model.to(device)
    projection_head = load_model(proj_domains[train_domain])
    print(f"Model loaded successfully for {train_domain}")
    for domain in all_domains:
        if domain!=train_domain:
            folder_path = f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/Episode_all_{args.dataset_name}/{domain}"
            target_Dataloader = create_ViT_test_dataloaders(
                target_domain=os.path.join(f"/users/student/Datasets/vaibhav/{args.dataset_name}", domain),
                csv_dir_path=folder_path,
                batch_size=128,
                transform=train_transform,
                selected_classes=selected_classes,
                split = num_classes_mapping[args.dataset_name]
            )
            with torch.no_grad():
                print(f'Testing on {domain}_Domain dataset...')
                adapted_model.eval()
                projection_head.eval()
                all_feats = []
                targets = np.array([])
                mask = np.array([])

                print('Collating features...')
                # First extract all features
                for batch_idx, data in enumerate(tqdm(target_Dataloader)):
                    images, label = data
                    if isinstance(images, list):
                        images = torch.cat(images, dim=0).to(device)
                        label = torch.cat([label for _ in range(2)]).to(device)
                    else:
                        images = images.to(device)
                        label = label.to(device)
                        
                    feats = adapted_model(images)
                    feats=projection_head(feats)
                    feats = torch.nn.functional.normalize(feats, dim=-1)

                    all_feats.append(feats.cpu().numpy())
                    targets = np.append(targets, label.cpu().numpy())
                    # mask = np.append(mask, np.array([True if x.item() in range(num_classes_mapping[args.dataset_name])
                    #                                 else False for x in label]))

                all_feats = np.concatenate(all_feats)
                plot_umap(
                    tsne_dir=f"/users/student/pg/pg23/vaibhav.rathore/HypDGCD/tsne_dir/{args.dataset_name}/{train_domain}",
                    X=all_feats,
                    labels=targets,
                    domain_name=domain,
                    episode=checkpoint,
                    class_names=class_names,
                )
