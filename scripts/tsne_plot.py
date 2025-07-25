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
    import matplotlib.patches as mpatches

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='random', random_state=42)
    x0 = tsne.fit_transform(X)
    x, y = x0[:, 0], x0[:, 1]

    unique_labels = np.unique(labels)
    colors = plt.get_cmap('tab20', len(unique_labels))  # Better for many distinct labels

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    for i, label in enumerate(unique_labels):
        indices = labels == label
        ax.scatter(x[indices], y[indices], s=10, color=colors(i), label=class_names.get(label, f"Class {label}"))

    # Use multiple columns for the legend to save vertical space
    ncol = 3 if len(unique_labels) <= 30 else 4
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),  # Place legend below the plot
        ncol=ncol,
        fontsize='small',
        title="Classes",
        markerscale=1.2,
        frameon=False
    )

    ax.set_title(f"t-SNE visualization")
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend

    # Save
    domain_folder = os.path.join(tsne_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)
    image_path = os.path.join(domain_folder, f"DGCD_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}.png")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

from umap import UMAP  # Make sure to `pip install umap-learn`

def plot_umap(tsne_dir: str, domain_name: str, X: np.ndarray, labels: np.ndarray, episode: int, class_names: dict) -> None:
    # Run UMAP on the data
    umap = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
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
    plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    fontsize='xx-small',
    title="Classes",
    frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(right=0.75)
    plt.title(f"UMAP visualization")

    # Prepare the output directory and file
    domain_folder = os.path.join(tsne_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)
    image_path = os.path.join(domain_folder, f"DGCD_Umap_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}_UMAP.png")
    plt.savefig(image_path, dpi=300)
    plt.close()

from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.metrics import silhouette_score

def lorentz_inner_product(u, v):
    return -u[0]*v[0] + np.dot(u[1:], v[1:])

def hyperbolic_distance(u, v):
    ip = lorentz_inner_product(u, v)
    ip = np.clip(ip, -1e9, -1.0)  # numerical stability
    return np.arccosh(-ip)

def hyperbolic_distance_matrix(X):
    n = X.shape[0]
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            dist = hyperbolic_distance(X[i], X[j])
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    return dist_matrix

def plot_hyperbolic(
    tsne_dir: str,
    domain_name: str,
    X: np.ndarray,
    labels: np.ndarray,
    episode: int,
    class_names: dict
) -> None:
    # 1) fit UMAP in hyperboloid space
    hyper_mapper = UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        output_metric="hyperboloid",
        random_state=42
    )
    coords = hyper_mapper.fit_transform(X)
    x, y = coords[:, 0], coords[:, 1]
    # lift onto the two-sheeted hyperboloid
    z = np.sqrt(1 + x**2 + y**2)
    # 2) compute silhouette score
    # Note: hyperbolic distance matrix is not symmetric, so we use the 'precomputed' metric
    # to compute the silhouette score
    dist_mat = hyperbolic_distance_matrix(coords)
    score = silhouette_score(dist_mat, labels, metric='precomputed')
    print(f"Silhouette Score (Hyperbolic): {score:.4f}")

    
    # --- (a) 3D hyperboloid plot ---
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    for lbl in np.unique(labels):
        idx = labels == lbl
        ax3d.scatter(
            x[idx], y[idx], z[idx],
            label=class_names.get(lbl, f"Class {lbl}"),
            s=8
        )
    ax3d.set_title("Hyperboloid UMAP embedding")
    ax3d.view_init( elev=30, azim=45 )
    ax3d.axis("off")
    
    # add legend (you may need to adjust ncol/fontsize for many classes)
    ax3d.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # push it down
        ncol=1,
        fontsize='small',
        title="Classes",
        frameon=False
    )
    # --- (b) Poincaré disk projection ---
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for lbl in np.unique(labels):
        idx = labels == lbl
        ax2.scatter(
            disk_x[idx], disk_y[idx],
            label=class_names.get(lbl, f"Class {lbl}"),
            s=8
        )
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor="black")
    ax2.add_artist(circle)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Poincaré disk projection")
    ax2.text(
    0, -1.1, f"Silhouette Score: {score:.4f}",
    ha='center', fontsize=10, transform=ax2.transAxes
    )
    # save both
    out_dir = os.path.join(tsne_dir, domain_name, "hyperbolic")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(
            out_dir,
            f"DGCD_hyperboloid_{os.path.basename(tsne_dir)}_{domain_name}_ckpt{episode}.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    fig2.savefig(
        os.path.join(
            out_dir,
            f"DGCD_poincare_{os.path.basename(tsne_dir)}_{domain_name}_ckpt{episode}.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig); plt.close(fig2)



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
        "Art": f"HypDGCD/checkpoints/{args.dataset_name}/Art/Intermediate_Art_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "Clipart": f"HypDGCD/checkpoints/{args.dataset_name}/Clipart/Intermediate_Clipart_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "Product": f"HypDGCD/checkpoints/{args.dataset_name}/Product/Intermediate_Product_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "Real_world": f"HypDGCD/checkpoints/{args.dataset_name}/Real_world/Intermediate_Real_world_hyper_bus_manual_trained_model_{args.checkpoint}.pkl"
    },
    'PACS': {
        "art_painting": f"HypDGCD/checkpoints/{args.dataset_name}/art_painting/Intermediate_art_painting_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "photo": f"HypDGCD/checkpoints/{args.dataset_name}/photo/Intermediate_photo_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "cartoon": f"HypDGCD/checkpoints/{args.dataset_name}/cartoon/Intermediate_cartoon_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
        "sketch": f"HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_hyper_bus_manual_trained_model_{args.checkpoint}.pkl"
    },
    'Domain_Net': {
        "clipart": f"HypDGCD/checkpoints/{args.dataset_name}/clipart/Intermediate_clipart_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "sketch": f"HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
        "painting": f"HypDGCD/checkpoints/{args.dataset_name}/painting/Intermediate_painting_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
    }
}

# proj_source_domains_mapping = {
#     'Office_Home': {
#         "Art": f"HypDGCD/checkpoints/{args.dataset_name}/Art/Intermediate_Art_hyper_bus_manual_trained_model_c0.03_{args.checkpoint}.pkl",
#         "Clipart": f"HypDGCD/checkpoints/{args.dataset_name}/Clipart/Intermediate_Clipart_hyper_bus_manual_trained_model_c0.03_{args.checkpoint}.pkl",
#         "Product": f"HypDGCD/checkpoints/{args.dataset_name}/Product/Intermediate_Product_hyper_bus_manual_trained_model_c0.03_{args.checkpoint}.pkl",
#         "Real_world": f"HypDGCD/checkpoints/{args.dataset_name}/Real_world/Intermediate_Real_world_hyper_bus_manual_trained_model_c0.03_{args.checkpoint}.pkl"
#     },
#     'PACS': {
#         "art_painting": f"HypDGCD/checkpoints/{args.dataset_name}/art_painting/Projection_Head_art_painting_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
#         "photo": f"HypDGCD/checkpoints/{args.dataset_name}/photo/Projection_Head_photo_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
#         "cartoon": f"HypDGCD/checkpoints/{args.dataset_name}/cartoon/Projection_Head_cartoon_hyper_bus_manual_trained_model_{args.checkpoint}.pkl",
#         "sketch": f"HypDGCD/checkpoints/{args.dataset_name}/sketch/Projection_Head_sketch_hyper_bus_manual_trained_model_{args.checkpoint}.pkl"
#     },
#     'Domain_Net': {
#         "clipart": f"HypDGCD/checkpoints/{args.dataset_name}/clipart/Intermediate_clipart_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
#         "sketch": f"HypDGCD/checkpoints/{args.dataset_name}/sketch/Intermediate_sketch_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
#         "painting": f"HypDGCD/checkpoints/{args.dataset_name}/painting/Intermediate_painting_avg_hyper_bus_manual_trained_model_{args.checkpoint}epoch.pkl",
#     }
# }

# Set source domains based on the dataset_name
if args.dataset_name in source_domains_mapping:
    source_domains = source_domains_mapping[args.dataset_name]
    # proj_domains= proj_source_domains_mapping[args.dataset_name]
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


selected_classes=create_list(source_domain=os.path.join(f"{args.dataset_name}", all_domains[0]), num_classes=num_classes_mapping[args.dataset_name])

for train_domain, model_file in source_domains.items():
    adapted_model = load_model(model_file)
    adapted_model=adapted_model.to(device)
    # projection_head = load_model(proj_domains[train_domain])
    print(f"Model loaded successfully for {train_domain}")
    for domain in all_domains:
        if domain!=train_domain:
            folder_path = f"HypDGCD/Episode_all_{args.dataset_name}/{domain}"
            target_Dataloader = create_ViT_test_dataloaders(
                target_domain=os.path.join(f"{args.dataset_name}", domain),
                csv_dir_path=folder_path,
                batch_size=128,
                transform=train_transform,
                selected_classes=selected_classes,
                split = num_classes_mapping[args.dataset_name]
            )
            with torch.no_grad():
                print(f'Testing on {domain}_Domain dataset...')
                adapted_model.eval()
                # projection_head.eval()
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
                    # feats=projection_head(feats)
                    feats = torch.nn.functional.normalize(feats, dim=-1)

                    all_feats.append(feats.cpu().numpy())
                    targets = np.append(targets, label.cpu().numpy())
                    # mask = np.append(mask, np.array([True if x.item() in range(num_classes_mapping[args.dataset_name])
                    #                                 else False for x in label]))

                all_feats = np.concatenate(all_feats)
                targets = targets.astype(int)
                all_classes = np.unique(targets)
                selected_classes = np.random.choice(all_classes, size=7, replace=False)  # pick 10 unique classes

                sampled_feats = []
                sampled_labels = []

                for cls in selected_classes:
                    cls_indices = np.where(targets == cls)[0]
                    if len(cls_indices) >= 40:
                        sampled_idx = np.random.choice(cls_indices, size=40, replace=False)
                    else:
                        sampled_idx = np.random.choice(cls_indices, size=40, replace=True)
                    sampled_feats.append(all_feats[sampled_idx])
                    sampled_labels.extend([cls] * 40)

                X_sampled = np.concatenate(sampled_feats)
                labels_sampled = np.array(sampled_labels)
                # plot_tsne(
                #     tsne_dir=f"HypDGCD/tsne_dir/{args.dataset_name}/{train_domain}",
                #     X=X_sampled,
                #     labels=labels_sampled,
                #     domain_name=domain,
                #     episode=checkpoint,
                #     class_names=class_names,
                # )
                # new hyperbolic UMAP call
                plot_hyperbolic(
                    tsne_dir=f"HypDGCD/tsne_dir/{args.dataset_name}/{train_domain}",
                    domain_name=domain,
                    X=X_sampled,
                    labels=labels_sampled,
                    episode=checkpoint,
                    class_names=class_names
                )
