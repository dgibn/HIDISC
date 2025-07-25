import os
import sys
import csv
import argparse
import warnings
import random

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# Local project utilities
from project_utils.my_utils import *
from project_utils.loss import *
from project_utils.data_setup import *
# from project_utils.hyperbolic_utils import *
from project_utils.hutils import *
from project_utils.pmath import *
from project_utils.plot_tsne import plot_tsne
# from project_utils.grad_cam import TempClassifier,generate_gradcam_visualization

# --------------------------
# WandB Integration
# --------------------------
import wandb

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Set the seed
set_seed(42)

# -------------------------------------------------------------------
# Argument Parser Setup
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Meta-training for Domain Generalization on Image Data."
)
parser.add_argument('--task_epochs', type=int, default=50,
                    help='Number of task-specific training epochs')
parser.add_argument('--task_lr', type=float, default=0.01,
                    help='Task-specific learning rate.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--n_views', type=int, default=2,
                    help='Number of views for Augmentations.')
parser.add_argument('--image_size', type=int, default=224,
                    help='Image size for transformations.')
parser.add_argument('--dataset_name', type=str, default='PACS',
                    help='Dataset name may be OfficeHome, PACS, Domain_Net.')
parser.add_argument('--source_domain_name', type=str, default='photo',
                    help='Source domain name.')
parser.add_argument('--transform', type=str, default='imagenet',
                    help='Transformation name from augmentations.')
parser.add_argument('--c', type=float, default=0.05,
                    help='Curvature for the hyperbolic space.')
parser.add_argument('--device_id', type=int, default=6,
                    help='CUDA device ID.')
parser.add_argument('--do_hyperbolic', type=str2bool, default=True,
                    help='Use hyperbolic space or not.')
parser.add_argument('--prototype_dim', type=int, default=32,
                    help='Dimension of the learned prototypes (d in the equation).')
parser.add_argument('--penalty', type=float, default=0.75,
                    help='Penalty value for the penalized Busemann loss.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')
args = parser.parse_args()

# --------------------------
# Initialize wandb
# # --------------------------
# wandb.login(key="4883a15d69990032fd28ba66b983caf542ea78f5")
# wandb.init(project="HypDGCD_Project", config=vars(args))
# config = wandb.config

# -------------------------------------------------------------------
# Device and Dataset Setup
# -------------------------------------------------------------------
device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(args.device_id)

num_classes_mapping = {
    'Office_Home': 40,
    'PACS': 4,
    'Domain_Net': 250
}

radius = {
    "PACS": 1,
    "Office_Home": 1.5,
    "Domain_Net": 2.3
}

# Hyper-parameters
NUM_CLASSES = num_classes_mapping[args.dataset_name]
TASK_EPOCHS = args.task_epochs
TASK_LEARNING_RATE = args.task_lr
BATCH_SIZE = args.batch_size
c = args.c
n_views = args.n_views
image_size = args.image_size
transform = args.transform
source_domain_name = args.source_domain_name
feat_dim = 768
emb_dim = 32    # desired embedding dimension from projection head
ALPHA = 0.7
do_hyperbolic = args.do_hyperbolic
prototype_dim = args.prototype_dim
penalty_value = args.penalty  # for penalized Busemann loss
episode=2
# make c a learnable parameter
curvature = nn.Parameter(torch.tensor(args.c, device=device, dtype=torch.float32),
                         requires_grad=True)

# Define paths and directories
source_domain = f"{args.dataset_name}/{source_domain_name}"
task_model_path = f"HypDGCD/checkpoints/{args.dataset_name}/{source_domain_name}/task_model_bus_{source_domain_name}_{args.c}.pkl"
os.makedirs(f"HypDGCD/checkpoints/{args.dataset_name}/{source_domain_name}", exist_ok=True)
tsne_dir_path = f"HypDGCD/tsne_dir/{args.dataset_name}/{source_domain_name}"
os.makedirs(tsne_dir_path, exist_ok=True)
target_data_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"data/dataset_path_{args.dataset_name}.csv")

# Read the target dataset paths from the CSV file
target_dataset_paths = []
with open(target_data_csv, mode='r') as file:
    reader = csv.DictReader(file)
    for i,row in enumerate(reader):
        if i >=2:
            break
        target_dataset_paths.append(row['data_path'])
print(f"Target dataset paths: {target_dataset_paths}")
# -------------------------------------------------------------------
# Transformation and Model Setup
# -------------------------------------------------------------------
train_transform, test_transform = get_transform(transform, image_size=image_size)
train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=n_views)

# Load the VITB16 model pre-trained with DINO and remove the classification head
global_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
# global_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',pretrained=True)
global_model.head = nn.Identity()
global_model = global_model.to(device)

# Freeze all parameters and unfreeze only the last one transformer blocks for fine-tuning
for param in global_model.parameters():
    param.requires_grad = False
for param in global_model.blocks[-1].parameters():
    param.requires_grad = True

# Save the initial task model
# save_model(model=global_model, path=task_model_path)

# Define the Projection Head for the INFO-NCE loss
projection_head = DINOHead(in_dim=feat_dim, out_dim=emb_dim, nlayers=3)
projection_head = projection_head.to(device)

# -------------------------------------------------------------------
# For Domain Net: randomly select a subset of classes
# -------------------------------------------------------------------
selected_classes = create_list(source_domain=source_domain, num_classes=NUM_CLASSES)
class_names = {i: selected_classes[i] for i in range(NUM_CLASSES)}

# Make combined CSV file for all Synthetic Domains
combine_csv_path = f'HypDGCD/Episode_all_{args.dataset_name}/{source_domain_name}'
csv_combined_path = create_combined_csv(combine_csv_path=combine_csv_path,
                                        source_domain=source_domain,
                                        synthetic_domains=target_dataset_paths,
                                        selected_classes=selected_classes,
                                        episode=episode)

combine_dataloader = get_combined_dataloader(csv_path=csv_combined_path,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True,
                                             num_workers=4,
                                             transform=train_transform)

# -------------------------------------------------------------------
# Loss Function Setup
# -------------------------------------------------------------------

busemann_loss = PenalizedBusemannLoss(phi=penalty_value).to(device)

# -------------------------------------------------------------------
# Optimizer Setup for Task Model
# -------------------------------------------------------------------

task_optimizer = torch.optim.SGD([
    {'params': filter(lambda p: p.requires_grad, global_model.parameters()), 'lr': TASK_LEARNING_RATE},
    {'params': projection_head.parameters(), 'lr': TASK_LEARNING_RATE},
    {'params': [curvature], 'lr': TASK_LEARNING_RATE * 1e-3}
], momentum=0.9, weight_decay=5e-5)
task_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(task_optimizer,
                                                            T_max=TASK_EPOCHS,
                                                            eta_min=TASK_LEARNING_RATE * 1e-3)

# -------------------------------------------------------------------
# Prototype Learning Setup
# -------------------------------------------------------------------
num_classes = num_classes_mapping[args.dataset_name]
proto_epochs = 1000
proto_lr = 0.1
proto_momentum = 0.9

# Initialize prototypes and set up their optimizer
prototypes = torch.randn(num_classes, prototype_dim, device=device)
prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
proto_optimizer = torch.optim.SGD([prototypes], lr=proto_lr, momentum=proto_momentum)

for i in range(proto_epochs):
    proto_optimizer.zero_grad()
    loss_proto, max_sim = prototype_loss(prototypes)
    loss_proto.backward()
    proto_optimizer.step()
    with torch.no_grad():
        prototypes.div_(prototypes.norm(dim=1, keepdim=True))


repel_margin = None  # Adaptive outlier margin

# -------------------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------------------
for epoch in tqdm(range(TASK_EPOCHS), desc="Task Training Epochs"):
    global_model.train()
    projection_head.train()
    total_loss = 0
    total_busemann_loss = 0
    total_contrastive_loss = 0
    total_outlier_loss = 0
    tsne_features_hyp = []
    tsne_features_euc = []
    tsne_labels = []
    
    for batch_idx, batch in tqdm(enumerate(combine_dataloader), desc="Processing Batches", leave=False, total=len(combine_dataloader)):
        images, numerical_labels, domain_label = batch
        numerical_labels = numerical_labels.to(device)
        images = torch.cat(images, dim=0).to(device)
        domain_label = domain_label.to(device)
        
        # Feature extraction using global model and projection head
        image_features = global_model(images)
        features = projection_head(image_features)
        features = clip_feature(features, r=radius[args.dataset_name])
        
        # 2) feature‐mix in tangent to simulate novel
        B = features.size(0)
        idx = torch.randperm(B, device=device)
        f1, f2 = features, features[idx]
        lam = torch.from_numpy(
            np.random.beta(1.0, 1.0, size=(B,1))
        ).float().to(device)
        f_mix = lam * f1 + (1 - lam) * f2
        
        # Map features to hyperbolic space if enabled
        if do_hyperbolic:
            features_hyp = expmap0(features, c=curvature)
            z_mix      = expmap0(f_mix, c=curvature)
        else:
            features_hyp = features
            z_mix      = f_mix
        
        # Ensure embeddings lie in the Poincaré ball via tanh
        z = torch.tanh(features_hyp)
        z_mix= torch.tanh(z_mix)
        
        # --- Fix for Dimension Mismatch ---
        # In this version, we replicate prototypes for 2 views by concatenation.
        # Here, since our z is of shape [BATCH_SIZE * n_views, prototype_dim],
        # we assume n_views==2 and build batch_prototypes of shape [BATCH_SIZE * 2, prototype_dim]
        batch_prototypes = torch.cat([prototypes[numerical_labels] for _ in range(2)], dim=0)
        
        # Compute penalized Busemann loss (applied to entire z and batch_prototypes)
        loss_busemann = busemann_loss(z, batch_prototypes)
        total_busemann_loss += loss_busemann.item()
        
        # Compute InfoNCE Loss (for paired images)
        contrastive_logits, contrastive_labels = hyperbolic_info_nce_logits(features=features_hyp, device=device, n_views=n_views , c=curvature,alpha_d=get_alpha_d(epoch, TASK_EPOCHS))
        contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
        total_contrastive_loss=contrastive_loss.item()
        # Compute adaptive repel_margin in first batch
        if epoch == 0 and batch_idx == 0:
            dists = hypo_dist(z_mix, prototypes)
            mins = dists.min(dim=1).values
            repel_margin = torch.quantile(mins, 0.8).detach()
            print("Setting adaptive repel_margin to", repel_margin.item())
            
        # Outlier loss using adaptive margin
        dist2proto = hypo_dist(z_mix, prototypes)
        min_dist, _ = dist2proto.min(dim=1)
        outlier_loss = F.relu(repel_margin - min_dist).mean()
        total_outlier_loss += outlier_loss.item()
        # Combine losses (weighted sum)
        loss = 0.60 * loss_busemann + 0.25 * contrastive_loss + 0.15 * outlier_loss

        task_optimizer.zero_grad()
        loss.backward()
        task_optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(combine_dataloader)
    avg_busemann_loss = total_busemann_loss / len(combine_dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(combine_dataloader)
    avg_outlier_loss = total_outlier_loss / len(combine_dataloader)
    print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Busemann Loss: {avg_busemann_loss:.6f} | Avg Contrastive Loss: {avg_contrastive_loss:.6f} | Avg Outlier Loss: {avg_outlier_loss:.6f}")

    task_scheduler.step()
    torch.cuda.empty_cache()
    
    if epoch % 5 == 0:
        save_model(model=global_model, path=f"HypDGCD/checkpoints/{args.dataset_name}/{source_domain_name}/Intermediate_{source_domain_name}_hyper_all_in_trained_model_{epoch}.pkl")

