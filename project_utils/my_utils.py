import csv,pickle,time
import os
import numpy as np
import random
import torch
from torch import nn
from torch.nn.init import trunc_normal_
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
import argparse
import torch
import numpy as np
import os
from models.vision_transformer import VisionTransformer

def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Setting this to False may slow down training but ensures deterministic results
    
def str2bool(v):
    """
    Convert string to boolean.
    This is useful for argparse when you want to pass boolean flags.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# class GradientReversalFunction(Function):
#     @staticmethod
#     def forward(ctx, x, lambda_):
#         ctx.lambda_ = lambda_
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.neg() * ctx.lambda_, None


# def grad_reverse(x, lambda_=1.0):
#     return GradientReversalFunction.apply(x, lambda_)


# class DomainClassifier(nn.Module):
#     def __init__(self, feature_dim):
#         super(DomainClassifier, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(feature_dim // 2, feature_dim // 4),
#             nn.LeakyReLU(),
#             nn.BatchNorm1d(feature_dim // 4),
#             nn.Linear(feature_dim // 4, feature_dim // 8),
#             nn.LeakyReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(feature_dim // 8, 2)  # Two classes: source vs target
#         )

#     def forward(self, x, reverse=False):
#         if reverse:
#             x = grad_reverse(x)
#         return self.net(x)

def create_list(source_domain: str, num_classes: int = 40) -> list:
    random.seed(42)
    # source_domain = os.path.join("/users/student/pg/pg23/vaibhav.rathore/datasets",source_domain)
    all_folders = [folder for folder in os.listdir(source_domain) if os.path.isdir(os.path.join(source_domain, folder))]
    selected_folders = random.sample(all_folders, num_classes)
    return selected_folders

'''
def create_csv(source_domain: str, aug_domain: str, csv_dir_path: str, selected_classes: list, episode: int) -> tuple:
    col_names = ['index', 'image_path', 'label', 'numeric_label']
    random.seed(42 + episode)
    if len(selected_classes)==40:
        size = random.randint(25,30)
    if len(selected_classes)==4:
        size=random.randint(2,3)
    if len(selected_classes)==250:
        size = random.randint(180,230)
    os.makedirs(csv_dir_path,exist_ok=True)
    csv_train_path = os.path.join(csv_dir_path, f"episode{episode}_source.csv")
    csv_synthetic_path = os.path.join(csv_dir_path, f"episode{episode}_synthetic.csv")
    
    source_labels = []
    train_classes = random.sample(population=selected_classes, k=size)
    # Assign continuous numeric labels to the Source Domain classes
    continuous_numeric_labels = {class_name: idx for idx, class_name in enumerate(train_classes)}
    
    # Assign numeric labels to the remaining classes for the Synthetic Domain
    remaining_classes = [class_name for class_name in selected_classes if class_name not in train_classes]
    next_label = len(train_classes)
    for class_name in remaining_classes:
        continuous_numeric_labels[class_name] = next_label
        next_label += 1
    
    if os.path.exists(csv_train_path) and os.path.exists(csv_synthetic_path):
        return csv_train_path, csv_synthetic_path, size #, labels, labelled_or_not

    else:
        with open(csv_train_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=col_names)
            writer.writeheader()
            index = 0
            class_to_indices = {category: [] for category in train_classes}
            for folder_name in train_classes:
                folder_path = os.path.join(source_domain, folder_name)
                for img in os.listdir(folder_path):
                    if img.endswith('.jpg') or img.endswith('.png'):
                        image_path = os.path.join(folder_path, img)
                        class_to_indices[folder_name].append(index)
                        source_labels.append(continuous_numeric_labels[folder_name])
                        index += 1
                        writer.writerow({
                            'index': index,
                            'image_path': image_path,
                            'label': folder_name,
                            'numeric_label': continuous_numeric_labels[folder_name],
                        })

        with open(csv_synthetic_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=col_names)
            writer.writeheader()
            index = 0
            for folder_name in selected_classes:
                folder_path = os.path.join(aug_domain, folder_name)
                images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png'))]
                for img in images:
                    image_path = os.path.join(folder_path, img)
                    writer.writerow({
                        'index': index,
                        'image_path': image_path,
                        'label': folder_name,
                        'numeric_label': continuous_numeric_labels[folder_name],
                    })
                    index += 1

            target_labels = torch.full((index,), float('nan'), dtype=torch.float)
        return csv_train_path, csv_synthetic_path, size  #,labels, labelled_or_not
'''

def create_combined_csv(combine_csv_path,source_domain: str, synthetic_domains: List[str], selected_classes: List[str],episode: int) -> str:
    col_names = ['index', 'image_path', 'label', 'numeric_label', 'domain_label']
    csv_combined_path = os.path.join(combine_csv_path, f"combined_train{episode}.csv")

    # if os.path.exists(csv_combined_path):
    #     print(f"Combined CSV already exists at: {csv_combined_path}")
    #     return csv_combined_path   

    with open(csv_combined_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=col_names)
        writer.writeheader()
        index = 0

        # Write from source domain
        for numeric_label, folder_name in enumerate(selected_classes):
            folder_path = os.path.join(source_domain, folder_name)
            if not os.path.isdir(folder_path): continue

            for img in os.listdir(folder_path):
                if img.endswith(('.jpg', '.png')):
                    image_path = os.path.join(folder_path, img)
                    writer.writerow({
                        'index': index,
                        'image_path': image_path,
                        'label': folder_name,
                        'numeric_label': numeric_label,
                        'domain_label': 0
                    })
                    index += 1
        print(f"Source domain CSV created at: {csv_combined_path} with {index} entries.")
        print(f"Writing synthetic domains to CSV...")
        # Write from each synthetic domain
        for domain_idx, syn_domain_path in enumerate(synthetic_domains, start=1):  # start=1 to distinguish from source
            for numeric_label, folder_name in enumerate(selected_classes):
                folder_path = os.path.join(syn_domain_path, folder_name)
                if not os.path.isdir(folder_path): continue
                print(f"Writing images from {folder_name} in synthetic domain {domain_idx}...")
                for img in os.listdir(folder_path):
                    if img.endswith(('.jpg', '.png')):
                        image_path = os.path.join(folder_path, img)
                        writer.writerow({
                            'index': index,
                            'image_path': image_path,
                            'label': folder_name,
                            'numeric_label': numeric_label,
                            'domain_label': domain_idx
                        })
                        index += 1

    print(f"Combined CSV created at: {csv_combined_path} with {index} entries.")
    return csv_combined_path


def create_target_csv(target_domain: str, csv_dir_path: str, selected_classes: list, split: int) -> str:
    '''
    Takes image-dataset from Target Domain.
    All the 65 classes will be taken to csv file and includes triplets (positive and negative anchors).
    
    The selected_classes are assigned continuous numeric labels from 0 to 39.
    Then, the remaining classes are assigned continuous numeric labels from 40 to 64.
    '''
    col_names = ['index', 'image_path', 'label', 'continuous_numeric_label', 'positive_anchor', 'neg_anchor']

    domain_name = os.path.basename(target_domain)
    csv_target_filename = "_".join([domain_name,"target.csv"])
    csv_target_path = os.path.join(csv_dir_path, csv_target_filename)

    with open(csv_target_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=col_names)
        writer.writeheader()

        index = 0
        # continuous_numeric_label = 0
        continuous_label_mapping = {}
        class_to_images = {}
        
        # Assign continuous numeric labels to selected classes
        for i, class_name in enumerate(selected_classes):
            continuous_label_mapping[class_name] = i
        # Assign continuous numeric labels to the remaining classes
        remaining_classes = [cls for cls in os.listdir(target_domain) if cls not in selected_classes]
        for i, class_name in enumerate(remaining_classes, start = split):
            continuous_label_mapping[class_name] = i

        # Gather all images by class
        for folder_name in os.listdir(target_domain):
            folder_path = os.path.join(target_domain, folder_name)
            images = [img for img in os.listdir(folder_path) if img.endswith('.jpg') or img.endswith('.png')]
            class_to_images[folder_name] = images

        # Write rows and generate triplets
        for folder_name, images in class_to_images.items():
            folder_path = os.path.join(target_domain, folder_name)
            for img in images:
                image_path = os.path.join(folder_path, img)

                # Select a positive anchor (different image from the same class)
                positive_index = random.randint(0, len(images) - 1)
                while images[positive_index] == img:
                    positive_index = random.randint(0, len(images) - 1)
                positive_image_path = os.path.join(folder_path, images[positive_index])

                # Select a negative anchor (image from a different class)
                negative_label = folder_name
                while negative_label == folder_name:
                    negative_label = random.choice(list(class_to_images.keys()))
                negative_folder_path = os.path.join(target_domain, negative_label)
                negative_images = class_to_images[negative_label]
                negative_index = random.randint(0, len(negative_images) - 1)
                negative_image_path = os.path.join(negative_folder_path, negative_images[negative_index])

                writer.writerow({
                    'index': index,
                    'image_path': image_path,
                    'label': folder_name,
                    'continuous_numeric_label': continuous_label_mapping[folder_name],
                    'positive_anchor': positive_image_path,
                    'neg_anchor': negative_image_path
                })
                index += 1

    return csv_target_path


def create_TrainTest_target_csv(csv_dir_path: str, csv_path: str) -> tuple[str, str]:
    
    # Read the generated CSV file
    df = pd.read_csv(csv_path)
    
    # Split the dataset into two based on the class labels
    classes_40 = df[df['continuous_numeric_label'].between(0, 39)]
    classes_65 = df[df['continuous_numeric_label'].between(40, 64)]
    # For classes common in both CSV files, split the images accordingly
    dfs_40 = []
    dfs_65 = []
    
    for class_label in classes_40['continuous_numeric_label'].unique():
        class_images = classes_40[classes_40['continuous_numeric_label'] == class_label]
        split_idx = int(0.4 * len(class_images))
        dfs_40.append(class_images.iloc[:split_idx])
        dfs_65.append(class_images.iloc[split_idx:])
    
    # Combine with other classes
    df_40_classes = pd.concat(dfs_40)                           # contains the 40% of images from the common classes (0-39).
    df_65_classes = pd.concat(dfs_65 + [classes_65])            # Contains the remaining 60% of images from the common classes (0-39),
                                                                # plus all images from classes (40-64)
    
    # Save the new CSV files
    path_40_classes = f"{csv_dir_path}/train_target.csv"
    path_65_classes = f"{csv_dir_path}/test_target.csv"
    
    df_40_classes.to_csv(path_40_classes, index=False)
    df_65_classes.to_csv(path_65_classes, index=False)
    
    return path_40_classes, path_65_classes


def test_kmeans_hidisc(model, 
                     test_loader,
                     epoch, 
                     save_name,
                     num_train_classes,
                     device,
                     num_unlabelled_classes):   

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, data in enumerate(tqdm(test_loader)):
        images, label = data
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
            label = torch.cat([label for _ in range(2)]).to(device)
        else:
            images = images.to(device)
            label = label.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(num_train_classes)
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_train_classes + num_unlabelled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=['v1', 'v2'], save_name=save_name)

    return all_acc, old_acc, new_acc

def test_kmeans_mix(model, 
                     test_loader,
                     epoch, 
                     save_name,
                     num_train_classes,
                     device,
                     num_unlabelled_classes):   

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, data in enumerate(tqdm(test_loader)):
        images, label = data
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
            label = torch.cat([label for _ in range(2)]).to(device)
        else:
            images = images.to(device)
            label = label.to(device)
        # images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats_euc = model(images)

        feats = torch.nn.functional.normalize(feats_euc, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(num_train_classes)
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_train_classes + num_unlabelled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=['v1', 'v2'], save_name=save_name)

    return all_acc, old_acc, new_acc

def cluster_acc(y_true, y_pred, return_ind=False):
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity_score(y_true, y_pred):
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def evaluate_clustering(y_true, y_pred):
    acc = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    return acc, nmi, ari, pur

def test_kmeans(K, all_feats, targets, mask_lab, verbose=False):
    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    mask = mask_lab

    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask], preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    if verbose:
        print('K')
        print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi, labelled_ari))
        print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

    return labelled_acc

def test_kmeans_for_scipy(K, all_feats, targets, mask_lab, verbose=False):
    K = int(K)

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    mask = mask_lab

    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask], preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi, labelled_ari))
    print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

    return -labelled_acc

def binary_search(all_feats, targets, mask_lab, num_labeled_classes, max_classes):
    min_classes = num_labeled_classes
    big_k = max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big = test_kmeans(big_k, all_feats, targets, mask_lab)
    labelled_acc_small = test_kmeans(small_k, all_feats, targets, mask_lab)
    labelled_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_lab)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):
        if labelled_acc_big > labelled_acc_small:
            best_acc = max(labelled_acc_middle, labelled_acc_big)
            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
        else:
            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_lab)

        print(f'Iter {i}: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')
        return best_acc_at_k

def scipy_optimise(all_feats, targets, mask_lab, num_labeled_classes, max_classes):
    from functools import partial
    from scipy.optimize import minimize_scalar

    small_k = num_labeled_classes
    big_k = max_classes
    test_k_means_partial = partial(test_kmeans_for_scipy, all_feats=all_feats, targets=targets, mask_lab=mask_lab, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x}')
    return res.x

def semi_supervised_kmeans(features, labels, mask_lab, num_known_classes, total_clusters):
    # Initialize centroids for known classes
    known_centroids = [features[labels == i].mean(axis=0) for i in range(num_known_classes)]
    
    # Apply k-means++ on the unlabeled dataset to initialize centroids for unknown classes
    kmeans_plus = KMeans(n_clusters=total_clusters-num_known_classes, init='k-means++')
    kmeans_plus.fit(features[~mask_lab])
    unknown_centroids = kmeans_plus.cluster_centers_
    
    # Combine known and unknown centroids
    centroids = np.vstack([known_centroids, unknown_centroids])
    
    # Perform k-means with initialized centroids
    kmeans = KMeans(n_clusters=total_clusters, init=centroids, n_init=1)
    predicted_labels = kmeans.fit_predict(features)
    
    return predicted_labels

def split_cluster_acc_v1(y_true, y_pred, mask):
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, print_output=False):
    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):
        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'
        
        if i == 1:                                              # i=0->v1,   i=1->v2
            to_return = (all_acc, old_acc, new_acc)
        
        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def get_entropy(features, centers, temperature=0.7):
    return F.cosine_similarity(features.unsqueeze(1), centers, dim = 2)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # Current value
        self.avg = 0  # Average value
        self.sum = 0  # Sum of all values
        self.count = 0  # Count of values

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Save the model at the end of training
def save_model(model, path="final_model.pkl"):
    print("=> Saving the final model")
    # torch.save(model.state_dict(), path)
    t1 = time.time()
    pickle.dump(model,open(path,"wb"))
    t2 = time.time()
    t = t2-t1
    print(f"The model is saved at {path} and it took {t/60:.2f} mints")
    
# Load the saved models for further use
def load_model(path):
    print("=> Loading model from", path)
    t1 = time.time()
    model=pickle.load(open(path,"rb"))
    t2 = time.time()
    t = t2-t1
    print(f"The model is loaded from {path} and it took {t/60:.2f} mints")
    return model
