import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from project_utils.hutils import *
# from project_utils.pmath import *
from typing import Union

def prototype_loss(prototypes):
    """
    Computes the loss based on maximum cosine similarity between distinct prototypes.
    Args:
        prototypes: Tensor of shape (num_classes, prototype_dim) assumed to be normalized.
    Returns:
        A tuple of (loss, max_similarity)
    """
    # Compute cosine similarity matrix (prototypes are already unit-normalized)
    sim_matrix = torch.matmul(prototypes, prototypes.t()) + 1.0  # shift to have positive values
    # Remove self-similarity by filling the diagonal with zeros
    sim_matrix.fill_diagonal_(0)
    # For each prototype, select the maximum similarity (undesirable high similarity indicates low separation)
    loss_per_proto, _ = sim_matrix.max(dim=1)
    loss = loss_per_proto.mean()
    return loss, sim_matrix.max()

class PenalizedBusemannLoss(nn.Module):
    """
    Implements the penalized Busemann loss for hyperbolic learning:
    
        ℓ(z, p) = log(||p - z||²) - (phi + 1) * log(1 - ||z||²)
    
    where:
      - z are the hyperbolic embeddings (assumed to lie in the Poincaré ball, ||z|| < 1)
      - p are the corresponding class prototypes on the ideal boundary (||p|| = 1)
      - phi is the penalty scalar.
    
    An epsilon is added for numerical stability.
    """
    def __init__(self, phi, eps=1e-6):
        super(PenalizedBusemannLoss, self).__init__()
        self.phi = phi    # The penalty term, e.g. phi = 0.75 (or phi = s * d if you follow the paper)
        self.eps = eps

    def forward(self, z, p):
        """
        Args:
            z: Tensor of shape (batch_size, dims) representing hyperbolic embeddings.
            p: Tensor of shape (batch_size, dims) corresponding to the ideal prototypes for each example.
               (Typically, p is obtained by indexing your prototype tensor with the ground-truth labels.)
        
        Returns:
            The scalar penalized Busemann loss averaged over the batch.
        """
        # Ensure the squared norm of z is strictly less than one for numerical stability.
        z_norm_sq = torch.clamp(torch.sum(z ** 2, dim=1), 0, 1 - self.eps)
        # Compute the squared Euclidean distance between the embedding and its prototype.
        diff_norm_sq = torch.sum((p - z) ** 2, dim=1) + self.eps
        # Compute the loss per sample
        # Note: b_p(z) = log( diff_norm_sq ) - log(1 - z_norm_sq)
        # Then, the full loss becomes:
        #      log(diff_norm_sq) - (phi + 1)*log(1 - z_norm_sq)
        loss = torch.log(diff_norm_sq) - (self.phi + 1) * torch.log(1 - z_norm_sq)
        return torch.mean(loss)


def hyperbolic_similarity_matrix(features, c, alpha_d):
    """
    Compute combined similarity matrix using both hyperbolic distance and cosine similarity.
    Args:
        features: (N, D) tensor in Poincaré ball
        c: curvature
        alpha_d: weight for distance similarity
    Returns:
        (N, N) similarity matrix
    """
    N = features.size(0)

    # Compute hyperbolic pairwise distance matrix
    x_i = features.unsqueeze(1)  # (N, 1, D)
    x_j = features.unsqueeze(0)  # (1, N, D)
    diff = x_i - x_j
    mobius_diff = mobius_add(-x_i, x_j, c)  # (N, N, D)
    dist_matrix = torch.norm(mobius_diff, dim=-1)  # (N, N)
    dist_sim = -2 / (c**0.5) * torch.atanh(torch.clamp(c**0.5 * dist_matrix, max=1 - 1e-5))

    # Cosine similarity matrix (angle-based)
    features_norm = F.normalize(features, dim=1)
    cos_sim = torch.matmul(features_norm, features_norm.T)  # (N, N)

    # Combine both similarities
    sim_matrix = alpha_d * dist_sim + (1 - alpha_d) * cos_sim
    return sim_matrix

def hyperbolic_info_nce_logits(features, device, temperature=0.7, n_views=2, c=0.01, alpha_d=0.5):
    """
    Compute logits and labels for InfoNCE in hyperbolic space.
    """
    b_ = int(features.size(0) // n_views)
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    # Similarity matrix in hyperbolic space
    sim_matrix = hyperbolic_similarity_matrix(features, c, alpha_d)

    # Remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    # Positive and negative splits
    positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


class HyperbolicSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(HyperbolicSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, c=0.01, alpha_d=0.5):
        """
        Args:
            features: Tensor of shape [bsz, n_views, dim], already in hyperbolic space.
            labels: Tensor of shape [bsz], class labels.
            c: Hyperbolic curvature.
            alpha_d: Distance-based loss weight (linearly increased during training).
        """
        device = features.device
        if len(features.shape) != 3:
            raise ValueError("Expected shape [bsz, n_views, dim]")

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        features = features.view(batch_size * contrast_count, -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Labels do not match features")

        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(contrast_count, contrast_count)

        # Compute pairwise similarities
        features_anchor = features.unsqueeze(1)  # (N, 1, D)
        features_contrast = features.unsqueeze(0)  # (1, N, D)

        # --- Hyperbolic distance ---
        diff = mobius_add(-features_anchor, features_contrast, c)
        dists = torch.norm(diff, dim=-1)
        dist_sim = -2 / (c**0.5) * torch.atanh(torch.clamp(c**0.5 * dists, max=1 - 1e-5))  # (N, N)

        # --- Angular similarity ---
        features_norm = F.normalize(features, dim=-1)
        ang_sim = torch.matmul(features_norm, features_norm.T)  # (N, N)

        # --- Combined similarity matrix ---
        sim_matrix = alpha_d * dist_sim + (1 - alpha_d) * ang_sim
        sim_matrix = sim_matrix / self.temperature

        # --- Numerical stability ---
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask out self-comparisons
        logits_mask = ~torch.eye(batch_size * contrast_count, dtype=torch.bool).to(device)
        mask = mask * logits_mask.float()

        # Log probability
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Mean log-likelihood over positives
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # avoid divide-by-zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss