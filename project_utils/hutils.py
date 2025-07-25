import numpy as np
import torch

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

#-------------------------
#----- Poincaré Disk -----
#-------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)
    
# distance in poincare disk
def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

# compute symmetric poincare distance matrix
def poincare_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array from poincare disk to hyperboloid
def poincare_pts_to_hyperboloid(Y, eps=1e-6, metric='lorentz'):
    mink_pts = np.zeros((Y.shape[0], Y.shape[1]+1))
    r = norm(Y, axis=1)
    if metric == 'minkowski':
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * Y[:, 1]
    else:
        mink_pts[:, 0] = 2/(1 - r**2 + eps) * Y[:, 0]
        mink_pts[:, 1] = 2/(1 - r**2 + eps) * Y[:, 1]
        mink_pts[:, 2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    return mink_pts

# convert single point to hyperboloid
def poincare_pt_to_hyperboloid(y, eps=1e-6, metric='lorentz'):
    mink_pt = np.zeros((3, ))
    r = norm(y)
    if metric == 'minkowski':
        mink_pt[0] = 2/(1 - r**2 + eps) * (1 + r**2)/2
        mink_pt[1] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[2] = 2/(1 - r**2 + eps) * y[1]
    else:
        mink_pt[0] = 2/(1 - r**2 + eps) * y[0]
        mink_pt[1] = 2/(1 - r**2 + eps) * y[1]
        mink_pt[2] = 2/(1 - r**2 + eps) * (1 + r**2)/2
    return mink_pt

#------------------------------
#----- Hyperboloid Model ------
#------------------------------

# NOTE: POSSIBLE ISSUE WITH DIFFERENT WAYS TO SPECIFY MINKOWSKI DOT PRODUCT
# arbritray sign gives different signatures (+, +, +, -), (+, -, -, -)

# define hyperboloid bilinear form
def hyperboloid_dot(u, v):
    return np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]

# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 

# hyperboloid distance function
def hyperboloid_dist(u, v, eps=1e-6, metric='lorentz'):
    if metric == 'minkowski':
        dist = np.arccosh(-1*minkowski_dot(u, v))
    else:
        dist = np.arccosh(-1*hyperboloid_dot(u, v))
    if np.isnan(dist):
        #print('Hyperboloid dist returned nan value')
        return eps
    else:
        return dist

# compute symmetric hyperboloid distance matrix
def hyperboloid_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = hyperboloid_dist(embedding[i], embedding[j])
    return dist_matrix

# convert array to poincare disk
def hyperboloid_pts_to_poincare(X, eps=1e-6, metric='lorentz'):
    poincare_pts = np.zeros((X.shape[0], X.shape[1]-1))
    if metric == 'minkowski':
        poincare_pts[:, 0] = X[:, 1] / ((X[:, 0]+1) + eps)
        poincare_pts[:, 1] = X[:, 2] / ((X[:, 0]+1) + eps)
    else:
        poincare_pts[:, 0] = X[:, 0] / ((X[:, 2]+1) + eps)
        poincare_pts[:, 1] = X[:, 1] / ((X[:, 2]+1) + eps)
    return poincare_pts

# project within disk
def proj(theta,eps=0.1):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta

# convert single point to poincare
def hyperboloid_pt_to_poincare(x, eps=1e-6, metric='lorentz'):
    poincare_pt = np.zeros((2, ))
    if metric == 'minkowski':
        poincare_pt[0] = x[1] / ((x[0]+1) + eps)
        poincare_pt[1] = x[2] / ((x[0]+1) + eps)
    else:
        poincare_pt[0] = x[0] / ((x[2]+1) + eps)
        poincare_pt[1] = x[1] / ((x[2]+1) + eps)
    return proj(poincare_pt)
    
# helper function to generate samples
def generate_data(n, radius=0.7, hyperboloid=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    if hyperboloid:
        return poincare_pts_to_hyperboloid(init_data)
    else:
        return init_data
    
def clip_feature(z: torch.Tensor, r: float = 1.0) -> torch.Tensor:
    """
    Clips the Euclidean feature vector to a ball of radius `r` before hyperbolic mapping.

    Args:
        z (torch.Tensor): Input tensor of shape (batch_size, feature_dim).
        r (float): Clipping radius.

    Returns:
        torch.Tensor: Clipped feature tensor of the same shape.
    """
    norm = torch.norm(z, p=2, dim=-1, keepdim=True)  # ||z||_2
    scale = torch.clamp(r / norm, max=1.0)            # min{1, r / ||z||}
    return z * scale

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Möbius addition for Poincaré ball model.

    Args:
        x, y: Tensors of shape (..., D)
        c: Positive curvature scalar (actual curvature is -c²)

    Returns:
        Möbius sum tensor of shape (..., D)
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c * c * x2 * y2
    return numerator / (denominator + 1e-5)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """
    Hyperbolic distance (Poincaré distance) between two points x and y.

    Args:
        x, y (torch.Tensor): Tensors of shape (batch_size, dim) in the Poincaré ball.
        c (float): Curvature (positive scalar; actual curvature is -c^2).

    Returns:
        torch.Tensor: Distance tensor of shape (batch_size, 1).
    """
    mobius_diff = mobius_add(-x, y, c)
    norm = torch.norm(mobius_diff, p=2, dim=-1, keepdim=True)
    dist = 2 / (c**0.5) * torch.atanh(torch.clamp(c**0.5 * norm, max=1 - 1e-5))
    return dist

def get_alpha_d(epoch, max_epoch, alpha_max=1.0):
    """
    Linearly increase alpha_d from 0 to alpha_max over training epochs.

    Args:
        epoch (int): Current epoch.
        max_epoch (int): Total number of training epochs.
        alpha_max (float): Final value of alpha_d.

    Returns:
        float: alpha_d for the current epoch.
    """
    return min(alpha_max * epoch / max_epoch, alpha_max)


def hypo_dist(x, P, c=1.0, eps=1e-6):
    """
    Hyperbolic distance in the Poincaré ball:
      dB(x, p) = arcosh( 1 + 2||x-p||^2 / ((1-||x||^2)*(1-||p||^2)) )

    x:  Tensor[B, D]
    P:  Tensor[K, D]
    c:  curvature (if !=1, divide x, P by sqrt(c) before computing)
    eps: small constant to avoid div/√(negatives)
    returns: Tensor[B, K]
    """
    # squared norms
    x2 = x.pow(2).sum(dim=1, keepdim=True)             # [B,1]
    p2 = P.pow(2).sum(dim=1, keepdim=True).T           # [1,K]

    # pairwise squared distances
    diff2 = (x.unsqueeze(1) - P.unsqueeze(0)).pow(2).sum(dim=2)  # [B,K]

    # numerator and denominator, with clamping
    num = 2 * diff2
    den = (1 - x2).clamp(min=eps) * (1 - p2).clamp(min=eps)     # [B,K]

    arg = 1 + num / den
    # ensure argument ≥ 1 + eps for arccosh
    arg = arg.clamp(min=1 + eps)

    return torch.acosh(arg)  # [B,K]
