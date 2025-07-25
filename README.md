# HIDISC

# Domain Generalization in Generalized Category Discovery (DG-GCD) in Hyperbolic Space

This repository contains an initial implementation of a meta-training framework for tackling the **DG-GCD (Domain Generalization in Generalized Category Discovery)** problem using **hyperbolic geometry**. It draws inspiration from the CVPR 2025 paper *"When Domain Generalization meets Generalized Category Discovery: An Adaptive Task-Arithmetic Driven Approach"*, extending its core ideas into the non-Euclidean setting for better hierarchical representation and generalization.

## Project Overview

**HIDISC**, a hyperbolic framework that jointly addresses the dual challenges of DG-GCD: domain-invariant representation learning and unsupervised semantic disentanglement. The
synthesis-driven components of the model include: (i) Synthetic Domain Augmentation, which introduces a compact set of diverse, diffusion-generated domains to simulate realistic distribution shifts
without relying on target access; (ii) Tangent CutMix, a curvature-aware interpolation mechanism operating in the tangent space of the Poincaré ball, generating pseudo-novel samples while preserving manifold fidelity. 

Complementing these are three loss-driven modules: (iii) Prototype Anchoring, which aligns seen-class embeddings to fixed ideal prototypes on the Poincaré boundary, reserving central space for novel classes; (iv) Adaptive Outlier Loss, which ensures synthetic samples are repelled from known-class clusters, promoting open-space regularization; and (v) Hybrid Hyperbolic Contrastive Loss, which combines geodesic and angular similarity to improve local cohesion and global separability.

<center> <img width="790" height="237" alt="Screenshot 2025-07-25 at 3 37 24 PM" src="https://github.com/user-attachments/assets/09676650-7420-44c5-b809-444739cde77e"/> </center> 

---

## ⚙️ Requirements
- Python 3.8+
- PyTorch >= 1.10
- torchvision
- tqdm
- numpy
- pandas
- [Geoopt](https://github.com/geoopt/geoopt) for hyperbolic geometry (planned)
- DINO pre-trained ViT from FacebookResearch

---
## 🧪 Running the Code

i) For training 
```bash bash_scripts/train.sh``

ii) For testing
```bash bash_scripts/test.sh``

``
## Acknowledgements

Facebook Research for DINO ViT
Geoopt for hyperbolic operations

