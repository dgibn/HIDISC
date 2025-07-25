# HIDISC

# Domain Generalization in Generalized Category Discovery (DG-GCD) in Hyperbolic Space

This repository contains an initial implementation of a meta-training framework for tackling the **DG-GCD (Domain Generalization in Generalized Category Discovery)** problem using **hyperbolic geometry**. It draws inspiration from the CVPR 2025 paper *"When Domain Generalization meets Generalized Category Discovery: An Adaptive Task-Arithmetic Driven Approach"*, extending its core ideas into the non-Euclidean setting for better hierarchical representation and generalization.

## Project Overview


<img width="790" height="237" alt="Screenshot 2025-07-25 at 3 37 24 PM" src="https://github.com/user-attachments/assets/09676650-7420-44c5-b809-444739cde77e" />


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

## 🧠 Key Components


---

## 🧪 Training
```bash bash_scripts/train.sh``

## 🧪 Testing

```bash bash_scripts/test.sh``

``
## Acknowledgements

Facebook Research for DINO ViT
Geoopt for hyperbolic operations

