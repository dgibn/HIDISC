# HIDISC: A Hyperbolic Framework for Domain Generalization with Generalized Category Discovery

This repository contains the official implementation of the paper **"HIDISC: A Hyperbolic Framework for Domain Generalization with Generalized Category Discovery"** by Vaibhav Rathore, Divyam Gupta, and Biplab Banerjee.

[![Conference](https://img.shields.io/badge/Conference-grey)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-red)](https://arxiv.org/pdf/2510.17188)
[![Project](https://img.shields.io/badge/Project-blue)](http://vaibhavrathore1999.github.io/HiDISC/)
[![Poster](https://img.shields.io/badge/Poster-yellow)](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/119696.png?t=1763530622.5565956)

## 📄 Abstract

HIDISC addresses the challenging problem of **Domain Generalization (DG)** combined with **Generalized Category Discovery (GCD)**. In this setting, a model trained on a source domain must generalize to unseen target domains, recognizing known classes while simultaneously discovering and clustering novel classes.

Traditional Euclidean methods often struggle to capture the complex, hierarchical relationships inherent in such open-world scenarios. HIDISC leverages **hyperbolic geometry** (specifically the Poincaré ball model) to learn representations that naturally accommodate hierarchy and improve the separation between known and novel categories.

### Key Features
*   **Hyperbolic Representation Learning:** Maps visual features to a hyperbolic manifold with learnable curvature to better represent hierarchical data structures.
*   **Tangent CutMix:** A novel augmentation strategy performed in the tangent space (Euclidean approximation) to synthesize pseudo-novel samples, simulating open-world shifts during training.
*   **Unified Hyperbolic Loss:**
    *   **Penalized Busemann Loss:** Aligns samples with class prototypes using Busemann functions.
    *   **Hyperbolic InfoNCE:** Enforces contrastive constraints in hyperbolic space.
    *   **Adaptive Outlier Repulsion:** Ensures novel samples are pushed away from known class prototypes.

## 🛠️ Installation

### Prerequisites
*   Linux
*   NVIDIA GPU + CUDA
*   Anaconda or Miniconda

### Environment Setup
Create the conda environment using the provided `environment.yaml` file:

```bash
# Create the environment
conda env create -f environment.yaml

# Activate the environment
conda activate dgcd
```

## 📂 Data Preparation

The framework supports standard DG benchmarks: **Office-Home**, **PACS**, and **DomainNet**.

Ensure your dataset paths are correctly configured in the CSV files located in the `data/` directory:
*   `data/dataset_path_Office_Home.csv`
*   `data/dataset_path_PACS.csv`
*   `data/dataset_path_Domain_Net.csv`

These files should map image paths to their corresponding labels and domains.

For detailed instructions on setting up the synthetic datasets, please refer to the guide in the D_GCD repository:
[DATASET.md](https://github.com/Shubh-Nil/D_GCD/blob/main/DATASET.md)

## 🚀 Training

To train the model, you can use the `scripts/train.py` script. The model is trained on a specific source domain, treating others as unseen target domains.

### Single Domain Training
```bash
python scripts/train.py \
    --dataset_name Office_Home \
    --source_domain_name Art \
    --do_hyperbolic True \
    --c 0.05 \
    --penalty 0.75 \
    --task_epochs 50 \
    --batch_size 128
```

### Automated Training Script
Use the provided bash script to automate training across multiple domains defined in the script:

```bash
bash bash_scripts/train.sh
```
*Note: Edit `bash_scripts/train.sh` to select the dataset and domains you wish to train on.*

### Key Arguments
*   `--dataset_name`: Name of the dataset (e.g., `Office_Home`, `PACS`).
*   `--source_domain_name`: The domain to train on (e.g., `Art`, `photo`).
*   `--do_hyperbolic`: Set to `True` to enable HIDISC (Hyperbolic) mode.
*   `--c`: Initial curvature for the hyperbolic space.
*   `--penalty`: Penalty weight for the Busemann loss.

## 🧪 Testing & Evaluation

Evaluation involves testing the trained model on unseen target domains to measure clustering accuracy on both known and novel classes.

### Single Checkpoint Testing
```bash
python scripts/test.py --checkpoint 50 --dataset_name Office_Home
```

### Automated Testing Script
Use the `test.sh` script to evaluate a range of checkpoints automatically.

**Usage:**
```bash
./bash_scripts/test.sh [DATASET_NAME] [START_CHECKPOINT] [END_CHECKPOINT] [STEP]
```

**Example:**
```bash
# Test Office_Home checkpoints 0, 5, 10, ..., 50
bash bash_scripts/test.sh Office_Home 0 50 5
```

Results are saved to `HIDISC/Results/<dataset_name>/HIDISC_Results.xlsx`.

## 📁 Project Structure

```
HIDISC/
├── bash_scripts/       # Helper shell scripts for training and testing
├── checkpoints/        # Saved model checkpoints
├── data/               # Data utilities, augmentations, and path CSVs
├── models/             # Network architectures (ViT, Decoders, etc.)
├── project_utils/      # Core utilities (Losses, Hyperbolic ops, Data setup)
├── scripts/            # Main Python scripts (train.py, test.py)
├── Results/            # Evaluation outputs
├── environment.yaml    # Conda environment specification
└── README.md           # Project documentation
```

## 📚 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{Rathore2025HiDISC,
  author    = {Rathore, Vaibhav and Gupta, Divyam and Banerjee, Biplab},
  title     = {HiDISC: A Hyperbolic Framework for Domain Generalization with Generalized Category Discovery},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

This repository also utilizes code from the following work:

```bibtex
@InProceedings{vaze2022gcd,
               title={Generalized Category Discovery},
               author={Sagar Vaze and Kai Han and Andrea Vedaldi and Andrew Zisserman},
               booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
               year={2022}}
```

