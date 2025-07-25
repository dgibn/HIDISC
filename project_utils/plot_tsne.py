from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_tsne(tsne_dir: str, domain_name: str, X: np.ndarray, labels: np.ndarray, episode: int, class_names: dict,epoch :int,hyp : bool) -> None:
    # Run t-SNE on the data
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=42)
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
    if hyp:
        image_path = os.path.join(domain_folder, f"HypCD_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}_epoch{epoch}.png")
    else:
        image_path = os.path.join(domain_folder, f"EucCD_train_{os.path.basename(tsne_dir)}_test_{domain_name}_checkpoint{episode}_epoch{epoch}.png")
    plt.savefig(image_path, dpi=300)
    plt.close()