"""Qualitative t-SNE and UMAP diagnostics without inferential statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


def plot_embeddings(
    features: np.ndarray,
    source_labels: np.ndarray,
    class_labels: np.ndarray,
    output: Path,
    *,
    seed: int = 2024,
) -> Path:
    """Plot t-SNE and UMAP from one feature matrix as qualitative diagnostics only."""
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 4:
        raise ValueError("embedding features must be a 2-D matrix with at least four rows")
    if len(source_labels) != len(matrix) or len(class_labels) != len(matrix):
        raise ValueError("embedding labels must align with feature rows")
    perplexity = min(30.0, max(2.0, (len(matrix) - 1) / 3))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(matrix)
    neighbors = min(15, len(matrix) - 1)
    umap = UMAP(n_components=2, random_state=seed, n_neighbors=neighbors, n_jobs=1).fit_transform(
        matrix
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    palette = {"real": "#0072B2", "synthetic": "#D55E00"}
    classes = np.unique(class_labels)
    markers = ("o", "s", "^", "D", "P", "X")
    for axis, coordinates, title in zip(axes, (tsne, umap), ("t-SNE", "UMAP"), strict=True):
        for source in np.unique(source_labels):
            for class_index, class_label in enumerate(classes):
                selected = (source_labels == source) & (class_labels == class_label)
                if selected.any():
                    axis.scatter(
                        coordinates[selected, 0],
                        coordinates[selected, 1],
                        s=18,
                        alpha=0.7,
                        label=f"{source} / {class_label}",
                        color=palette.get(str(source)),
                        marker=markers[class_index % len(markers)],
                    )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output
