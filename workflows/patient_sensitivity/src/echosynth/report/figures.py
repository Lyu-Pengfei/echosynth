"""Regenerate publication figures from run predictions and embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from echosynth.eval.embedding import plot_embeddings
from echosynth.eval.metrics import auprc, auroc


def _figure_three(run: Path, output: Path) -> Path | None:
    artifact = run / "embeddings.npz"
    if not artifact.is_file():
        return None
    values = np.load(artifact)
    return plot_embeddings(values["features"], values["source"], values["classes"], output)


def _figure_four(run: Path, output: Path) -> Path | None:
    paths = {name: run / f"predictions_{name}.csv" for name in ("TRTR", "TSTR")}
    if not all(path.is_file() for path in paths.values()):
        return None
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    colors = {"TRTR": "#0072B2", "TSTR": "#D55E00"}
    for name, frame in frames.items():
        fpr, tpr, _ = roc_curve(frame["label"], frame["p"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["p"])
        axes[0].plot(
            fpr,
            tpr,
            color=colors[name],
            label=f"{name} AUROC={auroc(frame['label'], frame['p']):.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=colors[name],
            label=f"{name} AUPRC={auprc(frame['label'], frame['p']):.3f}",
        )
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes:
        axis.legend(frameon=False)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)
    return output


def make_figures(runs_dir: Path, out_dir: Path) -> list[Path]:
    """Write Figure 3 (t-SNE/UMAP) and Figure 4 (TRTR/TSTR ROC/PR) when artifacts exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for run in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if not outputs:
            figure = _figure_three(run, out_dir / "figure3.png")
            if figure is not None:
                outputs.append(figure)
        if not any(path.name == "figure4.png" for path in outputs):
            figure = _figure_four(run, out_dir / "figure4.png")
            if figure is not None:
                outputs.append(figure)
    return outputs
