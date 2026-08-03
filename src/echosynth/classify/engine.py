"""Classifier training and guarded encounter-level evaluation loops."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from echosynth.classify.models import CompactCNN
from echosynth.eval.guard import TestSetGuard
from echosynth.utils.checkpoint import save_checkpoint
from echosynth.utils.seed import set_seed


@dataclass(frozen=True)
class EvaluationResult:
    """Encounter-level AUROC plus image/encounter predictions."""

    auroc: float
    images: pd.DataFrame
    encounters: pd.DataFrame


def _evaluation_context(
    split: str, guard: TestSetGuard | None
) -> AbstractContextManager[TestSetGuard] | AbstractContextManager[None]:
    from contextlib import nullcontext

    if split == "test":
        if guard is None:
            raise ValueError("TestSetGuard is required for test evaluation")
        return guard
    return nullcontext()


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str]]],
    split: str,
    guard: TestSetGuard | None = None,
    device: torch.device | str = "cpu",
) -> EvaluationResult:
    """Evaluate image probabilities and report only encounter-level AUROC."""
    records: list[dict[str, object]] = []
    model.eval()
    with _evaluation_context(split, guard), torch.no_grad():
        for images, labels, encounter_ids in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for label, probability, encounter_id in zip(
                labels.numpy(), probabilities, encounter_ids, strict=True
            ):
                records.append(
                    {"encounter_id": encounter_id, "label": int(label), "p": float(probability)}
                )
    image_frame = pd.DataFrame(records)
    consistency = image_frame.groupby("encounter_id")["label"].nunique()
    bad = consistency[consistency > 1].index.tolist()
    if bad:
        raise ValueError(f"inconsistent labels within encounters: {bad[:20]}")
    encounter_frame = image_frame.groupby("encounter_id", as_index=False).agg(
        label=("label", "first"), p=("p", "mean")
    )
    score = float(roc_auc_score(encounter_frame["label"], encounter_frame["p"]))
    return EvaluationResult(score, image_frame, encounter_frame)


@dataclass(frozen=True)
class TrainingResult:
    """Best validation checkpoint and selection history."""

    checkpoint: Path
    best_val_auroc: float
    history: tuple[float, ...]


def train_classifier(
    model: CompactCNN,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str]]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, list[str]]],
    out_dir: Path,
    *,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    seed: int = 2024,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Select best checkpoint on validation encounter AUROC, never test loss."""
    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    best = -np.inf
    stale = 0
    history: list[float] = []
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    for epoch in range(epochs):
        model.train()
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation = evaluate(model, val_loader, "val", device=device)
        history.append(validation.auroc)
        metadata = {
            "epoch": epoch,
            "val_auroc": validation.auroc,
            "model_config": model.model_config,
        }
        save_checkpoint(last_path, {"model": model.state_dict()}, metadata)
        if validation.auroc > best:
            best = validation.auroc
            stale = 0
            save_checkpoint(best_path, {"model": model.state_dict()}, metadata)
        else:
            stale += 1
            if stale >= patience:
                break
    return TrainingResult(best_path, float(best), tuple(history))
