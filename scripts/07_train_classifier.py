"""CLI for a validation-selected binary CompactCNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from echosynth.classify.engine import train_classifier
from echosynth.classify.models import CompactCNN
from echosynth.classify.transforms import build_transform
from echosynth.data.datasets import EchoClassificationDataset
from echosynth.data.manifest import read_manifest
from echosynth.utils.config import load_config, save_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs/classifier/ckpt"))
    parser.add_argument("--positive", default="CA")
    args = parser.parse_args()
    resolved = load_config(args.config)
    config = resolved.to_dict()
    settings = config.get("classifier", config)
    manifest = args.manifest or Path(config.get("data", {}).get("manifest", ""))
    frame = read_manifest(manifest)
    frame["target"] = frame["group"].astype(str).eq(args.positive).astype(int).astype(str)
    loaders = {}
    for split, training in (("train", True), ("val", False)):
        selected = frame.loc[frame["split"].astype(str) == split]
        dataset = EchoClassificationDataset(
            selected, "target", {"0": 0, "1": 1}, build_transform(training, 112)
        )
        loaders[split] = DataLoader(
            dataset, batch_size=int(settings.get("batch_size", 32)), shuffle=training
        )
    model = CompactCNN(
        channels=tuple(settings.get("channels", (32, 64, 128, 256))),
        dropout=float(settings.get("dropout", 0.3)),
    )
    result = train_classifier(
        model,
        loaders["train"],
        loaders["val"],
        args.out_dir,
        epochs=int(settings.get("epochs", 100)),
        patience=int(settings.get("patience", 15)),
        lr=float(settings.get("lr", 1e-3)),
        seed=int(config.get("seed", 2024)),
    )
    save_resolved(resolved, args.out_dir.parent)
    print(f"checkpoint={result.checkpoint} val_auroc={result.best_val_auroc:.4f}")


if __name__ == "__main__":
    main()
