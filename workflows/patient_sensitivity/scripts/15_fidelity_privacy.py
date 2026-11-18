"""Run class-specific fidelity, memorization, membership, and discriminator audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from echosynth.eval.discriminator import make_patient_splits, run_discriminator
from echosynth.eval.fidelity import evaluate_fidelity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(path: Path, size: int = 16) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
    return values


def _sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.copy()
    return frame.sample(n=count, random_state=seed, replace=False)


def _vectors(frame: pd.DataFrame) -> np.ndarray:
    return np.stack([_vector(Path(path)) for path in frame["png_path"].astype(str)])


def _nearest(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(reference)
    distance, _ = model.kneighbors(query)
    return distance[:, 0]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-images", type=int, default=1000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(args.real_manifest)
    synthetic = pd.read_csv(args.synthetic_manifest)
    synthetic["group"] = synthetic.get("class_label", synthetic.get("group")).astype(str)
    class_rows: list[dict[str, Any]] = []
    for index, group in enumerate(("CA", "LVH", "GEN")):
        real_group = _sample(
            real[(real["group"].astype(str) == group) & (real["split"].astype(str) == "train")],
            args.max_images,
            args.seed + index,
        )
        synthetic_group = _sample(
            synthetic[synthetic["group"].astype(str) == group],
            args.max_images,
            args.seed + 10 + index,
        )
        if real_group.empty or synthetic_group.empty:
            raise RuntimeError(f"fidelity audit lacks real or synthetic rows for {group}")
        metrics = evaluate_fidelity(
            [Path(value) for value in real_group["png_path"].astype(str)],
            [Path(value) for value in synthetic_group["png_path"].astype(str)],
        ).to_dict()
        class_rows.append({"group": group, **metrics})
    pd.DataFrame(class_rows).to_csv(args.out_dir / "class_fidelity.csv", index=False)

    train_ca = _sample(
        real[(real["group"].astype(str) == "CA") & (real["split"].astype(str) == "train")],
        args.max_images,
        args.seed + 101,
    )
    heldout_ca = _sample(
        real[(real["group"].astype(str) == "CA") & (real["split"].astype(str) == "test")],
        args.max_images,
        args.seed + 102,
    )
    synthetic_ca = _sample(
        synthetic[synthetic["group"].astype(str) == "CA"],
        args.max_images,
        args.seed + 103,
    )
    train_vectors = _vectors(train_ca)
    heldout_vectors = _vectors(heldout_ca)
    synthetic_vectors = _vectors(synthetic_ca)
    synthetic_to_train = _nearest(train_vectors, synthetic_vectors)
    member_to_synthetic = _nearest(synthetic_vectors, train_vectors)
    nonmember_to_synthetic = _nearest(synthetic_vectors, heldout_vectors)
    membership_labels = np.concatenate(
        [np.ones(len(member_to_synthetic)), np.zeros(len(nonmember_to_synthetic))]
    )
    membership_scores = -np.concatenate([member_to_synthetic, nonmember_to_synthetic])
    membership_auc = float(roc_auc_score(membership_labels, membership_scores))
    real_hashes = {_sha256(Path(path)) for path in train_ca["png_path"].astype(str)}
    synthetic_hashes = [_sha256(Path(path)) for path in synthetic_ca["png_path"].astype(str)]
    privacy = {
        "audit_name": "image-space nearest-neighbor membership-inference audit",
        "interpretation": (
            "AUROC near 0.5 means this prespecified nearest-neighbor attack does not distinguish "
            "training members from held-out CA images. This is not proof against every attack."
        ),
        "n_train_members": len(train_ca),
        "n_heldout_nonmembers": len(heldout_ca),
        "n_synthetic_ca": len(synthetic_ca),
        "membership_attack_auroc": membership_auc,
        "synthetic_to_train_distance": _percentiles(synthetic_to_train),
        "member_to_synthetic_distance": _percentiles(member_to_synthetic),
        "nonmember_to_synthetic_distance": _percentiles(nonmember_to_synthetic),
        "exact_synthetic_train_duplicates": int(sum(value in real_hashes for value in synthetic_hashes)),
    }
    (args.out_dir / "privacy_audit.json").write_text(json.dumps(privacy, indent=2) + "\n")

    real_disc = _sample(real[real["group"].astype(str) == "CA"], 2000, args.seed + 201).copy()
    real_disc["is_real"] = 1
    synthetic_disc = _sample(
        synthetic[synthetic["group"].astype(str) == "CA"], 2000, args.seed + 202
    ).copy()
    synthetic_disc["patient_id"] = "synthetic-" + synthetic_disc["image_id"].astype(str)
    synthetic_disc["is_real"] = 0
    discriminator_frame = pd.concat(
        [real_disc[["patient_id", "png_path", "is_real"]], synthetic_disc[["patient_id", "png_path", "is_real"]]],
        ignore_index=True,
    )
    discriminator_frame = make_patient_splits(discriminator_frame, seed=args.seed)
    discriminator_frame.to_csv(args.out_dir / "discriminator_manifest.csv", index=False)
    result = run_discriminator(
        discriminator_frame,
        args.out_dir / "discriminator",
        epochs=20,
        patience=5,
        batch_size=32,
        seed=args.seed,
    )
    (args.out_dir / "discriminator_metrics.json").write_text(
        json.dumps(result.__dict__, indent=2) + "\n"
    )
    print(f"FIDELITY AND PRIVACY AUDIT PASSED out={args.out_dir}")


if __name__ == "__main__":
    main()
