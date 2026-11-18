"""Prespecified nearest-neighbour, diversity, duplicate, and membership audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def image_vector(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
        ).reshape(-1) / 255.0
    array = (array - array.mean()) / max(float(array.std()), 1e-6)
    norm = max(float(np.linalg.norm(array)), 1e-6)
    return array / norm


def matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.stack([image_vector(Path(path)) for path in frame["png_path"].astype(str)])


def resolve_png_paths(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Resolve v6 manifest paths against the v6 root, not the caller's directory."""
    output = frame.copy()

    def resolve(value: object) -> str:
        path = Path(str(value))
        return str(path if path.is_absolute() else project_root / path)

    output["png_path"] = output["png_path"].map(resolve)
    return output


def one_per_patient(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1.0, random_state=seed)
    return shuffled.drop_duplicates("patient_id").reset_index(drop=True)


def limited(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)
    return frame.sample(n=maximum, random_state=seed, replace=False).reset_index(drop=True)


def nearest(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1).fit(reference)
    distances, indices = model.kneighbors(query)
    return distances[:, 0], indices[:, 0]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    values = []
    for _ in range(replicates):
        indices = np.concatenate(
            [generator.choice(positives, len(positives), replace=True), generator.choice(negatives, len(negatives), replace=True)]
        )
        values.append(roc_auc_score(labels[indices], scores[indices]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def random_pair_distances(values: np.ndarray, pairs: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    first = generator.integers(0, len(values), pairs)
    second = generator.integers(0, len(values), pairs)
    unequal = first != second
    return np.linalg.norm(values[first[unequal]] - values[second[unequal]], axis=1)


def contact_sheet(synthetic: pd.DataFrame, real: pd.DataFrame, synthetic_indices: np.ndarray, real_indices: np.ndarray, output: Path) -> None:
    count = min(8, len(synthetic_indices))
    figure, axes = plt.subplots(count, 2, figsize=(5, 2.2 * count), constrained_layout=True)
    if count == 1:
        axes = np.asarray([axes])
    for row in range(count):
        for column, frame, index, title in (
            (0, synthetic, int(synthetic_indices[row]), "Synthetic"),
            (1, real, int(real_indices[row]), "Nearest real training image"),
        ):
            with Image.open(frame.iloc[index]["png_path"]) as image:
                axes[row, column].imshow(image.convert("L"), cmap="gray")
            axes[row, column].set_title(f"{title}, pair {row + 1}")
            axes[row, column].axis("off")
    figure.savefig(output, dpi=250)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    real = resolve_png_paths(
        pd.read_csv(args.v6_root / "work/full_revision/manifests/ca_master.csv"), args.v6_root
    )
    synthetic = resolve_png_paths(
        pd.read_csv(args.v6_root / "work/full_revision/manifests/ca_synthetic.csv"), args.v6_root
    )
    if "group" not in synthetic:
        synthetic["group"] = synthetic["class_label"]
    train = real[(real["group"].astype(str) == "CA") & (real["split"].astype(str) == "train")].copy()
    test = real[(real["group"].astype(str) == "CA") & (real["split"].astype(str) == "test")].copy()
    generated = synthetic[synthetic["group"].astype(str) == "CA"].copy()
    member = limited(one_per_patient(train, 8101), args.max_images, 8102)
    nonmember = limited(one_per_patient(test, 8103), args.max_images, 8104)
    reference = limited(train, args.max_images, 8105)
    generated = limited(generated, args.max_images, 8106)
    if min(len(member), len(nonmember), len(reference), len(generated)) < 10:
        raise RuntimeError("privacy audit has fewer than 10 images in one required set")
    member_values = matrix(member)
    nonmember_values = matrix(nonmember)
    reference_values = matrix(reference)
    synthetic_values = matrix(generated)
    member_distances, _ = nearest(synthetic_values, member_values)
    nonmember_distances, _ = nearest(synthetic_values, nonmember_values)
    labels = np.concatenate([np.ones(len(member_distances), dtype=int), np.zeros(len(nonmember_distances), dtype=int)])
    scores = -np.concatenate([member_distances, nonmember_distances])
    attack_auc = float(roc_auc_score(labels, scores))
    attack_low, attack_high = bootstrap_auc(labels, scores, 8107, args.bootstrap)
    synthetic_to_train, nearest_train_index = nearest(reference_values, synthetic_values)
    order = np.argsort(synthetic_to_train)[:8]
    contact_sheet(generated, reference, order, nearest_train_index[order], args.out_dir / "nearest_neighbor_examples.png")
    train_hashes = {hash_file(Path(path)) for path in reference["png_path"].astype(str)}
    synthetic_hashes = [hash_file(Path(path)) for path in generated["png_path"].astype(str)]
    synthetic_diversity = random_pair_distances(synthetic_values, 5000, 8108)
    real_diversity = random_pair_distances(reference_values, 5000, 8109)
    summary = {
        "status": "PASS",
        "feature_representation": "32x32 grayscale bilinear pixels; per-image z-score; L2-normalized flattened vector",
        "distance": "Euclidean",
        "n_training_member_patients": len(member),
        "n_heldout_nonmember_patients": len(nonmember),
        "n_real_training_reference_images": len(reference),
        "n_synthetic_ca_images": len(generated),
        "membership_attack_auroc": attack_auc,
        "membership_attack_95_ci": [attack_low, attack_high],
        "synthetic_to_train_distance": {
            "minimum": float(synthetic_to_train.min()),
            "p01": float(np.quantile(synthetic_to_train, 0.01)),
            "p05": float(np.quantile(synthetic_to_train, 0.05)),
            "median": float(np.median(synthetic_to_train)),
        },
        "exact_synthetic_train_duplicates_in_audited_sets": int(sum(value in train_hashes for value in synthetic_hashes)),
        "synthetic_pairwise_distance_mean": float(synthetic_diversity.mean()),
        "synthetic_pairwise_distance_sd": float(synthetic_diversity.std(ddof=1)),
        "real_train_pairwise_distance_mean": float(real_diversity.mean()),
        "real_train_pairwise_distance_sd": float(real_diversity.std(ddof=1)),
        "interpretation": "This evaluates one prespecified image-space nearest-neighbour attack. A result near 0.5 does not guarantee privacy against other attacks.",
    }
    (args.out_dir / "privacy_nearest_neighbor.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({"synthetic_to_train_distance": synthetic_to_train}).to_csv(
        args.out_dir / "synthetic_to_train_distances.csv", index=False
    )
    print(f"PRIVACY/NEAREST-NEIGHBOR AUDIT COMPLETE: {args.out_dir}")


if __name__ == "__main__":
    main()
