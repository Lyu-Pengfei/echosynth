"""Procedural CPU smoke: split, diffuse, sample, classify, evaluate, and report."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from echosynth.classify.paradigms import run_experiment
from echosynth.data.manifest import write_manifest
from echosynth.data.splits import make_patient_splits
from echosynth.generative.sample import generate
from echosynth.generative.trainer import train_diffusion
from echosynth.report.figures import make_figures
from echosynth.report.tables import make_tables


def _wedge(group: str, variant: int) -> np.ndarray:
    size = 112
    y, x = np.ogrid[:size, :size]
    center_x, center_y = 56 + variant % 3 - 1, 8
    radius = np.hypot(x - center_x, y - center_y)
    angle = np.arctan2(y - center_y, x - center_x)
    delta = np.angle(np.exp(1j * (angle - math.pi / 2)))
    support = (radius < 100) & (np.abs(delta) < math.radians(32))
    image = np.zeros((size, size), dtype=np.uint8)
    base = {"CA": 190, "LVH": 125, "GEN": 70}[group]
    texture = (12 * np.sin((x + variant) / 5) + 8 * np.cos(y / 7))[support]
    image[support] = np.clip(base + texture, 1, 255).astype(np.uint8)
    return image


def _real_manifest(root: Path) -> Path:
    image_dir = root / "real"
    image_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    groups = ["CA"] * 7 + ["LVH"] * 7 + ["GEN"] * 6
    for encounter_index, group in enumerate(groups):
        for image_index in range(3):
            image_id = f"real-{encounter_index}-{image_index}"
            path = image_dir / f"{image_id}.png"
            Image.fromarray(_wedge(group, image_index)).save(path)
            rows.append(
                {
                    "image_id": image_id,
                    "patient_id": f"patient-{encounter_index}",
                    "encounter_id": f"encounter-{encounter_index}",
                    "group": group,
                    "study_date": "2020-01-01",
                    "dicom_path": f"/mounted/{image_id}.dcm",
                    "png_path": str(path),
                    "sop_class_uid": "1.2.3",
                    "color_doppler": False,
                    "view_pred": None,
                    "view_conf": None,
                    "qc_flag": False,
                    "qc_features": json.dumps({}),
                    "qc_excluded": False,
                    "qc_reason": "",
                }
            )
    split = make_patient_splits(pd.DataFrame(rows), seed=2024)
    manifest = root / "master.csv"
    write_manifest(split, manifest)
    return manifest


@pytest.mark.slow
def test_end_to_end_cpu_smoke(tmp_path: Path) -> None:
    started = time.monotonic()
    manifest = _real_manifest(tmp_path)
    master = pd.read_csv(manifest)
    train_manifest = tmp_path / "train.csv"
    write_manifest(master.loc[master["split"] == "train"].copy(), train_manifest)
    checkpoint = train_diffusion(
        train_manifest,
        tmp_path / "diffusion",
        {"CA": 0, "LVH": 1, "GEN": 2},
        image_size=112,
        steps=200,
        batch_size=6,
        model_dim=2,
        dim_mults=(1,),
        full_attn=(False,),
        timesteps=8,
        sampling_timesteps=2,
        save_and_sample_every=100,
    )
    synthetic_manifest = tmp_path / "synthetic.csv"
    synthetic = generate(
        checkpoint,
        {"CA": 10, "LVH": 10, "GEN": 10},
        1.5,
        2,
        tmp_path / "synthetic",
        2024,
        synthetic_manifest,
    )
    assert len(synthetic) == 30
    results_dir = tmp_path / "runs"
    config = {
        "name": "cpu_smoke",
        "run_id": "smoke",
        "seed": 2024,
        "results_dir": str(results_dir),
        "data": {"manifest": str(manifest), "image_size": 112},
        "synthetic_manifest": str(synthetic_manifest),
        "task": {"type": "two_step", "compose": "product"},
        "classifier": {
            "channels": [2, 4, 8, 16],
            "epochs": 3,
            "patience": 3,
            "batch_size": 8,
            "lr": 0.01,
            "dropout": 0.0,
        },
        "paradigms": ["TRTR", "TSTR", "TRSTR", "BOOTSTRAP"],
        "synthetic_grid": {"start": 2, "stop": 4, "step": 2},
    }
    experiment = run_experiment(config)
    assert experiment.selected_synthetic_count in {2, 4}
    assert (experiment.run_dir / "metrics.json").is_file()
    assert len((experiment.run_dir / "test_access.jsonl").read_text().splitlines()) == 1
    real_features = []
    for path in master["png_path"].iloc[:10]:
        pixels = np.asarray(Image.open(path).convert("L"))
        real_features.append(np.histogram(pixels, bins=16, range=(0, 256))[0])
    synthetic_features = []
    for path in synthetic["png_path"].iloc[:10]:
        pixels = np.asarray(Image.open(path).convert("L"))
        synthetic_features.append(np.histogram(pixels, bins=16, range=(0, 256))[0])
    np.savez(
        experiment.run_dir / "embeddings.npz",
        features=np.asarray(real_features + synthetic_features),
        source=np.asarray(["real"] * 10 + ["synthetic"] * 10),
        classes=np.asarray(["mixed"] * 20),
    )
    report_dir = tmp_path / "report"
    tables = make_tables(results_dir, report_dir)
    figures = make_figures(results_dir, report_dir)
    assert len(tables) == 6
    assert {path.name for path in figures} == {"figure3.png", "figure4.png"}
    metrics = json.loads((experiment.run_dir / "metrics.json").read_text())
    assert all("auroc" in metrics[arm] for arm in config["paradigms"])
    assert time.monotonic() - started < 300
