"""Conditional U-Net, leakage, training, and deterministic sampling smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from echosynth.data.manifest import LeakageError, write_manifest
from echosynth.generative.dataset import LabeledEchoDataset
from echosynth.generative.sample import generate
from echosynth.generative.trainer import train_diffusion
from echosynth.generative.unet_cfg import ClassConditionalUnet
from echosynth.utils.checkpoint import load_checkpoint


def _row(index: int, path: Path, split: str = "train") -> dict[str, object]:
    return {
        "image_id": f"img-{index}",
        "patient_id": f"patient-{index}",
        "encounter_id": f"encounter-{index}",
        "group": "CA" if index % 2 == 0 else "GEN",
        "study_date": "2020-01-01",
        "dicom_path": f"/mounted/{index}.dcm",
        "png_path": str(path),
        "sop_class_uid": "1.2.3",
        "color_doppler": False,
        "view_pred": None,
        "view_conf": None,
        "qc_flag": False,
        "qc_features": json.dumps({}),
        "qc_excluded": False,
        "qc_reason": "",
        "split": split,
    }


def _tiny_manifest(tmp_path: Path, *, bad_split: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    for index in range(6):
        path = tmp_path / f"echo-{index}.png"
        array = np.zeros((16, 16), dtype=np.uint8)
        array[2:14, 3 + index % 2 : 13] = 40 + index * 20
        Image.fromarray(array).save(path)
        rows.append(_row(index, path, "test" if bad_split and index == 5 else "train"))
    manifest = tmp_path / "manifest.csv"
    write_manifest(pd.DataFrame(rows), manifest)
    return manifest


def test_unet_shape_and_cfg_direction() -> None:
    torch.manual_seed(4)
    model = ClassConditionalUnet(
        dim=4,
        dim_mults=(1, 2),
        num_classes=3,
        resnet_block_groups=4,
        attn_dim_head=4,
        attn_heads=1,
        full_attn=(False, False),
    )
    images = torch.randn(4, 1, 112, 112)
    times = torch.tensor([1, 2, 3, 4])
    classes = torch.tensor([0, 1, 2, 0])
    conditional = model(images, times, classes, cond_drop_prob=0)
    unconditional = model(images, times, classes, cond_drop_prob=1)
    scale_one = model.forward_with_cond_scale(images, times, classes, 1)
    scale_two = model.forward_with_cond_scale(images, times, classes, 2)
    assert conditional.shape == images.shape
    assert torch.allclose(scale_one, conditional)
    assert torch.allclose(scale_two - unconditional, 2 * (conditional - unconditional), atol=1e-5)


def test_generator_dataset_rejects_test_row(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path, bad_split=True)
    with pytest.raises(LeakageError, match="non-train"):
        LabeledEchoDataset(manifest, "train", "group", {"CA": 0, "GEN": 1}, 16)


@pytest.mark.slow
def test_twenty_step_train_and_reproducible_sample(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    checkpoint = train_diffusion(
        manifest,
        tmp_path / "ckpt",
        {"CA": 0, "GEN": 1},
        image_size=16,
        steps=20,
        batch_size=3,
        model_dim=4,
        dim_mults=(1, 2),
        full_attn=(False, False),
        timesteps=8,
        sampling_timesteps=2,
        save_and_sample_every=10,
    )
    state, metadata = load_checkpoint(checkpoint)
    assert "model" in state and metadata["steps"] == 20
    first = generate(checkpoint, {"CA": 1}, 1.5, 2, tmp_path / "samples-a", 9, tmp_path / "a.csv")
    second = generate(checkpoint, {"CA": 1}, 1.5, 2, tmp_path / "samples-b", 9, tmp_path / "b.csv")
    assert (
        Path(first.loc[0, "png_path"]).read_bytes() == Path(second.loc[0, "png_path"]).read_bytes()
    )
    assert first.loc[0, "split"] == "train"
