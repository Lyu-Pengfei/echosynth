"""ONNX and isolated-Keras view inference adapted from the MIT reference project."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from PIL import Image

from echosynth.data.manifest import read_manifest, write_manifest

LABEL_MAP = {
    "plax": 0,
    "psax-av": 1,
    "psax-mv": 2,
    "psax-ap": 3,
    "a4c": 4,
    "a5c": 5,
    "a3c": 6,
    "a2c": 7,
}
INDEX_TO_LABEL = {index: label for label, index in LABEL_MAP.items()}
WEIGHTS_URL = "https://github.com/raventan95/echo-view-classifier#how-to-run"


class MissingWeightsError(FileNotFoundError):
    """Raised with acquisition guidance when local view weights are absent."""


class PredictModel(Protocol):
    """Minimal prediction interface shared by tests and Keras-like models."""

    def predict(self, batch: np.ndarray) -> np.ndarray: ...


def prepare_view_input(path: Path) -> np.ndarray:
    """Upscale 112 grayscale to 224 RGB for view inference only."""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return np.repeat(array[..., None], 3, axis=2)


def _onnx_predict(weights: Path, batch: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    model_batch = batch
    if len(input_meta.shape) == 4 and input_meta.shape[1] == 3:
        model_batch = np.transpose(batch, (0, 3, 1, 2))
    predictions = session.run(None, {input_meta.name: model_batch.astype(np.float32)})[0]
    return cast(np.ndarray, predictions)


def _predict(model: Any, batch: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        result = cast(PredictModel, model).predict(batch)
    elif callable(model):
        result = cast(Callable[[np.ndarray], np.ndarray], model)(batch)
    else:
        raise TypeError("model must be callable or expose predict(batch)")
    predictions = np.asarray(result, dtype=float)
    if predictions.shape != (len(batch), len(LABEL_MAP)):
        raise ValueError(f"expected prediction shape {(len(batch), 8)}, got {predictions.shape}")
    return predictions


def _classify_in_process(frame: pd.DataFrame, weights: Path, model: Any) -> pd.DataFrame:
    batch = np.stack([prepare_view_input(Path(path)) for path in frame["png_path"].astype(str)])
    predictions = _onnx_predict(weights, batch) if model is None else _predict(model, batch)
    indices = np.argmax(predictions, axis=1)
    output = frame.copy()
    output["view_pred"] = [INDEX_TO_LABEL[int(index)] for index in indices]
    output["view_conf"] = predictions[np.arange(len(predictions)), indices]
    return output


def _keras_subprocess(manifest: Path, weights: Path) -> pd.DataFrame:
    interpreter = os.environ.get("ECHOSYNTH_VIEW_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="echosynth-view-") as directory:
        output = Path(directory) / "predictions.csv"
        command = [
            interpreter,
            "-m",
            "echosynth.views.keras_worker",
            "--manifest",
            str(manifest),
            "--weights",
            str(weights),
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True)
        return pd.read_csv(output)


def classify_views(
    manifest: Path, weights: Path, mode: str = "onnx", model: Any = None
) -> pd.DataFrame:
    """Predict the eight-view label/confidence and round-trip the master manifest."""
    if mode not in {"onnx", "keras"}:
        raise ValueError("mode must be 'onnx' or 'keras'")
    if model is None and not weights.is_file():
        raise MissingWeightsError(
            f"View-classifier weights not found: {weights}. Download: {WEIGHTS_URL}"
        )
    frame = read_manifest(manifest)
    result = (
        _keras_subprocess(manifest, weights)
        if mode == "keras" and model is None
        else _classify_in_process(frame, weights, model)
    )
    write_manifest(result, manifest)
    return read_manifest(manifest)
