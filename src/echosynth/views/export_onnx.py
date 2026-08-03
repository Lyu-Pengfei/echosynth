"""One-time Keras-to-ONNX view-classifier conversion for the separate view environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_onnx(weights: Path, output: Path) -> None:
    """Convert local Keras weights to ONNX without importing PyTorch."""
    if not weights.is_file():
        raise FileNotFoundError(weights)
    import tensorflow as tf
    import tf2onnx

    model: Any = tf.keras.models.load_model(weights)
    signature = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(model, input_signature=signature, output_path=str(output))
