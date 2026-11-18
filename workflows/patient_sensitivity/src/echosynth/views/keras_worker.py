"""TensorFlow-only subprocess worker; this module must run in the view environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from echosynth.views.view_classifier import INDEX_TO_LABEL, prepare_view_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    batch = np.stack([prepare_view_input(Path(path)) for path in frame["png_path"].astype(str)])
    predictions = np.asarray(load_model(args.weights).predict(batch, verbose=0))
    indices = np.argmax(predictions, axis=1)
    frame["view_pred"] = [INDEX_TO_LABEL[int(index)] for index in indices]
    frame["view_conf"] = predictions[np.arange(len(predictions)), indices]
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
