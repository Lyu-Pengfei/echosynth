"""Lightweight local end-to-end smoke for the formal PRCC workflow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from echosynth.classify.paradigms import run_experiment
from echosynth.data.ca_dicom_full import prepare_full_cohort
from echosynth.generative.sample import generate
from echosynth.generative.trainer import train_diffusion


def _dicom(path: Path, patient: str, date: str, value: int) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = patient
    dataset.StudyDate = date
    dataset.Manufacturer = "SmokeVendor"
    dataset.ManufacturerModelName = "SmokeModel"
    dataset.Rows = 112
    dataset.Columns = 112
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    generator = np.random.default_rng(value)
    pixels = np.clip(generator.normal(value, 12, size=(112, 112)), 0, 255).astype(np.uint8)
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation/local_formal_smoke.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aml-formal-smoke-") as temporary:
        root = Path(temporary)
        groups: dict[str, str] = {}
        for group_index, (group, intensity) in enumerate(
            (("CA", 190), ("LVH", 125), ("GEN", 65))
        ):
            source = root / group
            source.mkdir()
            groups[group] = str(source)
            for patient_index in range(14):
                _dicom(
                    source / f"{group}_{patient_index}.dcm",
                    f"{group}-patient-{patient_index}",
                    f"202{group_index}0101",
                    intensity + patient_index,
                )
        prepared = prepare_full_cohort(
            {
                "seed": 2024,
                "workers": 1,
                "privacy": {"hash_salt": "local-formal-smoke-fixed-salt"},
                "data": {
                    "groups": groups,
                    "extensions": [".dcm"],
                    "standard_sop_class_uids": ["1.2.840.10008.5.1.4.1.1.6.1"],
                    "train_fraction": 0.70,
                    "validation_fraction": 0.15,
                },
            },
            root / "work",
        )
        checkpoint = train_diffusion(
            prepared.train_manifest,
            root / "diffusion",
            {"CA": 0, "LVH": 1, "GEN": 2},
            image_size=112,
            steps=2,
            batch_size=6,
            model_dim=2,
            dim_mults=(1,),
            full_attn=(False,),
            timesteps=4,
            sampling_timesteps=1,
            save_and_sample_every=2,
        )
        synthetic_manifest = root / "synthetic.csv"
        generate(
            checkpoint,
            {"CA": 4, "LVH": 4, "GEN": 4},
            1.5,
            1,
            root / "synthetic",
            2024,
            synthetic_manifest,
            batch_size=4,
        )
        experiment = run_experiment(
            {
                "seed": 2024,
                "name": "local_formal_smoke",
                "run_id": "local_formal_smoke",
                "results_dir": str(root / "results"),
                "data": {"manifest": str(prepared.manifest), "image_size": 112},
                "synthetic_manifest": str(synthetic_manifest),
                "task": {
                    "type": "two_step",
                    "compose": "product",
                    "positive_label": "CA",
                    "intermediate_label": "LVH",
                },
                "classifier": {
                    "channels": [2, 4, 8, 16],
                    "epochs": 1,
                    "patience": 1,
                    "batch_size": 8,
                    "num_workers": 0,
                    "lr": 0.01,
                    "dropout": 0.0,
                },
                "paradigms": [
                    "TRTR",
                    "TRSTR",
                    "BOOTSTRAP",
                    "CLASS_WEIGHTED",
                    "BALANCED_SAMPLING",
                    "TSTR",
                ],
                "synthetic_grid": {"start": 2, "stop": 2, "step": 1},
            }
        )
        payload = {
            "status": "PASS",
            "accepted_images": prepared.accepted_images,
            "accepted_patients": prepared.accepted_patients,
            "accepted_encounters": prepared.accepted_encounters,
            "selected_synthetic_count": experiment.selected_synthetic_count,
            "paradigms": sorted(
                key for key, value in experiment.metrics.items() if isinstance(value, dict)
            ),
            "patient_leakage": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
