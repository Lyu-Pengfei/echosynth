"""Prepare the complete PRCC CA/LVH/general-control DICOM cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from echosynth.data.ca_dicom_full import prepare_full_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config_prcc_full.yaml"))
    parser.add_argument("--out-root", type=Path, default=Path("work/full_revision"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = prepare_full_cohort(config, args.out_root)
    print(
        "FULL DICOM PREPARATION PASSED "
        f"images={result.accepted_images} patients={result.accepted_patients} "
        f"encounters={result.accepted_encounters} manifest={result.manifest}"
    )


if __name__ == "__main__":
    main()
