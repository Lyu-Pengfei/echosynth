"""Run one fixed-seed v6 experiment with horizontal flipping disabled."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--follow-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from echosynth.classify import paradigms
    from echosynth.classify.transforms import EvalTransform, TrainTransform

    def build_no_flip_transform(training: bool, image_size: int = 112):
        if training:
            return TrainTransform(image_size=image_size, flip_probability=0.0)
        return EvalTransform(image_size=image_size)

    # paradigms imported build_transform into its own module namespace; patch that
    # exact reference without modifying the completed v6 source tree.
    paradigms.build_transform = build_no_flip_transform
    probe = paradigms.build_transform(True, 112)
    if not isinstance(probe, TrainTransform) or probe.flip_probability != 0.0:
        raise RuntimeError("no-flip transform patch did not take effect")
    if args.dry_run:
        print("NO-FLIP PATCH VERIFIED: flip_probability=0.0")
        return

    config_path = args.v6_root / "configs/experiment/ca_revision_prcc_full.yaml"
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["seed"] = args.seed
    values["name"] = "ca_revision_no_flip"
    values["run_id"] = f"ca_revision_no_flip_seed{args.seed}"
    values["results_dir"] = str(args.follow_root / "results/no_flip/runs")
    values["data"]["manifest"] = str(
        args.v6_root / "work/full_revision/manifests/ca_master.csv"
    )
    values["synthetic_manifest"] = str(
        args.v6_root / "work/full_revision/manifests/ca_synthetic.csv"
    )
    values["paradigms"] = ["TRTR", "TRSTR"]
    # The locked v6 manifests intentionally store image paths relative to the v6
    # project root. Running from the follow-up directory would incorrectly resolve
    # them below Uploaded_Codes/aml_reviewer_followup_v1.
    os.chdir(args.v6_root)
    result = paradigms.run_experiment(values)
    print(
        f"NO-FLIP COMPLETE seed={args.seed} selected={result.selected_synthetic_count} "
        f"dir={result.run_dir}"
    )


if __name__ == "__main__":
    main()
