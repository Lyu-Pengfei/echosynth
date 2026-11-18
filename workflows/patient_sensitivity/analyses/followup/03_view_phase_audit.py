"""Audit whether standardized view and cardiac-phase analyses are defensible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def completeness(frame: pd.DataFrame, column: str) -> tuple[int, float]:
    if column not in frame:
        return 0, 0.0
    values = frame[column].fillna("").astype(str).str.strip()
    valid = ~values.str.lower().isin({"", "nan", "none", "unknown", "unavailable"})
    return int(valid.sum()), float(valid.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--required-completeness", type=float, default=0.80)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.v6_root / "work/full_revision/manifests/ca_master.csv")
    columns = [
        "view_code", "view_pred", "view_conf", "cardiac_phase",
        "trigger_time_available", "frame_time_available", "series_description", "protocol_name",
    ]
    rows: list[dict[str, object]] = []
    for column in columns:
        available, fraction = completeness(manifest, column)
        rows.append(
            {"field": column, "available_images": available, "total_images": len(manifest), "completeness": fraction}
        )
        if column in manifest:
            manifest[column].fillna("MISSING").astype(str).value_counts().head(25).rename_axis("value").reset_index(name="n_images").assign(field=column).to_csv(
                args.out_dir / f"distribution_{column}.csv", index=False
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(args.out_dir / "metadata_completeness.csv", index=False)
    view_fraction = float(audit.loc[audit["field"].eq("view_code"), "completeness"].iloc[0])
    phase_fraction = float(audit.loc[audit["field"].eq("cardiac_phase"), "completeness"].iloc[0])
    status = {
        "status": "PASS",
        "required_completeness": args.required_completeness,
        "standardized_view_analysis": "FEASIBLE" if view_fraction >= args.required_completeness else "NOT_FEASIBLE",
        "standardized_phase_analysis": "FEASIBLE" if phase_fraction >= args.required_completeness else "NOT_FEASIBLE",
        "view_completeness": view_fraction,
        "phase_completeness": phase_fraction,
        "interpretation": (
            "No view- or phase-conditioned performance claim should be made unless validated standardized labels meet the prespecified completeness threshold. "
            "SeriesDescription, timing tags, or model-predicted views are descriptive metadata and are not substituted for clinical labels."
        ),
    }
    (args.out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    (args.out_dir / "MANUSCRIPT_NOTE.md").write_text(
        "# View and cardiac-phase feasibility\n\n"
        f"Validated standardized view-label completeness was {view_fraction:.1%}, and standardized cardiac-phase completeness was {phase_fraction:.1%}. "
        "The prespecified threshold for subgroup analysis was 80%. Analyses below that threshold were not performed because inferring labels from free text or timing tags would introduce unvalidated label error.\n"
    )
    print(f"VIEW/PHASE AUDIT COMPLETE: {args.out_dir}")


if __name__ == "__main__":
    main()
