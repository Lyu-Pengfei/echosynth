"""Regenerate paper tables from run artifacts rather than result literals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _write_table(frame: pd.DataFrame, stem: Path) -> list[Path]:
    csv_path = stem.with_suffix(".csv")
    markdown_path = stem.with_suffix(".md")
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    return [csv_path, markdown_path]


def _configs(runs: list[Path]) -> dict[Path, dict[str, Any]]:
    output: dict[Path, dict[str, Any]] = {}
    for run in runs:
        path = run / "config.yaml"
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                output[run] = loaded
    return output


def _table_one(configs: dict[Path, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for config in configs.values():
        manifest_value = config.get("data", {}).get("manifest")
        if not manifest_value:
            continue
        manifest = Path(manifest_value)
        if not manifest.is_file() or manifest in seen:
            continue
        seen.add(manifest)
        frame = pd.read_csv(manifest)
        required = {"encounter_id", "group", "split"}
        if not required.issubset(frame):
            continue
        encounters = frame[list(required)].drop_duplicates()
        counts = encounters.groupby(["group", "split"]).size().rename("encounters").reset_index()
        counts.insert(0, "dataset", manifest.stem)
        records.extend(counts.to_dict(orient="records"))
    return pd.DataFrame(records, columns=["dataset", "group", "split", "encounters"])


def _table_two(runs: list[Path]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for run in runs:
        path = run / "survey.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame.insert(0, "run_id", run.name)
            tables.append(frame)
    return (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(
            columns=["run_id", "source", "item", "response", "count", "n", "percentage"]
        )
    )


def _table_three(runs: list[Path], configs: dict[Path, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for run in runs:
        path = run / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            continue
        task = configs.get(run, {}).get("name", run.name)
        for paradigm in ("TRTR", "TSTR", "BOOTSTRAP", "TRSTR"):
            values = metrics.get(paradigm)
            if not isinstance(values, dict) or "auroc" not in values:
                continue
            delta = values.get("delta_vs_trtr", {})
            records.append(
                {
                    "run_id": run.name,
                    "task": task,
                    "paradigm": paradigm,
                    "auroc": values["auroc"],
                    "auroc_ci_low": values.get("auroc_ci", [None, None])[0],
                    "auroc_ci_high": values.get("auroc_ci", [None, None])[1],
                    "auprc": values.get("auprc"),
                    "delta_vs_trtr": delta.get("delta"),
                    "delta_ci_low": delta.get("ci", [None, None])[0] if delta else None,
                    "delta_ci_high": delta.get("ci", [None, None])[1] if delta else None,
                    "selected_synthetic_count": metrics.get("selected_synthetic_count"),
                }
            )
    return pd.DataFrame(records)


def make_tables(runs_dir: Path, out_dir: Path) -> list[Path]:
    """Write Tables 1-3 as tidy CSV and Markdown from on-disk artifacts."""
    runs = sorted(path for path in runs_dir.iterdir() if path.is_dir()) if runs_dir.exists() else []
    if not runs:
        raise FileNotFoundError(f"no run directories found under {runs_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = _configs(runs)
    paths: list[Path] = []
    paths.extend(_write_table(_table_one(configs), out_dir / "table1"))
    paths.extend(_write_table(_table_two(runs), out_dir / "table2"))
    paths.extend(_write_table(_table_three(runs, configs), out_dir / "table3"))
    return paths
