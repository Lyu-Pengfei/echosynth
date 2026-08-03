#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/cluster.env"; [[ -f "$ENV_FILE" ]] || ENV_FILE="$SCRIPT_DIR/cluster.env.example"; source "$ENV_FILE"
COMMAND=(sbatch --partition "$CPU_PARTITION" --account "$ACCOUNT" --mem=16G --time=08:00:00 --array=0-99 "$0")
if [[ "${1:-}" == "--dry-run" ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then exec "${COMMAND[@]}"; fi
SHARD="${DICOM_SHARD_ROOT}/shard_${SLURM_ARRAY_TASK_ID:?}"
cd "$PROJECT_ROOT"
exec "$PYTHON" scripts/00_ingest_dicom.py --root "$SHARD" --out-dir "$PREPROCESS_OUTPUT" --manifest "manifests/shard_${SLURM_ARRAY_TASK_ID}.csv" --config configs/data/duke.yaml
