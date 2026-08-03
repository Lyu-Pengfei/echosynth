#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/cluster.env"; [[ -f "$ENV_FILE" ]] || ENV_FILE="$SCRIPT_DIR/cluster.env.example"; source "$ENV_FILE"
COMMAND=(sbatch --partition "$PARTITION" --account "$ACCOUNT" --gres=gpu:1 --mem=32G --time=08:00:00 --array=0-29 "$0")
if [[ "${1:-}" == "--dry-run" ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then exec "${COMMAND[@]}"; fi
COUNT=$((100 * (${SLURM_ARRAY_TASK_ID:?} + 1)))
cd "$PROJECT_ROOT"
exec "$PYTHON" scripts/08_run_experiment.py --config "$EXPERIMENT_CONFIG" --set "synthetic_grid.start=$COUNT" --set "synthetic_grid.stop=$COUNT"
