#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/cluster.env"
[[ -f "$ENV_FILE" ]] || ENV_FILE="$SCRIPT_DIR/cluster.env.example"
source "$ENV_FILE"
COMMAND=(sbatch --partition "$PARTITION" --account "$ACCOUNT" --mail-user "$EMAIL" --gres=gpu:1 --mem=80G --time=3-00:00:00 "$0")
if [[ "${1:-}" == "--dry-run" ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then exec "${COMMAND[@]}"; fi
cd "$PROJECT_ROOT"
exec "$PYTHON" scripts/05_train_diffusion.py --config "$DIFFUSION_CONFIG" --manifest "$TRAIN_MANIFEST" --out-dir results/runs/diffusion/ckpt
