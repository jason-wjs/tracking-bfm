#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

append_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    cmd+=("$flag" "$value")
  fi
}

TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
cmd=(uv run tracking-bfm-evaluate "$TASK")

append_arg --wandb-run-path "${WANDB_RUN_PATH:-}"
append_arg --wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME:-}"
append_arg --num-envs "${NUM_ENVS:-}"
append_arg --device "${DEVICE:-}"
append_arg --output-file "${OUTPUT_FILE:-}"

exec "${cmd[@]}" "$@"
