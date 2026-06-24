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
cmd=(uv run tracking-bfm-train "$TASK")

append_arg --registry-name "${REGISTRY_NAME:-}"
append_arg --env.commands.motion.motion-file "${MOTION_FILE:-}"
append_arg --env.commands.motion.motion-path "${MOTION_PATH:-}"
append_arg --env.scene.num-envs "${NUM_ENVS:-}"
append_arg --agent.max-iterations "${MAX_ITERATIONS:-}"
append_arg --agent.experiment-name "${EXPERIMENT_NAME:-}"
append_arg --agent.run-name "${RUN_NAME:-}"
append_arg --agent.wandb-project "${WANDB_PROJECT:-}"
append_arg --wandb-run-path "${WANDB_RUN_PATH:-}"
append_arg --wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME:-}"
append_arg --torchrunx-log-dir "${TORCHRUNX_LOG_DIR:-}"

if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a gpu_ids <<< "$GPU_IDS"
  cmd+=(--gpu-ids "${gpu_ids[@]}")
fi

if [[ "${DEBUG:-0}" == "1" || "${DEBUG:-}" == "true" ]]; then
  cmd+=(--debug)
fi

if [[ "${VIDEO:-0}" == "1" || "${VIDEO:-}" == "true" ]]; then
  cmd+=(--video)
fi

if [[ "${ENABLE_NAN_GUARD:-0}" == "1" || "${ENABLE_NAN_GUARD:-}" == "true" ]]; then
  cmd+=(--enable-nan-guard)
fi

exec "${cmd[@]}" "$@"
