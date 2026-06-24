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

append_rollout_args() {
  append_arg --wandb-run-path "${WANDB_RUN_PATH:-}"
  append_arg --wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME:-}"
  append_arg --checkpoint-file "${CHECKPOINT_FILE:-}"
  append_arg --motion-path "${MOTION_PATH:-}"
  append_arg --motion-type "${MOTION_TYPE:-}"
  append_arg --history-steps "${HISTORY_STEPS:-}"
  append_arg --future-steps "${FUTURE_STEPS:-}"
  append_arg --num-envs "${NUM_ENVS:-}"
  append_arg --device "${DEVICE:-}"
  append_arg --torchrunx-log-dir "${TORCHRUNX_LOG_DIR:-}"

  if [[ -n "${GPU_IDS:-}" ]]; then
    read -r -a gpu_ids <<< "$GPU_IDS"
    cmd+=(--gpu-ids "${gpu_ids[@]}")
  fi
}

MODE="${MODE:-filter}"

case "$MODE" in
  filter)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-filter-motions "$TASK")
    append_rollout_args
    append_arg --failure-threshold "${FAILURE_THRESHOLD:-}"
    append_arg --output-file "${OUTPUT_FILE:-}"
    append_arg --viewer "${VIEWER:-}"
    ;;
  generate)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-generate-motion-dataset "$TASK")
    append_rollout_args
    append_arg --completion-threshold "${COMPLETION_THRESHOLD:-}"
    append_arg --output-motion-path "${OUTPUT_MOTION_PATH:-}"
    append_arg --output-file "${OUTPUT_FILE:-}"
    ;;
  delete)
    cmd=(uv run tracking-bfm-delete-failed-motions)
    append_arg --report-file "${REPORT_FILE:-}"
    if [[ "${EXECUTE:-0}" == "1" || "${EXECUTE:-}" == "true" ]]; then
      cmd+=(--execute)
    fi
    if [[ "${STRICT_MISSING:-0}" == "1" || "${STRICT_MISSING:-}" == "true" ]]; then
      cmd+=(--strict-missing)
    fi
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=filter, MODE=generate, or MODE=delete." >&2
    exit 2
    ;;
esac

exec "${cmd[@]}" "$@"
