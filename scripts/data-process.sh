#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

MODE="${MODE:-filter}"

tracking_bfm_add_rollout_args() {
  local target="$1"
  tracking_bfm_add_arg_if_set "$target" "--wandb-run-path" WANDB_RUN_PATH
  tracking_bfm_add_arg_if_set "$target" "--wandb-checkpoint-name" WANDB_CHECKPOINT_NAME
  tracking_bfm_add_arg_if_set "$target" "--checkpoint-file" CHECKPOINT_FILE
  tracking_bfm_add_arg_if_set "$target" "--motion-path" MOTION_PATH
  tracking_bfm_add_arg_if_set "$target" "--motion-type" MOTION_TYPE
  tracking_bfm_add_arg_if_set "$target" "--history-steps" HISTORY_STEPS
  tracking_bfm_add_arg_if_set "$target" "--future-steps" FUTURE_STEPS
  tracking_bfm_add_arg_if_set "$target" "--num-envs" NUM_ENVS
  tracking_bfm_add_arg_if_set "$target" "--device" DEVICE
  tracking_bfm_add_arg_if_set "$target" "--torchrunx-log-dir" TORCHRUNX_LOG_DIR
  tracking_bfm_add_words_if_set "$target" "--gpu-ids" GPU_IDS
}

case "$MODE" in
  filter)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-filter-motions "$TASK")
    tracking_bfm_add_rollout_args cmd
    tracking_bfm_add_arg_if_set cmd "--failure-threshold" FAILURE_THRESHOLD
    tracking_bfm_add_arg_if_set cmd "--output-file" OUTPUT_FILE
    tracking_bfm_add_arg_if_set cmd "--viewer" VIEWER
    ;;
  generate)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-generate-motion-dataset "$TASK")
    tracking_bfm_add_rollout_args cmd
    tracking_bfm_add_arg_if_set cmd "--completion-threshold" COMPLETION_THRESHOLD
    tracking_bfm_add_arg_if_set cmd "--output-motion-path" OUTPUT_MOTION_PATH
    tracking_bfm_add_arg_if_set cmd "--output-file" OUTPUT_FILE
    ;;
  delete)
    cmd=(uv run tracking-bfm-delete-failed-motions)
    tracking_bfm_add_arg_if_set cmd "--report-file" REPORT_FILE
    tracking_bfm_add_bool_flag_if_true cmd "--execute" EXECUTE
    tracking_bfm_add_bool_flag_if_true cmd "--strict-missing" STRICT_MISSING
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=filter, MODE=generate, or MODE=delete." >&2
    exit 2
    ;;
esac

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
