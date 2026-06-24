#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
cmd=(uv run tracking-bfm-evaluate "$TASK")

tracking_bfm_add_arg_if_set cmd "--wandb-run-path" WANDB_RUN_PATH
tracking_bfm_add_arg_if_set cmd "--wandb-checkpoint-name" WANDB_CHECKPOINT_NAME
tracking_bfm_add_arg_if_set cmd "--num-envs" NUM_ENVS
tracking_bfm_add_arg_if_set cmd "--device" DEVICE
tracking_bfm_add_arg_if_set cmd "--output-file" OUTPUT_FILE

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
