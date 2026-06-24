#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
cmd=(uv run tracking-bfm-play "$TASK")

tracking_bfm_add_arg_if_set cmd "--agent" AGENT
tracking_bfm_add_arg_if_set cmd "--registry-name" REGISTRY_NAME
tracking_bfm_add_arg_if_set cmd "--wandb-run-path" WANDB_RUN_PATH
tracking_bfm_add_arg_if_set cmd "--wandb-checkpoint-name" WANDB_CHECKPOINT_NAME
tracking_bfm_add_arg_if_set cmd "--checkpoint-file" CHECKPOINT_FILE
tracking_bfm_add_arg_if_set cmd "--motion-file" MOTION_FILE
tracking_bfm_add_arg_if_set cmd "--motion-type" MOTION_TYPE
tracking_bfm_add_arg_if_set cmd "--num-envs" NUM_ENVS
tracking_bfm_add_arg_if_set cmd "--device" DEVICE
tracking_bfm_add_arg_if_set cmd "--viewer" VIEWER
tracking_bfm_add_arg_if_set cmd "--video-length" VIDEO_LENGTH
tracking_bfm_add_arg_if_set cmd "--video-height" VIDEO_HEIGHT
tracking_bfm_add_arg_if_set cmd "--video-width" VIDEO_WIDTH
tracking_bfm_add_arg_if_set cmd "--camera" CAMERA
tracking_bfm_add_bool_flag_if_true cmd "--stochastic-policy" STOCHASTIC_POLICY
tracking_bfm_add_bool_flag_if_true cmd "--video" VIDEO
tracking_bfm_add_bool_flag_if_true cmd "--no-terminations" NO_TERMINATIONS
tracking_bfm_add_bool_switch_if_set \
  cmd "--show-reference-motion" "--no-show-reference-motion" SHOW_REFERENCE_MOTION

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
