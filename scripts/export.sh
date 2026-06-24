#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

MODE="${MODE:-tracking}"

case "$MODE" in
  tracking)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-export-onnx --task-id "$TASK")
    tracking_bfm_add_arg_if_set cmd "--checkpoint" CHECKPOINT
    tracking_bfm_add_arg_if_set cmd "--checkpoint-family" CHECKPOINT_FAMILY
    tracking_bfm_add_arg_if_set cmd "--obs-group" OBS_GROUP
    tracking_bfm_add_arg_if_set cmd "--motion-path" MOTION_PATH
    tracking_bfm_add_arg_if_set cmd "--motion-file" MOTION_FILE
    tracking_bfm_add_arg_if_set cmd "--student-history-steps" STUDENT_HISTORY_STEPS
    tracking_bfm_add_arg_if_set cmd "--student-future-steps" STUDENT_FUTURE_STEPS
    tracking_bfm_add_arg_if_set cmd "--student-robot-history-steps" STUDENT_ROBOT_HISTORY_STEPS
    ;;
  latent)
    TASK="${TASK:-Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage}"
    cmd=(uv run tracking-bfm-export-latent-onnx --task-id "$TASK")
    tracking_bfm_add_arg_if_set cmd "--checkpoint" CHECKPOINT
    tracking_bfm_add_arg_if_set cmd "--decoder-checkpoint" DECODER_CHECKPOINT
    tracking_bfm_add_arg_if_set cmd "--obs-group" OBS_GROUP
    tracking_bfm_add_arg_if_set cmd "--proprio-obs-group" PROPRIO_OBS_GROUP
    tracking_bfm_add_arg_if_set cmd "--motion-path" MOTION_PATH
    tracking_bfm_add_arg_if_set cmd "--motion-file" MOTION_FILE
    tracking_bfm_add_arg_if_set cmd "--latent-action-clip" LATENT_ACTION_CLIP
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=tracking or MODE=latent." >&2
    exit 2
    ;;
esac

tracking_bfm_add_arg_if_set cmd "--output-name" OUTPUT_NAME
tracking_bfm_add_arg_if_set cmd "--robot-name" ROBOT_NAME
tracking_bfm_add_arg_if_set cmd "--device" DEVICE
tracking_bfm_add_bool_flag_if_true cmd "--overwrite" OVERWRITE
tracking_bfm_add_bool_flag_if_true cmd "--verbose" VERBOSE

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
