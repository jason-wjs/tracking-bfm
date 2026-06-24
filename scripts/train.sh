#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
cmd=(uv run tracking-bfm-train "$TASK")

tracking_bfm_add_arg_if_set cmd "--registry-name" REGISTRY_NAME
tracking_bfm_add_arg_if_set cmd "--env.commands.motion.motion-file" MOTION_FILE
tracking_bfm_add_arg_if_set cmd "--env.commands.motion.motion-path" MOTION_PATH
tracking_bfm_add_arg_if_set cmd "--env.scene.num-envs" NUM_ENVS
tracking_bfm_add_arg_if_set cmd "--agent.max-iterations" MAX_ITERATIONS
tracking_bfm_add_arg_if_set cmd "--agent.experiment-name" EXPERIMENT_NAME
tracking_bfm_add_arg_if_set cmd "--agent.run-name" RUN_NAME
tracking_bfm_add_arg_if_set cmd "--agent.wandb-project" WANDB_PROJECT
tracking_bfm_add_arg_if_set cmd "--wandb-run-path" WANDB_RUN_PATH
tracking_bfm_add_arg_if_set cmd "--wandb-checkpoint-name" WANDB_CHECKPOINT_NAME
tracking_bfm_add_arg_if_set cmd "--torchrunx-log-dir" TORCHRUNX_LOG_DIR
tracking_bfm_add_words_if_set cmd "--gpu-ids" GPU_IDS
tracking_bfm_add_bool_flag_if_true cmd "--debug" DEBUG
tracking_bfm_add_bool_flag_if_true cmd "--video" VIDEO
tracking_bfm_add_bool_flag_if_true cmd "--enable-nan-guard" ENABLE_NAN_GUARD

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
