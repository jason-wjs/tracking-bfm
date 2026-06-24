#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
tracking_bfm_cd_repo_root

MODE="${MODE:-checkpoint}"

case "$MODE" in
  checkpoint)
    cmd=(uv run tracking-bfm-inspect-checkpoint)
    if [[ -n "${CHECKPOINT:-}" ]]; then
      cmd+=("$CHECKPOINT")
    fi
    tracking_bfm_add_bool_flag_if_true cmd "--json" JSON
    ;;
  latent)
    TASK="${TASK:-Mjlab-LatentDistillationBFM-Flat-Unitree-G1}"
    OUTPUT_DIR="${OUTPUT_DIR:-logs/diagnostics/latent-space}"
    cmd=(uv run tracking-bfm-analyze-latent-space "$TASK" --output-dir "$OUTPUT_DIR")
    tracking_bfm_add_arg_if_set cmd "--checkpoint-file" CHECKPOINT
    tracking_bfm_add_arg_if_set cmd "--motion-path" MOTION_PATH
    tracking_bfm_add_arg_if_set cmd "--num-envs" NUM_ENVS
    tracking_bfm_add_arg_if_set cmd "--num-points" NUM_POINTS
    tracking_bfm_add_arg_if_set cmd "--device" DEVICE
    tracking_bfm_add_arg_if_set cmd "--sampling-mode" SAMPLING_MODE
    tracking_bfm_add_arg_if_set cmd "--motion-history-steps" MOTION_HISTORY_STEPS
    tracking_bfm_add_arg_if_set cmd "--motion-future-steps" MOTION_FUTURE_STEPS
    tracking_bfm_add_arg_if_set cmd "--proprio-history-length" PROPRIO_HISTORY_LENGTH
    tracking_bfm_add_arg_if_set cmd "--sim-njmax" SIM_NJMAX
    tracking_bfm_add_arg_if_set cmd "--sim-nconmax" SIM_NCONMAX
    tracking_bfm_add_arg_if_set cmd "--max-plot-points" MAX_PLOT_POINTS
    tracking_bfm_add_bool_switch_if_set cmd "--deterministic" "--no-deterministic" DETERMINISTIC
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=checkpoint or MODE=latent." >&2
    exit 2
    ;;
esac

cmd+=("$@")
tracking_bfm_run_or_print "${cmd[@]}"
