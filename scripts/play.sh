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
cmd=(uv run tracking-bfm-play "$TASK")

append_arg --agent "${AGENT:-}"
append_arg --registry-name "${REGISTRY_NAME:-}"
append_arg --wandb-run-path "${WANDB_RUN_PATH:-}"
append_arg --wandb-checkpoint-name "${WANDB_CHECKPOINT_NAME:-}"
append_arg --checkpoint-file "${CHECKPOINT_FILE:-}"
append_arg --motion-file "${MOTION_FILE:-}"
append_arg --motion-type "${MOTION_TYPE:-}"
append_arg --num-envs "${NUM_ENVS:-}"
append_arg --device "${DEVICE:-}"
append_arg --viewer "${VIEWER:-}"
append_arg --video-length "${VIDEO_LENGTH:-}"
append_arg --video-height "${VIDEO_HEIGHT:-}"
append_arg --video-width "${VIDEO_WIDTH:-}"
append_arg --camera "${CAMERA:-}"

if [[ "${STOCHASTIC_POLICY:-0}" == "1" || "${STOCHASTIC_POLICY:-}" == "true" ]]; then
  cmd+=(--stochastic-policy)
fi

if [[ "${VIDEO:-0}" == "1" || "${VIDEO:-}" == "true" ]]; then
  cmd+=(--video)
fi

if [[ "${NO_TERMINATIONS:-0}" == "1" || "${NO_TERMINATIONS:-}" == "true" ]]; then
  cmd+=(--no-terminations)
fi

case "${SHOW_REFERENCE_MOTION:-}" in
  1|true|yes|on) cmd+=(--show-reference-motion) ;;
  0|false|no|off) cmd+=(--no-show-reference-motion) ;;
  "") ;;
  *)
    echo "Invalid SHOW_REFERENCE_MOTION=$SHOW_REFERENCE_MOTION" >&2
    exit 2
    ;;
esac

exec "${cmd[@]}" "$@"
