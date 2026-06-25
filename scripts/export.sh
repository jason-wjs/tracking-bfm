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

MODE="${MODE:-policy}"

case "$MODE" in
  policy)
    TASK="${TASK:-Mjlab-TrackingBFM-Flat-Unitree-G1}"
    cmd=(uv run tracking-bfm-export policy --task-id "$TASK")
    append_arg --checkpoint "${CHECKPOINT:-}"
    append_arg --checkpoint-family "${CHECKPOINT_FAMILY:-}"
    append_arg --obs-group "${OBS_GROUP:-}"
    append_arg --motion-path "${MOTION_PATH:-}"
    append_arg --motion-file "${MOTION_FILE:-}"
    append_arg --student-history-steps "${STUDENT_HISTORY_STEPS:-}"
    append_arg --student-future-steps "${STUDENT_FUTURE_STEPS:-}"
    append_arg --student-robot-history-steps "${STUDENT_ROBOT_HISTORY_STEPS:-}"
    ;;
  latent)
    TASK="${TASK:-Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage}"
    cmd=(uv run tracking-bfm-export latent --task-id "$TASK")
    append_arg --checkpoint "${CHECKPOINT:-}"
    append_arg --decoder-checkpoint "${DECODER_CHECKPOINT:-}"
    append_arg --obs-group "${OBS_GROUP:-}"
    append_arg --proprio-obs-group "${PROPRIO_OBS_GROUP:-}"
    append_arg --motion-path "${MOTION_PATH:-}"
    append_arg --motion-file "${MOTION_FILE:-}"
    append_arg --latent-action-clip "${LATENT_ACTION_CLIP:-}"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=policy or MODE=latent." >&2
    exit 2
    ;;
esac

append_arg --output-name "${OUTPUT_NAME:-}"
append_arg --robot-name "${ROBOT_NAME:-}"
append_arg --device "${DEVICE:-}"

if [[ "${OVERWRITE:-0}" == "1" || "${OVERWRITE:-}" == "true" ]]; then
  cmd+=(--overwrite)
fi

if [[ "${VERBOSE:-0}" == "1" || "${VERBOSE:-}" == "true" ]]; then
  cmd+=(--verbose)
fi

exec "${cmd[@]}" "$@"
