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

MODE="${MODE:-checkpoint}"

case "$MODE" in
  checkpoint)
    cmd=(uv run tracking-bfm-inspect-checkpoint)
    if [[ -n "${CHECKPOINT:-}" ]]; then
      cmd+=("$CHECKPOINT")
    fi
    if [[ "${JSON:-0}" == "1" || "${JSON:-}" == "true" ]]; then
      cmd+=(--json)
    fi
    ;;
  latent)
    TASK="${TASK:-Mjlab-LatentDistillationBFM-Flat-Unitree-G1}"
    OUTPUT_DIR="${OUTPUT_DIR:-logs/diagnostics/latent-space}"
    cmd=(uv run tracking-bfm-analyze-latent-space "$TASK" --output-dir "$OUTPUT_DIR")
    append_arg --checkpoint-file "${CHECKPOINT:-}"
    append_arg --motion-path "${MOTION_PATH:-}"
    append_arg --num-envs "${NUM_ENVS:-}"
    append_arg --num-points "${NUM_POINTS:-}"
    append_arg --device "${DEVICE:-}"
    append_arg --sampling-mode "${SAMPLING_MODE:-}"
    append_arg --motion-history-steps "${MOTION_HISTORY_STEPS:-}"
    append_arg --motion-future-steps "${MOTION_FUTURE_STEPS:-}"
    append_arg --proprio-history-length "${PROPRIO_HISTORY_LENGTH:-}"
    append_arg --sim-njmax "${SIM_NJMAX:-}"
    append_arg --sim-nconmax "${SIM_NCONMAX:-}"
    append_arg --max-plot-points "${MAX_PLOT_POINTS:-}"
    case "${DETERMINISTIC:-}" in
      1|true|yes|on) cmd+=(--deterministic) ;;
      0|false|no|off) cmd+=(--no-deterministic) ;;
      "") ;;
      *)
        echo "Invalid DETERMINISTIC=$DETERMINISTIC" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use MODE=checkpoint or MODE=latent." >&2
    exit 2
    ;;
esac

exec "${cmd[@]}" "$@"
