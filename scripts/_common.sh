#!/usr/bin/env bash

tracking_bfm_cd_repo_root() {
  local script_path
  local script_dir
  script_path="${BASH_SOURCE[1]}"
  script_dir="$(cd "$(dirname "$script_path")" && pwd)"
  cd "$script_dir/.."
}

tracking_bfm_is_true() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off|"") return 1 ;;
    *)
      echo "Invalid boolean value: $value" >&2
      exit 2
      ;;
  esac
}

tracking_bfm_add_arg_if_set() {
  local -n __tracking_bfm_command_ref="$1"
  local flag="$2"
  local env_name="$3"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    __tracking_bfm_command_ref+=("$flag" "$value")
  fi
}

tracking_bfm_add_words_if_set() {
  local -n __tracking_bfm_command_ref="$1"
  local flag="$2"
  local env_name="$3"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    read -r -a values <<< "$value"
    __tracking_bfm_command_ref+=("$flag" "${values[@]}")
  fi
}

tracking_bfm_add_bool_flag_if_true() {
  local -n __tracking_bfm_command_ref="$1"
  local flag="$2"
  local env_name="$3"
  local value="${!env_name:-}"
  if tracking_bfm_is_true "$value"; then
    __tracking_bfm_command_ref+=("$flag")
  fi
}

tracking_bfm_add_bool_switch_if_set() {
  local -n __tracking_bfm_command_ref="$1"
  local positive_flag="$2"
  local negative_flag="$3"
  local env_name="$4"
  local value="${!env_name:-}"
  if [[ -z "$value" ]]; then
    return
  fi
  if tracking_bfm_is_true "$value"; then
    __tracking_bfm_command_ref+=("$positive_flag")
  else
    __tracking_bfm_command_ref+=("$negative_flag")
  fi
}

tracking_bfm_run_or_print() {
  printf 'Running command:'
  printf ' %q' "$@"
  printf '\n'
  if tracking_bfm_is_true "${DRY_RUN:-false}"; then
    return 0
  fi
  exec "$@"
}
