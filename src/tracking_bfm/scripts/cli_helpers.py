"""Shared helpers for tracking-bfm CLI entrypoints."""

import sys


def maybe_print_top_level_help(prog: str) -> None:
  """Print a top-level usage message for two-stage tyro entrypoints."""
  if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
    print(f"usage: {prog} <TASK> [OPTIONS]")
    print()
    print(f"Run '{prog} <TASK> --help' for task-specific options.")
    print("Run 'uv run list-envs' to list available tasks.")
    sys.exit(0)
