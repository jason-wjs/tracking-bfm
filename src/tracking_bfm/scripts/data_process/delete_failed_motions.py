"""CLI wrapper for safe failed-motion cleanup."""

from __future__ import annotations

import argparse

from tracking_bfm.data_process.failed_motion_cleanup import (
  FailedMotionCleanupConfig,
  cleanup_failed_motions,
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="tracking-bfm-delete-failed-motions",
    description="Delete failed motion files from a filtering report.",
  )
  parser.add_argument(
    "--report-file", required=True, help="Filtering report JSON file."
  )
  parser.add_argument(
    "--execute",
    action="store_true",
    help="Actually delete files. The default is a dry run.",
  )
  parser.add_argument(
    "--strict-missing",
    action="store_false",
    dest="missing_ok",
    help="Fail if a reported motion file is already missing.",
  )
  parser.set_defaults(missing_ok=True)
  return parser


def config_from_args(args: argparse.Namespace) -> FailedMotionCleanupConfig:
  return FailedMotionCleanupConfig(
    report_file=args.report_file,
    execute=args.execute,
    missing_ok=args.missing_ok,
  )


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  result = cleanup_failed_motions(config_from_args(args))

  if result.dry_run:
    print(
      f"[DRY-RUN] Would delete {result.would_delete_count} motion files from "
      f"{result.report_file}."
    )
    print("[DRY-RUN] Re-run with --execute to remove files.")
  else:
    print(
      f"[INFO] Deleted {result.deleted_count} motion files from {result.report_file}."
    )

  if result.missing_count:
    print(f"[INFO] Skipped {result.missing_count} missing files.")


if __name__ == "__main__":
  main()
