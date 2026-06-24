"""Safe cleanup helpers for failed motion filtering reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FailedMotionCleanupConfig:
  """Configuration for deleting failed motions from a JSON report."""

  report_file: str
  execute: bool = False
  missing_ok: bool = True


@dataclass(frozen=True)
class FailedMotionCleanupResult:
  """Summary of a failed-motion cleanup run."""

  report_file: Path
  dry_run: bool
  matched_paths: tuple[Path, ...]
  would_delete_paths: tuple[Path, ...]
  deleted_paths: tuple[Path, ...]
  missing_paths: tuple[Path, ...]

  @property
  def matched_count(self) -> int:
    return len(self.matched_paths)

  @property
  def would_delete_count(self) -> int:
    return len(self.would_delete_paths)

  @property
  def deleted_count(self) -> int:
    return len(self.deleted_paths)

  @property
  def missing_count(self) -> int:
    return len(self.missing_paths)


def extract_failed_motion_paths(
  report: Mapping[str, Any],
  *,
  base_dir: Path | None = None,
) -> list[Path]:
  """Return unique failed motion paths from a filtering/generation report."""

  failed_paths: set[Path] = set()
  for entry in report.get("failed_motions", []):
    if not isinstance(entry, Mapping):
      continue

    path_value = entry.get("path")
    if not isinstance(path_value, str) or path_value == "":
      continue

    motion_path = Path(path_value).expanduser()
    if not motion_path.is_absolute() and base_dir is not None:
      motion_path = base_dir / motion_path
    failed_paths.add(motion_path.resolve(strict=False))

  return sorted(failed_paths, key=lambda path: str(path))


def load_failed_motion_paths(report_file: str | Path) -> list[Path]:
  """Load and parse failed motion paths from a report JSON file."""

  report_path = Path(report_file)
  if not report_path.exists():
    raise FileNotFoundError(f"Report file not found: {report_path}")

  with report_path.open("r", encoding="utf-8") as file:
    report = json.load(file)
  if not isinstance(report, Mapping):
    raise ValueError(f"Expected report object in {report_path}")

  return extract_failed_motion_paths(report, base_dir=report_path.parent)


def cleanup_failed_motions(
  cfg: FailedMotionCleanupConfig,
) -> FailedMotionCleanupResult:
  """Delete failed motions only when ``cfg.execute`` is true.

  The default mode is a dry run. It reports which existing files would be deleted
  without mutating the filesystem.
  """

  report_path = Path(cfg.report_file)
  failed_paths = load_failed_motion_paths(report_path)
  would_delete_paths: list[Path] = []
  deleted_paths: list[Path] = []
  missing_paths: list[Path] = []

  for motion_path in failed_paths:
    if motion_path.exists():
      if motion_path.is_dir():
        raise IsADirectoryError(f"Refusing to delete directory: {motion_path}")
      if cfg.execute:
        motion_path.unlink()
        deleted_paths.append(motion_path)
      else:
        would_delete_paths.append(motion_path)
      continue

    if not cfg.missing_ok:
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    missing_paths.append(motion_path)

  return FailedMotionCleanupResult(
    report_file=report_path.resolve(strict=False),
    dry_run=not cfg.execute,
    matched_paths=tuple(failed_paths),
    would_delete_paths=tuple(would_delete_paths),
    deleted_paths=tuple(deleted_paths),
    missing_paths=tuple(missing_paths),
  )


__all__ = [
  "FailedMotionCleanupConfig",
  "FailedMotionCleanupResult",
  "cleanup_failed_motions",
  "extract_failed_motion_paths",
  "load_failed_motion_paths",
]
