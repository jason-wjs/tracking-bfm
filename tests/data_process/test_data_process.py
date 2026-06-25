from __future__ import annotations

import importlib
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _data_process_modules():
  """Import data-process modules without importing root task registration."""

  original_root = sys.modules.get("tracking_bfm")
  loaded_before = set(sys.modules)
  root_stub = types.ModuleType("tracking_bfm")
  root_stub.__path__ = [str(ROOT / "src" / "tracking_bfm")]  # type: ignore[attr-defined]
  sys.modules["tracking_bfm"] = root_stub

  try:
    cleanup_module = importlib.import_module(
      "tracking_bfm.data_process.failed_motion_cleanup"
    )
    delete_cli_module = importlib.import_module(
      "tracking_bfm.scripts.data_process.delete_failed_motions"
    )
    filter_cli_module = importlib.import_module(
      "tracking_bfm.scripts.data_process.filter_motions"
    )
    generate_cli_module = importlib.import_module(
      "tracking_bfm.scripts.data_process.generate_motion_dataset"
    )
    yield cleanup_module, delete_cli_module, filter_cli_module, generate_cli_module
  finally:
    for module_name in list(sys.modules):
      if module_name.startswith("tracking_bfm.") and module_name not in loaded_before:
        sys.modules.pop(module_name, None)
    if original_root is None:
      sys.modules.pop("tracking_bfm", None)
    else:
      sys.modules["tracking_bfm"] = original_root


def _write_report(report_path: Path, failed_motions: list[object]) -> Path:
  report_path.write_text(
    json.dumps({"failed_motions": failed_motions}),
    encoding="utf-8",
  )
  return report_path


def test_load_failed_motion_paths_resolves_relative_paths_and_deduplicates(
  tmp_path: Path,
) -> None:
  motion_path = tmp_path / "motions" / "failed.npz"
  report_path = _write_report(
    tmp_path / "report.json",
    [
      {"path": "motions/failed.npz"},
      {"path": str(motion_path)},
      {"path": 123},
      {"not_path": "ignored.npz"},
      "ignored",
    ],
  )

  with _data_process_modules() as modules:
    cleanup_module = modules[0]

    assert cleanup_module.load_failed_motion_paths(report_path) == [
      motion_path.resolve()
    ]


def test_cleanup_failed_motions_dry_run_does_not_delete_files(tmp_path: Path) -> None:
  motion_path = tmp_path / "failed.npz"
  motion_path.write_bytes(b"motion")
  report_path = _write_report(tmp_path / "report.json", [{"path": str(motion_path)}])

  with _data_process_modules() as modules:
    cleanup_module = modules[0]
    result = cleanup_module.cleanup_failed_motions(
      cleanup_module.FailedMotionCleanupConfig(report_file=str(report_path))
    )

  assert result.dry_run is True
  assert result.would_delete_count == 1
  assert result.deleted_count == 0
  assert motion_path.exists()


def test_cleanup_failed_motions_execute_deletes_files_in_tmpdir(
  tmp_path: Path,
) -> None:
  first_motion = tmp_path / "failed-a.npz"
  second_motion = tmp_path / "nested" / "failed-b.npz"
  second_motion.parent.mkdir()
  first_motion.write_bytes(b"a")
  second_motion.write_bytes(b"b")
  report_path = _write_report(
    tmp_path / "report.json",
    [{"path": str(first_motion)}, {"path": "nested/failed-b.npz"}],
  )

  with _data_process_modules() as modules:
    cleanup_module = modules[0]
    result = cleanup_module.cleanup_failed_motions(
      cleanup_module.FailedMotionCleanupConfig(
        report_file=str(report_path),
        execute=True,
      )
    )

  assert result.dry_run is False
  assert result.deleted_count == 2
  assert result.would_delete_count == 0
  assert not first_motion.exists()
  assert not second_motion.exists()


def test_cli_wrappers_are_importable_and_parse_core_arguments(tmp_path: Path) -> None:
  with _data_process_modules() as modules:
    _, delete_failed_motions, filter_motions, generate_motion_dataset = modules

  filter_args = filter_motions.build_parser().parse_args(
    [
      "TrackingTask",
      "--motion-path",
      str(tmp_path),
      "--checkpoint-file",
      "policy.pt",
      "--failure-threshold",
      "0.8",
    ]
  )
  generate_args = generate_motion_dataset.build_parser().parse_args(
    [
      "TrackingTask",
      "--motion-path",
      str(tmp_path),
      "--checkpoint-file",
      "policy.pt",
      "--completion-threshold",
      "0.95",
    ]
  )
  delete_args = delete_failed_motions.build_parser().parse_args(
    ["--report-file", "report.json"]
  )
  execute_delete_args = delete_failed_motions.build_parser().parse_args(
    ["--report-file", "report.json", "--execute"]
  )

  assert filter_args.task_id == "TrackingTask"
  assert filter_args.failure_threshold == 0.8
  assert generate_args.task_id == "TrackingTask"
  assert generate_args.completion_threshold == 0.95
  assert delete_args.execute is False
  assert execute_delete_args.execute is True


def test_motion_filtering_configures_motion_source_and_history_steps() -> None:
  with _data_process_modules():
    motion_filtering = importlib.import_module(
      "tracking_bfm.data_process.motion_filtering"
    )

  command = SimpleNamespace(motion_file="old.npz", motion_path="")

  motion_filtering._configure_motion_command(
    command,
    motion_path="motions",
    motion_type="mujoco",
    history_steps=2,
    future_steps=3,
  )

  assert command.motion_file == ""
  assert command.motion_path == "motions"
  assert command.motion_type == "mujoco"
  assert command.history_steps == 2
  assert command.future_steps == 3


def test_motion_filtering_prefers_explicit_motion_path_without_wandb_lookup(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  with _data_process_modules():
    motion_filtering = importlib.import_module(
      "tracking_bfm.data_process.motion_filtering"
    )

  class FailingApi:
    def __init__(self) -> None:
      raise AssertionError("explicit motion_path should not touch W&B")

  monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=FailingApi))
  cfg = SimpleNamespace(motion_path=str(tmp_path), wandb_run_path="entity/project/run")

  assert motion_filtering._resolve_motion_root(cfg) == str(tmp_path)
