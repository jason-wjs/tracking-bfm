from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
  import tomllib
else:
  import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPT_ENTRY_POINTS = {
  "tracking-bfm-train": "tracking_bfm.scripts.train:main",
  "tracking-bfm-play": "tracking_bfm.scripts.play:main",
  "tracking-bfm-evaluate": "tracking_bfm.scripts.evaluate:main",
  "tracking-bfm-export": "tracking_bfm.scripts.export:main",
  "tracking-bfm-filter-motions": (
    "tracking_bfm.scripts.data_process.filter_motions:main"
  ),
  "tracking-bfm-generate-motion-dataset": (
    "tracking_bfm.scripts.data_process.generate_motion_dataset:main"
  ),
  "tracking-bfm-delete-failed-motions": (
    "tracking_bfm.scripts.data_process.delete_failed_motions:main"
  ),
  "tracking-bfm-analyze-latent-space": (
    "tracking_bfm.scripts.diagnostics.analyze_latent_space:main"
  ),
  "tracking-bfm-inspect-checkpoint": (
    "tracking_bfm.scripts.diagnostics.inspect_checkpoint:main"
  ),
}


def load_pyproject() -> dict:
  return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_planned_console_scripts_are_registered() -> None:
  scripts = load_pyproject()["project"]["scripts"]

  assert scripts == EXPECTED_SCRIPT_ENTRY_POINTS
  assert "tracking-bfm-export" in scripts
  assert "tracking-bfm-export-onnx" not in scripts
  assert "tracking-bfm-export-latent-onnx" not in scripts


def test_console_scripts_point_to_tracking_bfm_main_functions() -> None:
  scripts = load_pyproject()["project"]["scripts"]

  for target in scripts.values():
    module, function = target.split(":")
    assert module.startswith("tracking_bfm.scripts.")
    assert not module.startswith("mjlab.")
    assert function == "main"


def test_mjlab_tasks_entry_point_points_to_tracking_bfm_package() -> None:
  entry_points = load_pyproject()["project"]["entry-points"]["mjlab.tasks"]

  assert entry_points == {"tracking_bfm": "tracking_bfm"}
