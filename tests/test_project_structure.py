from __future__ import annotations

from pathlib import Path

try:
  import tomllib
except ModuleNotFoundError:
  import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

PRIMARY_TASK_IDS = {
  "Mjlab-TrackingBFM-Flat-Unitree-G1",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop",
  "Mjlab-DistillationBFM-Flat-Unitree-G1",
  "Mjlab-LatentDistillationBFM-Flat-Unitree-G1",
  "Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentVelocityBFM-Flat-Unitree-G1",
}

LEGACY_TASK_ALIASES = {
  "Mjlab-Trackingbfm-Flat-Unitree-G1",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  "Mjlab-Distillation-Flat-Unitree-G1",
  "Mjlab-LatentDistillation-Flat-Unitree-G1",
  "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentRL-Flat-Unitree-G1",
}


def test_project_does_not_vendor_mjlab() -> None:
  assert not (ROOT / "src" / "mjlab").exists()


def test_superpowers_outputs_are_gitignored() -> None:
  gitignore = (ROOT / ".gitignore").read_text()

  assert "docs/superpowers/" in gitignore


def test_package_layout_uses_tracking_bfm_namespace() -> None:
  assert (ROOT / "src" / "tracking_bfm" / "__init__.py").is_file()
  assert (ROOT / "src" / "tracking_bfm" / "tasks" / "registry.py").is_file()
  assert PYPROJECT["project"]["name"] == "tracking-bfm"


def test_pyproject_registers_mjlab_tasks_entry_point() -> None:
  entry_points = PYPROJECT["project"]["entry-points"]["mjlab.tasks"]
  assert entry_points["tracking_bfm"] == "tracking_bfm"


def test_readme_documents_primary_task_ids_and_legacy_aliases() -> None:
  readme = (ROOT / "README.md").read_text()

  for task_id in PRIMARY_TASK_IDS:
    assert task_id in readme

  for alias in LEGACY_TASK_ALIASES:
    assert alias in readme


def test_readme_documents_module_boundaries_and_commands() -> None:
  readme = (ROOT / "README.md").read_text()

  for module in (
    "tracking_bfm.tasks",
    "tracking_bfm.export",
    "tracking_bfm.data_process",
    "tracking_bfm.scripts",
  ):
    assert module in readme

  for command in (
    "uv sync",
    "uv run tracking-bfm-train",
    "uv run tracking-bfm-play",
    "uv run tracking-bfm-evaluate",
    "uv run tracking-bfm-export-onnx",
  ):
    assert command in readme
