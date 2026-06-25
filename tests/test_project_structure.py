from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
  import tomllib
else:
  import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())

PRIMARY_TASK_IDS = {
  "Mjlab-TrackingBFM-Flat-Unitree-G1",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal",
  "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
  "Mjlab-DistillationBFM-Flat-Unitree-G1",
  "Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs",
  "Mjlab-LatentDistillationBFM-Flat-Unitree-G1",
  "Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentVelocityBFM-Flat-Unitree-G1",
  "Mjlab-LatentVelocityBFM-Rough-Unitree-G1",
}

LEGACY_TASK_ALIASES = {
  "Mjlab-Trackingbfm-Flat-Unitree-G1",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
  "Mjlab-Distillation-Flat-Unitree-G1",
  "Mjlab-DistillationWbteleopObs-Flat-Unitree-G1",
  "Mjlab-LatentDistillation-Flat-Unitree-G1",
  "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentRL-Flat-Unitree-G1",
  "Mjlab-LatentRL-Rough-Unitree-G1",
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


def test_tracking_bfm_init_does_not_hide_missing_hard_dependencies() -> None:
  package_init = (ROOT / "src" / "tracking_bfm" / "__init__.py").read_text()

  assert "except ModuleNotFoundError" not in package_init
  assert "from tracking_bfm import tasks as tasks" in package_init


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


def test_architecture_context_and_migration_docs_exist() -> None:
  context = (ROOT / "CONTEXT.md").read_text()
  adr = (ROOT / "docs" / "adr" / "0001-standalone-mjlab-dependency.md").read_text()
  migration = (ROOT / "docs" / "migration.md").read_text()
  motion_source = (ROOT / "docs" / "architecture" / "motion-source.md").read_text()
  tracking_cleanup = (
    ROOT / "docs" / "architecture" / "tracking-cleanup.md"
  ).read_text()

  for term in ("BFM task package", "Motion source", "Legacy task alias"):
    assert term in context

  for decision in (
    "depend on mjlab as an external package",
    "do not vendor or fork mjlab",
    "do not import private mjlab script modules",
  ):
    assert decision in adr

  for feature in (
    "Mjlab-TrackingBFM-Flat-Unitree-G1",
    "Mjlab-Trackingbfm-Flat-Unitree-G1",
    "ActionTrunk",
    "Removed",
    "Existing checkpoints that require this task must stay on the old fork",
    "DistillationWbteleopObs",
    "TestOptimal",
    "NoRegNoDR",
    "Rough",
    "Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk",
    "Mjlab-LatentRL-Rough-Unitree-G1",
  ):
    assert feature in migration

  for item in (
    "MotionSourceSpec",
    "apply_motion_source_to_command",
    "First Migration Batch",
    "Deferred Work",
  ):
    assert item in motion_source

  for item in (
    "reuse-upstream",
    "bfm-owned",
    "compat-shim",
    "legacy-candidate",
    "multi_motion_command.py",
    "wbteleop/",
  ):
    assert item in tracking_cleanup


def test_task_registration_is_explicit_and_fail_fast() -> None:
  task_init = (ROOT / "src" / "tracking_bfm" / "tasks" / "__init__.py").read_text()

  assert "import_packages" not in task_init
  for import_target in (
    "tracking_bfm.tasks.tracking.config.g1",
    "tracking_bfm.tasks.tracking.wbteleop",
    "tracking_bfm.tasks.distillation.config.g1",
    "tracking_bfm.tasks.latent_tracking.config.g1",
    "tracking_bfm.tasks.latent_velocity.config.g1",
  ):
    assert import_target in task_init

  for relative_path in (
    "src/tracking_bfm/tasks/distillation/config/g1/__init__.py",
    "src/tracking_bfm/tasks/latent_tracking/config/g1/__init__.py",
  ):
    registration_module = (ROOT / relative_path).read_text()
    assert "_auto_register_tasks" not in registration_module
    assert "except ModuleNotFoundError" not in registration_module
