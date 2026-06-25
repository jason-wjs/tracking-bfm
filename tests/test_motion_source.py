from __future__ import annotations

from pathlib import Path

import pytest

from tracking_bfm.motion_source import (
  MotionSourceSpec,
  apply_motion_source_to_command,
  collect_motion_files,
  resolve_motion_source,
)


class SingleMotionCommand:
  def __init__(self) -> None:
    self.motion_file = ""


class MultiMotionCommand:
  def __init__(self) -> None:
    self.motion_file = "old.npz"
    self.motion_path = ""


class UnsupportedMotionCommand:
  pass


class FakeArtifact:
  type = "motions"

  def __init__(self, path: Path) -> None:
    self._path = path

  def download(self) -> str:
    return str(self._path)


class FakeRun:
  def __init__(self, artifacts: list[FakeArtifact]) -> None:
    self._artifacts = artifacts

  def used_artifacts(self) -> list[FakeArtifact]:
    return self._artifacts


class FakeWandbApi:
  def __init__(self, path: Path) -> None:
    self._path = path
    self.requested_run_path: str | None = None
    self.requested_registry_name: str | None = None

  def run(self, run_path: str) -> FakeRun:
    self.requested_run_path = run_path
    return FakeRun([FakeArtifact(self._path)])

  def artifact(self, registry_name: str) -> FakeArtifact:
    self.requested_registry_name = registry_name
    return FakeArtifact(self._path)


def test_motion_source_spec_rejects_multiple_sources() -> None:
  with pytest.raises(ValueError, match="Provide only one motion source"):
    resolve_motion_source(
      MotionSourceSpec(motion_file="motion.npz", motion_path="motions")
    )


def test_apply_local_motion_file_to_single_command() -> None:
  command = SingleMotionCommand()
  source = resolve_motion_source(MotionSourceSpec(motion_file="motion.npz"))

  applied = apply_motion_source_to_command(command, source)

  assert command.motion_file == "motion.npz"
  assert applied is not None
  assert applied.shape == "single"
  assert applied.motion_file == "motion.npz"
  assert applied.motion_path is None


def test_apply_local_motion_path_to_multi_command_clears_motion_file() -> None:
  command = MultiMotionCommand()
  source = resolve_motion_source(MotionSourceSpec(motion_path="motions"))

  applied = apply_motion_source_to_command(command, source)

  assert command.motion_path == "motions"
  assert command.motion_file == ""
  assert applied is not None
  assert applied.shape == "multi"
  assert applied.motion_file is None
  assert applied.motion_path == "motions"


def test_single_motion_command_rejects_motion_path() -> None:
  command = SingleMotionCommand()
  source = resolve_motion_source(MotionSourceSpec(motion_path="motions"))

  with pytest.raises(ValueError, match="does not support `motion_path`"):
    apply_motion_source_to_command(command, source)


@pytest.mark.parametrize(
  ("command", "expected_file", "expected_path"),
  [
    (SingleMotionCommand(), "artifact/motion.npz", None),
    (MultiMotionCommand(), None, "artifact"),
  ],
)
def test_wandb_run_artifact_dir_applies_by_command_shape(
  tmp_path: Path,
  command: SingleMotionCommand | MultiMotionCommand,
  expected_file: str | None,
  expected_path: str | None,
) -> None:
  artifact_dir = tmp_path / "artifact"
  api = FakeWandbApi(artifact_dir)

  source = resolve_motion_source(
    MotionSourceSpec(wandb_run_path="entity/project/run"),
    wandb_api=api,
  )
  applied = apply_motion_source_to_command(command, source)

  assert api.requested_run_path == "entity/project/run"
  assert applied is not None
  if expected_file is not None:
    assert command.motion_file == str(tmp_path / expected_file)
    assert applied.motion_file == str(tmp_path / expected_file)
  if expected_path is not None:
    assert command.motion_path == str(tmp_path / expected_path)
    assert command.motion_file == ""
    assert applied.motion_path == str(tmp_path / expected_path)


def test_wandb_registry_name_normalizes_latest_and_preserves_runner_handoff(
  tmp_path: Path,
) -> None:
  command = MultiMotionCommand()
  api = FakeWandbApi(tmp_path / "registry_artifact")

  source = resolve_motion_source(
    MotionSourceSpec(wandb_registry_name="org/motions/gait"),
    wandb_api=api,
  )
  applied = apply_motion_source_to_command(command, source)

  assert api.requested_registry_name == "org/motions/gait:latest"
  assert applied is not None
  assert applied.registry_name == "org/motions/gait:latest"
  assert command.motion_path == str(tmp_path / "registry_artifact")


def test_collect_motion_files_is_sorted_recursive_and_deterministically_sharded(
  tmp_path: Path,
) -> None:
  (tmp_path / "b").mkdir()
  (tmp_path / "b" / "motion_3.NPZ").write_bytes(b"")
  (tmp_path / "motion_1.npz").write_bytes(b"")
  (tmp_path / "motion_2.txt").write_text("ignored")
  (tmp_path / "a").mkdir()
  (tmp_path / "a" / "motion_2.npz").write_bytes(b"")
  (tmp_path / "motion_4.npz").write_bytes(b"")

  motion_files = collect_motion_files(tmp_path)
  shard = collect_motion_files(tmp_path, shard=True, rank=1, world_size=2)

  assert [path.name for path in motion_files] == [
    "motion_2.npz",
    "motion_3.NPZ",
    "motion_1.npz",
    "motion_4.npz",
  ]
  assert [path.name for path in shard] == ["motion_3.NPZ", "motion_4.npz"]


def test_apply_motion_source_rejects_unsupported_command() -> None:
  source = resolve_motion_source(MotionSourceSpec(motion_file="motion.npz"))

  with pytest.raises(ValueError, match="motion command"):
    apply_motion_source_to_command(UnsupportedMotionCommand(), source)
