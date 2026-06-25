from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tracking_bfm.tasks.tracking.rl import policy_artifact as artifact_module
from tracking_bfm.tasks.tracking.rl import runner as runner_module
from tracking_bfm.tasks.tracking.rl.policy_artifact import (
  PolicyArtifactPaths,
  motion_metadata,
  policy_artifact_paths,
)
from tracking_bfm.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


def test_policy_artifact_paths_match_mjlab_runner_export_paths() -> None:
  paths = policy_artifact_paths(Path("/tmp/runs/walk/model_42.pt"))

  assert paths == PolicyArtifactPaths(
    policy_dir=Path("/tmp/runs/walk"),
    filename="walk.onnx",
    onnx_path=Path("/tmp/runs/walk/walk.onnx"),
  )


def test_motion_metadata_adds_motion_reference_config(monkeypatch) -> None:
  motion_cfg = SimpleNamespace(
    anchor_body_name="torso_link",
    body_names=("torso_link", "left_foot", "right_foot"),
  )
  motion_term = SimpleNamespace(cfg=motion_cfg)
  command_manager = SimpleNamespace(get_term=lambda name: motion_term)
  unwrapped_env = SimpleNamespace(command_manager=command_manager)
  env = SimpleNamespace(unwrapped=unwrapped_env)
  calls = []

  def fake_get_base_metadata(base_env, run_name):
    calls.append((base_env, run_name))
    return {"run_name": run_name, "source": "base"}

  monkeypatch.setattr(
    artifact_module,
    "get_base_metadata",
    fake_get_base_metadata,
  )

  metadata = motion_metadata(env, "wandb-run")

  assert calls == [(unwrapped_env, "wandb-run")]
  assert metadata == {
    "run_name": "wandb-run",
    "source": "base",
    "anchor_body_name": "torso_link",
    "body_names": ["torso_link", "left_foot", "right_foot"],
  }


def _make_runner(monkeypatch, *, upload_model: bool, logger_type: str = "local"):
  runner = object.__new__(MotionTrackingOnPolicyRunner)
  runner.cfg = {"upload_model": upload_model}
  runner.logger = SimpleNamespace(logger_type=logger_type)
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace())
  runner.registry_name = None
  base_saves = []

  def fake_base_save(self, path, infos=None):
    base_saves.append((self, path, infos))

  monkeypatch.setattr(runner_module.MjlabOnPolicyRunner, "save", fake_base_save)
  return runner, base_saves


def test_runner_save_skips_policy_artifact_when_upload_model_is_disabled(
  monkeypatch,
) -> None:
  runner, base_saves = _make_runner(monkeypatch, upload_model=False)

  def unexpected_export(*args, **kwargs):
    raise AssertionError("export should not run when upload_model is disabled")

  runner.export_policy_to_onnx = unexpected_export

  runner.save("/tmp/runs/walk/model_42.pt", infos={"iter": 42})

  assert base_saves == [(runner, "/tmp/runs/walk/model_42.pt", {"iter": 42})]


def test_runner_save_exports_and_attaches_local_metadata(monkeypatch) -> None:
  runner, base_saves = _make_runner(monkeypatch, upload_model=True)
  exports = []
  metadata_calls = []
  attaches = []

  def fake_export(path, filename):
    exports.append((path, filename))

  def fake_motion_metadata(env, run_name):
    metadata_calls.append((env, run_name))
    return {"run_name": run_name}

  def fake_attach_metadata_to_onnx(path, metadata):
    attaches.append((path, metadata))

  runner.export_policy_to_onnx = fake_export
  monkeypatch.setattr(runner_module, "motion_metadata", fake_motion_metadata)
  monkeypatch.setattr(
    runner_module, "attach_metadata_to_onnx", fake_attach_metadata_to_onnx
  )
  monkeypatch.setattr(runner_module, "wandb", SimpleNamespace(run=None))

  runner.save("/tmp/runs/walk/model_42.pt")

  assert base_saves == [(runner, "/tmp/runs/walk/model_42.pt", None)]
  assert exports == [("/tmp/runs/walk", "walk.onnx")]
  assert metadata_calls == [(runner.env, "local")]
  assert attaches == [("/tmp/runs/walk/walk.onnx", {"run_name": "local"})]


def test_runner_save_uploads_wandb_artifact_and_clears_registry_name(
  monkeypatch,
) -> None:
  runner, _ = _make_runner(monkeypatch, upload_model=True, logger_type="wandb")
  runner.registry_name = "entity/project/motions:latest"
  saved = []
  used_artifacts = []
  metadata_calls = []
  fake_run = SimpleNamespace(
    name="wandb-run",
    path="entity/project/run_id",
    use_artifact=lambda registry_name: used_artifacts.append(registry_name),
  )

  runner.export_policy_to_onnx = lambda path, filename: None
  monkeypatch.setattr(
    runner_module,
    "motion_metadata",
    lambda env, run_name: metadata_calls.append((env, run_name))
    or {"run_name": run_name},
  )
  monkeypatch.setattr(runner_module, "attach_metadata_to_onnx", lambda path, meta: None)
  monkeypatch.setattr(
    runner_module,
    "wandb",
    SimpleNamespace(
      run=fake_run,
      save=lambda path, base_path: saved.append((path, base_path)),
    ),
  )

  runner.save("/tmp/runs/walk/model_42.pt")

  assert saved == [("/tmp/runs/walk/walk.onnx", "/tmp/runs/walk")]
  assert metadata_calls == [(runner.env, "entity/project/run_id")]
  assert used_artifacts == ["entity/project/motions:latest"]
  assert runner.registry_name is None


def test_runner_save_warns_and_continues_when_export_fails(
  monkeypatch, capsys
) -> None:
  runner, base_saves = _make_runner(monkeypatch, upload_model=True)

  def failing_export(path, filename):
    raise RuntimeError("boom")

  runner.export_policy_to_onnx = failing_export

  runner.save("/tmp/runs/walk/model_42.pt")

  assert base_saves == [(runner, "/tmp/runs/walk/model_42.pt", None)]
  assert "[WARN] ONNX export failed (training continues): boom" in capsys.readouterr().out
