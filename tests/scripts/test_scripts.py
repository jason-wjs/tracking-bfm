from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

ROOT_QUICK_SCRIPTS = {
  "train.sh": "tracking-bfm-train",
  "play.sh": "tracking-bfm-play",
  "evaluate.sh": "tracking-bfm-evaluate",
  "export.sh": "tracking-bfm-export-onnx",
  "data_process.sh": "tracking-bfm-filter-motions",
  "diagnostics.sh": "tracking-bfm-inspect-checkpoint",
}

PRIMARY_TRACKING_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1"


class FakePlayMotionCommand:
  def __init__(self) -> None:
    self.motion_file = "old.npz"
    self.motion_path = ""
    self.motion_type = "isaaclab"
    self.sampling_mode = "start"


class FakePlayEnvCfg:
  def __init__(self) -> None:
    self.commands = {"motion": FakePlayMotionCommand()}


def _definitions(path: Path) -> set[str]:
  tree = ast.parse(path.read_text(encoding="utf-8"))
  return {
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
  }


def _mjlab_scripts_imports(path: Path) -> list[str]:
  tree = ast.parse(path.read_text(encoding="utf-8"))
  imports = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name == "mjlab.scripts" or alias.name.startswith("mjlab.scripts."):
          imports.append(alias.name)
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
      if node.module == "mjlab.scripts" or node.module.startswith("mjlab.scripts."):
        imports.append(node.module)
  return imports


def test_train_play_evaluate_expose_main_entrypoints() -> None:
  script_dir = ROOT / "src" / "tracking_bfm" / "scripts"

  for script_name in ("train.py", "play.py", "evaluate.py"):
    assert "main" in _definitions(script_dir / script_name)


def test_root_quick_scripts_are_workflow_level_wrappers() -> None:
  script_dir = ROOT / "scripts"
  actual_shell_scripts = {path.name for path in script_dir.glob("*.sh")}

  assert actual_shell_scripts == set(ROOT_QUICK_SCRIPTS)
  assert not (script_dir / "_common.sh").exists()

  for script_name, command in ROOT_QUICK_SCRIPTS.items():
    path = script_dir / script_name
    assert path.is_file()
    assert os.access(path, os.X_OK)
    text = path.read_text()
    assert f"uv run {command}" in text
    assert "source " not in text
    assert "tracking_bfm_add_arg_if_set" not in text
    assert "uv run train" not in text
    assert "uv run play" not in text
    assert "H100" not in text
    assert "adaptive_sampling" not in text


def test_root_quick_scripts_document_workflow_usage() -> None:
  assert not (ROOT / "scripts" / "README.md").exists()
  project_readme = (ROOT / "README.md").read_text()

  for command in (
    "./scripts/train.sh",
    "./scripts/play.sh",
    "./scripts/evaluate.sh",
    "./scripts/export.sh",
    "./scripts/data_process.sh",
    "./scripts/diagnostics.sh",
  ):
    assert command in project_readme


def test_script_modules_importable() -> None:
  pytest.importorskip("mjlab")
  module_names = [
    "tracking_bfm.scripts.train",
    "tracking_bfm.scripts.play",
    "tracking_bfm.scripts.evaluate",
    "tracking_bfm.scripts.diagnostics.analyze_latent_space",
    "tracking_bfm.scripts.diagnostics.inspect_checkpoint",
  ]

  for module_name in module_names:
    module = importlib.import_module(module_name)
    assert callable(module.main)


def test_top_level_help_exits_before_tyro_task_parse(monkeypatch, capsys) -> None:
  from tracking_bfm.scripts.cli_helpers import maybe_print_top_level_help

  monkeypatch.setattr(sys, "argv", ["tracking-bfm-train", "--help"])

  with pytest.raises(SystemExit) as exc_info:
    maybe_print_top_level_help("tracking-bfm-train")

  assert exc_info.value.code == 0
  output = capsys.readouterr().out
  assert "usage: tracking-bfm-train <TASK> [OPTIONS]" in output
  assert "tracking-bfm-train <TASK> --help" in output
  assert "uv run list-envs" in output


@pytest.mark.parametrize(
  ("module_name", "command_name"),
  [
    ("tracking_bfm.scripts.train", "tracking-bfm-train"),
    ("tracking_bfm.scripts.play", "tracking-bfm-play"),
    ("tracking_bfm.scripts.evaluate", "tracking-bfm-evaluate"),
    (
      "tracking_bfm.scripts.diagnostics.analyze_latent_space",
      "tracking-bfm-analyze-latent-space",
    ),
  ],
)
def test_tyro_task_entrypoints_show_console_script_top_level_help(
  monkeypatch, capsys, module_name: str, command_name: str
) -> None:
  module = importlib.import_module(module_name)
  monkeypatch.setattr(sys, "argv", [command_name, "--help"])

  with pytest.raises(SystemExit) as exc_info:
    module.main()

  assert exc_info.value.code == 0
  output = capsys.readouterr().out
  assert f"usage: {command_name} <TASK> [OPTIONS]" in output
  assert f"{command_name} <TASK> --help" in output


def test_tracking_bfm_scripts_do_not_import_mjlab_private_cli() -> None:
  source_root = ROOT / "src" / "tracking_bfm"

  offenders = {
    path: imports
    for path in source_root.rglob("*.py")
    if (imports := _mjlab_scripts_imports(path))
  }

  assert offenders == {}


def test_motion_source_recognizes_registered_multi_motion_command() -> None:
  pytest.importorskip("mjlab")

  from mjlab.tasks.registry import load_env_cfg

  import tracking_bfm  # noqa: F401
  from tracking_bfm.motion_source import (
    is_motion_command_cfg,
    motion_command_source_shape,
  )

  motion_cmd = load_env_cfg(PRIMARY_TRACKING_ID).commands["motion"]

  assert is_motion_command_cfg(motion_cmd)
  assert motion_command_source_shape(motion_cmd) == "multi"


def test_play_motion_path_takes_priority_over_wandb_run_path(capsys) -> None:
  pytest.importorskip("mjlab")

  from tracking_bfm.scripts.play import PlayConfig, _apply_tracking_motion_source

  env_cfg = FakePlayEnvCfg()
  motion_cmd = env_cfg.commands["motion"]
  cfg = PlayConfig(motion_path="cli_motions", wandb_run_path="entity/project/run")

  applied = _apply_tracking_motion_source(env_cfg, cfg, dummy_mode=False)

  assert applied is not None
  assert applied.motion_path == "cli_motions"
  assert motion_cmd.motion_path == "cli_motions"
  assert motion_cmd.motion_file == ""
  assert "cli_motions" in capsys.readouterr().out


def test_play_dummy_tracking_requires_motion_source() -> None:
  pytest.importorskip("mjlab")

  from tracking_bfm.scripts.play import PlayConfig, _apply_tracking_motion_source

  env_cfg = FakePlayEnvCfg()

  with pytest.raises(ValueError, match="Tracking tasks require a motion source"):
    _apply_tracking_motion_source(env_cfg, PlayConfig(agent="zero"), dummy_mode=True)


def test_play_demo_mode_sets_uniform_sampling() -> None:
  pytest.importorskip("mjlab")

  from tracking_bfm.scripts.play import PlayConfig, _apply_tracking_motion_source

  env_cfg = FakePlayEnvCfg()
  motion_cmd = env_cfg.commands["motion"]
  cfg = PlayConfig(motion_path="demo_motions", motion_type="mujoco", _demo_mode=True)

  _apply_tracking_motion_source(env_cfg, cfg, dummy_mode=False)

  assert motion_cmd.motion_type == "mujoco"
  assert motion_cmd.sampling_mode == "uniform"


def test_inspect_checkpoint_prints_json(tmp_path: Path, capsys) -> None:
  torch = pytest.importorskip("torch")
  checkpoint_path = tmp_path / "model_1.pt"
  torch.save({"iter": 1, "actor_state_dict": {"weight": torch.ones(1)}}, checkpoint_path)

  from tracking_bfm.scripts.diagnostics.inspect_checkpoint import main

  original_argv = sys.argv
  try:
    sys.argv = ["tracking-bfm-inspect-checkpoint", str(checkpoint_path), "--json"]
    main()
  finally:
    sys.argv = original_argv

  payload = json.loads(capsys.readouterr().out)
  assert payload["keys"] == ["actor_state_dict", "iter"]
  assert payload["state_dicts"]["actor_state_dict"]["tensor_count"] == 1
