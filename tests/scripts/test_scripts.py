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


def _definitions(path: Path) -> set[str]:
  tree = ast.parse(path.read_text(encoding="utf-8"))
  return {
    node.name
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
  }


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
  script_readme = (ROOT / "scripts" / "README.md").read_text()
  project_readme = (ROOT / "README.md").read_text()

  for script_name in ROOT_QUICK_SCRIPTS:
    assert f"./scripts/{script_name}" in script_readme

  for command in (
    "./scripts/train.sh",
    "./scripts/play.sh",
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
