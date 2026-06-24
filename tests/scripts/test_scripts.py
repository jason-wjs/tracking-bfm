from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


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
