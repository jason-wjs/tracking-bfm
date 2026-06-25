from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab.rl.exporter_utils import get_base_metadata


@dataclass(frozen=True)
class PolicyArtifactPaths:
  policy_dir: Path
  filename: str
  onnx_path: Path


def policy_artifact_paths(checkpoint_path: str | Path) -> PolicyArtifactPaths:
  policy_dir = Path(checkpoint_path).parent
  filename = f"{policy_dir.name}.onnx"
  return PolicyArtifactPaths(
    policy_dir=policy_dir,
    filename=filename,
    onnx_path=policy_dir / filename,
  )


def motion_metadata(env: Any, run_name: str) -> dict[str, Any]:
  metadata = get_base_metadata(env.unwrapped, run_name)
  motion_term = env.unwrapped.command_manager.get_term("motion")
  metadata.update(
    {
      "anchor_body_name": motion_term.cfg.anchor_body_name,
      "body_names": list(motion_term.cfg.body_names),
    }
  )
  return metadata
