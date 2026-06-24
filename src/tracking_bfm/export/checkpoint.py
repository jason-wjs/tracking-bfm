"""Checkpoint and ONNX output-path helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

CheckpointFamily: TypeAlias = Literal["tracking", "distillation"]
CheckpointFamilyOption: TypeAlias = CheckpointFamily | Literal["auto"]


def detect_checkpoint_family(checkpoint: Mapping[str, Any]) -> CheckpointFamily:
  """Infer the export family from a saved checkpoint mapping."""
  if "actor_state_dict" in checkpoint:
    return "tracking"
  if "policy_state_dict" in checkpoint:
    return "distillation"
  raise ValueError(
    "Unsupported checkpoint format: expected `actor_state_dict` or "
    "`policy_state_dict`."
  )


def resolve_onnx_output_path(
  checkpoint_path: str | Path,
  output_name: str | Path | None = None,
) -> Path:
  """Resolve an ONNX output path beside a source checkpoint."""
  checkpoint_path = Path(checkpoint_path)
  if output_name is None:
    output_name = f"deploy_{checkpoint_path.stem}.onnx"
  else:
    output_name = str(output_name)
    if not output_name.endswith(".onnx"):
      output_name = f"{output_name}.onnx"
  return checkpoint_path.parent / output_name


def ensure_output_path_available(
  output_path: str | Path,
  *,
  overwrite: bool = False,
) -> Path:
  """Reject accidental ONNX overwrites unless explicitly requested."""
  output_path = Path(output_path)
  if output_path.exists() and not overwrite:
    raise FileExistsError(
      f"{output_path} already exists; pass overwrite=True or --overwrite to replace it."
    )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  return output_path


def load_checkpoint(
  checkpoint_path: str | Path,
  *,
  map_location: str = "cpu",
) -> Mapping[str, Any]:
  """Load a torch checkpoint with the repository's expected options."""
  import torch

  return torch.load(
    Path(checkpoint_path),
    map_location=map_location,
    weights_only=False,
  )

__all__ = [
  "CheckpointFamily",
  "CheckpointFamilyOption",
  "detect_checkpoint_family",
  "ensure_output_path_available",
  "load_checkpoint",
  "resolve_onnx_output_path",
]
