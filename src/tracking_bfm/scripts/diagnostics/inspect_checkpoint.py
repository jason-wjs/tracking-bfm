"""Inspect a BFM checkpoint without modifying it."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint")
  parser.add_argument(
    "--json",
    action="store_true",
    help="Emit a JSON summary instead of human-readable text.",
  )
  return parser


def _tensor_state_summary(value: Any) -> dict[str, Any] | None:
  if not isinstance(value, Mapping):
    return None

  tensor_shapes: dict[str, list[int]] = {}
  tensor_count = 0
  parameter_count = 0
  for key, item in value.items():
    if not torch.is_tensor(item):
      continue
    tensor_count += 1
    parameter_count += int(item.numel())
    tensor_shapes[str(key)] = list(item.shape)

  if tensor_count == 0:
    return None
  return {
    "tensor_count": tensor_count,
    "parameter_count": parameter_count,
    "tensor_shapes": tensor_shapes,
  }


def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
  """Return a bounded checkpoint summary suitable for diagnostics."""
  checkpoint_path = Path(checkpoint_path)
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, Mapping):
    return {
      "path": str(checkpoint_path),
      "type": type(checkpoint).__name__,
      "keys": [],
      "state_dicts": {},
    }

  state_dicts = {
    str(key): summary
    for key, value in checkpoint.items()
    if (summary := _tensor_state_summary(value)) is not None
  }
  return {
    "path": str(checkpoint_path),
    "type": type(checkpoint).__name__,
    "keys": sorted(str(key) for key in checkpoint),
    "state_dicts": state_dicts,
  }


def main(argv: Sequence[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  summary = inspect_checkpoint(args.checkpoint)
  if args.json:
    print(json.dumps(summary, indent=2, sort_keys=True))
    return

  print(f"path: {summary['path']}")
  print(f"type: {summary['type']}")
  print("keys:")
  for key in summary["keys"]:
    print(f"  - {key}")
  if summary["state_dicts"]:
    print("state_dicts:")
    for key, state in summary["state_dicts"].items():
      print(
        f"  - {key}: {state['tensor_count']} tensors, "
        f"{state['parameter_count']} parameters"
      )


if __name__ == "__main__":
  main()
