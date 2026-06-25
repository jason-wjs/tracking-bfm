"""Motion source resolution between workflow adapters and motion commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MotionCommandSourceShape = Literal["single", "multi"]
MotionSourceKind = Literal["local_file", "local_path", "artifact_dir"]


@dataclass(frozen=True)
class MotionSourceSpec:
  motion_file: str | Path | None = None
  motion_path: str | Path | None = None
  wandb_run_path: str | None = None
  wandb_registry_name: str | None = None


@dataclass(frozen=True)
class ResolvedMotionSource:
  kind: MotionSourceKind
  path: Path
  registry_name: str | None = None


@dataclass(frozen=True)
class AppliedMotionSource:
  shape: MotionCommandSourceShape
  motion_file: str | None = None
  motion_path: str | None = None
  registry_name: str | None = None


def _provided_sources(spec: MotionSourceSpec) -> list[str]:
  sources = []
  if spec.motion_file is not None:
    sources.append("motion_file")
  if spec.motion_path is not None:
    sources.append("motion_path")
  if spec.wandb_run_path is not None:
    sources.append("wandb_run_path")
  if spec.wandb_registry_name is not None:
    sources.append("wandb_registry_name")
  return sources


def _wandb_api(wandb_api: Any | None) -> Any:
  if wandb_api is not None:
    return wandb_api

  import wandb

  return wandb.Api()


def _motions_artifact_from_run(run: Any) -> Any:
  artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if artifact is None:
    raise RuntimeError("No motion artifact found in the W&B run.")
  return artifact


def normalize_wandb_registry_name(registry_name: str) -> str:
  if ":" in registry_name:
    return registry_name
  return f"{registry_name}:latest"


def resolve_motion_source(
  spec: MotionSourceSpec,
  *,
  wandb_api: Any | None = None,
  required: bool = False,
) -> ResolvedMotionSource | None:
  """Resolve a local or W&B motion source without applying it to a command."""
  sources = _provided_sources(spec)
  if len(sources) > 1:
    raise ValueError("Provide only one motion source.")
  if not sources:
    if required:
      raise ValueError("Provide a motion source.")
    return None

  if spec.motion_file is not None:
    return ResolvedMotionSource(kind="local_file", path=Path(spec.motion_file))
  if spec.motion_path is not None:
    return ResolvedMotionSource(kind="local_path", path=Path(spec.motion_path))
  if spec.wandb_run_path is not None:
    api = _wandb_api(wandb_api)
    artifact = _motions_artifact_from_run(api.run(spec.wandb_run_path))
    return ResolvedMotionSource(kind="artifact_dir", path=Path(artifact.download()))

  assert spec.wandb_registry_name is not None
  registry_name = normalize_wandb_registry_name(spec.wandb_registry_name)
  api = _wandb_api(wandb_api)
  artifact = api.artifact(registry_name)
  return ResolvedMotionSource(
    kind="artifact_dir",
    path=Path(artifact.download()),
    registry_name=registry_name,
  )


def motion_command_source_shape(command: Any) -> MotionCommandSourceShape:
  """Return the source shape supported by a motion command config."""
  if command is None:
    raise ValueError("The selected task is not a tracking task with a motion command.")
  if hasattr(command, "motion_path"):
    return "multi"
  if hasattr(command, "motion_file"):
    return "single"
  raise ValueError("The selected task is not a tracking task with a motion command.")


def is_motion_command_cfg(command: Any) -> bool:
  try:
    motion_command_source_shape(command)
  except ValueError:
    return False
  return True


def apply_motion_source_to_command(
  command: Any,
  source: ResolvedMotionSource | None,
) -> AppliedMotionSource | None:
  """Apply a resolved source to a single- or multi-motion command config."""
  if source is None:
    return None

  shape = motion_command_source_shape(command)
  if source.kind == "local_file":
    return _apply_motion_file(command, source.path, shape, source.registry_name)
  if source.kind == "local_path":
    return _apply_motion_path(command, source.path, shape, source.registry_name)

  if shape == "single":
    return _apply_motion_file(
      command,
      source.path / "motion.npz",
      shape,
      source.registry_name,
    )
  return _apply_motion_path(command, source.path, shape, source.registry_name)


def _apply_motion_file(
  command: Any,
  motion_file: Path,
  shape: MotionCommandSourceShape,
  registry_name: str | None,
) -> AppliedMotionSource:
  if not hasattr(command, "motion_file"):
    raise ValueError("This task motion command does not support `motion_file`.")
  command.motion_file = str(motion_file)
  if hasattr(command, "motion_path"):
    command.motion_path = ""
  return AppliedMotionSource(
    shape=shape,
    motion_file=str(motion_file),
    registry_name=registry_name,
  )


def _apply_motion_path(
  command: Any,
  motion_path: Path,
  shape: MotionCommandSourceShape,
  registry_name: str | None,
) -> AppliedMotionSource:
  if shape == "single" or not hasattr(command, "motion_path"):
    raise ValueError("This task motion command does not support `motion_path`.")
  command.motion_path = str(motion_path)
  if hasattr(command, "motion_file"):
    command.motion_file = ""
  return AppliedMotionSource(
    shape=shape,
    motion_path=str(motion_path),
    registry_name=registry_name,
  )


def collect_motion_files(
  motion_root: str | Path,
  *,
  shard: bool = False,
  rank: int = 0,
  world_size: int = 1,
) -> list[Path]:
  """Collect `.npz` motion files recursively with deterministic ordering."""
  motion_root_path = Path(motion_root)
  if not motion_root_path.exists():
    raise FileNotFoundError(f"Motion path not found: {motion_root}")
  if not motion_root_path.is_dir():
    raise ValueError(f"motion_path must be a directory: {motion_root}")

  motion_files = sorted(
    path
    for path in motion_root_path.rglob("*")
    if path.is_file() and path.suffix.lower() == ".npz"
  )
  if not motion_files:
    raise ValueError(f"No .npz motion files found under: {motion_root}")
  if not shard:
    return motion_files
  return shard_motion_files(motion_files, world_size=world_size, rank=rank)


def shard_motion_files(
  motion_files: list[Path],
  *,
  world_size: int,
  rank: int,
) -> list[Path]:
  if world_size <= 1:
    return motion_files
  if rank < 0 or rank >= world_size:
    raise ValueError(f"Expected rank in [0, {world_size}), got {rank}.")
  return motion_files[rank::world_size]


__all__ = [
  "AppliedMotionSource",
  "MotionCommandSourceShape",
  "MotionSourceSpec",
  "ResolvedMotionSource",
  "apply_motion_source_to_command",
  "collect_motion_files",
  "is_motion_command_cfg",
  "motion_command_source_shape",
  "normalize_wandb_registry_name",
  "resolve_motion_source",
  "shard_motion_files",
]
