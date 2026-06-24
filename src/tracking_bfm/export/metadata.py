"""Metadata helpers for exported BFM ONNX policies."""

from __future__ import annotations

from pathlib import Path

from .checkpoint import CheckpointFamily


def _drop_none(metadata: dict[str, str | None]) -> dict[str, str]:
  return {key: value for key, value in metadata.items() if value is not None}


def build_policy_metadata(
  *,
  task_id: str,
  obs_group: str,
  checkpoint_family: CheckpointFamily,
  robot_name: str | None = None,
) -> dict[str, str]:
  """Build string-only metadata for a direct policy export."""
  return _drop_none(
    {
      "task_id": task_id,
      "obs_group": obs_group,
      "checkpoint_family": checkpoint_family,
      "robot_name": robot_name,
    }
  )


def build_latent_policy_metadata(
  *,
  task_id: str,
  decoder_checkpoint_path: str | Path,
  obs_group: str,
  proprio_obs_group: str,
  robot_name: str | None = None,
) -> dict[str, str]:
  """Build string-only metadata for a latent actor plus decoder export."""
  return _drop_none(
    {
      "task_id": task_id,
      "checkpoint_family": "latent_tracking",
      "decoder_checkpoint": str(decoder_checkpoint_path),
      "obs_group": obs_group,
      "proprio_obs_group": proprio_obs_group,
      "robot_name": robot_name,
    }
  )


__all__ = ["build_latent_policy_metadata", "build_policy_metadata"]
