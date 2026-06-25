"""Export helpers for tracking-bfm checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tracking_bfm.export.checkpoint import (
  CheckpointFamily,
  CheckpointFamilyOption,
  detect_checkpoint_family,
  ensure_output_path_available,
  resolve_onnx_output_path,
)
from tracking_bfm.export.metadata import (
  build_latent_policy_metadata,
  build_policy_metadata,
)

if TYPE_CHECKING:
  from tracking_bfm.export.onnx_latent_policy import (
    LatentTrackingOnnxModel,
    export_actor_decoder_model_to_onnx,
    export_latent_tracking_checkpoint_to_onnx,
    resolve_latent_policy_onnx_path,
  )
  from tracking_bfm.export.onnx_policy import (
    export_actor_model_to_onnx,
    export_checkpoint_to_onnx,
    resolve_policy_onnx_path,
  )

__all__ = [
  "CheckpointFamily",
  "CheckpointFamilyOption",
  "LatentTrackingOnnxModel",
  "build_latent_policy_metadata",
  "build_policy_metadata",
  "detect_checkpoint_family",
  "ensure_output_path_available",
  "export_actor_decoder_model_to_onnx",
  "export_actor_model_to_onnx",
  "export_checkpoint_to_onnx",
  "export_latent_tracking_checkpoint_to_onnx",
  "resolve_latent_policy_onnx_path",
  "resolve_onnx_output_path",
  "resolve_policy_onnx_path",
]


def __getattr__(name: str):
  if name in {
    "export_actor_model_to_onnx",
    "export_checkpoint_to_onnx",
    "resolve_policy_onnx_path",
  }:
    from tracking_bfm.export import onnx_policy

    return getattr(onnx_policy, name)
  if name in {
    "LatentTrackingOnnxModel",
    "export_actor_decoder_model_to_onnx",
    "export_latent_tracking_checkpoint_to_onnx",
    "resolve_latent_policy_onnx_path",
  }:
    from tracking_bfm.export import onnx_latent_policy

    return getattr(onnx_latent_policy, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
