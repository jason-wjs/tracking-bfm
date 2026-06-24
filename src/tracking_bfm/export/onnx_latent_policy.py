"""ONNX export for latent tracking actors with frozen decoders."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
  import torch
  from torch import nn
except ModuleNotFoundError:
  torch = None

  class _MissingTorchModule:
    pass

  class _MissingTorchNN:
    Module = _MissingTorchModule

  nn = _MissingTorchNN()

from tracking_bfm.export.checkpoint import (
  ensure_output_path_available,
  resolve_onnx_output_path,
)
from tracking_bfm.export.metadata import build_latent_policy_metadata
from tracking_bfm.export.onnx_policy import _apply_motion_source


def resolve_latent_policy_onnx_path(
  checkpoint_path: str | Path,
  output_name: str | Path | None = None,
) -> Path:
  """Resolve the latent policy ONNX path beside the actor checkpoint."""
  return resolve_onnx_output_path(checkpoint_path, output_name=output_name)


class LatentTrackingOnnxModel(nn.Module):
  """ONNX-friendly model combining latent actor and frozen decoder."""

  def __init__(
    self,
    *,
    actor: nn.Module,
    decoder: nn.Module,
    latent_action_clip: float,
    verbose: bool = False,
  ) -> None:
    if torch is None:
      raise ModuleNotFoundError("torch")
    super().__init__()
    actor_onnx: Any = actor.as_onnx(verbose=verbose)  # type: ignore[attr-defined]
    decoder_module: Any = decoder.decoder  # type: ignore[attr-defined]
    decoder_onnx: Any = decoder_module.as_onnx(verbose=verbose)
    self.actor_onnx = actor_onnx
    self.decoder_onnx = decoder_onnx
    self.latent_action_clip = float(latent_action_clip)

    self.actor_input_size = int(actor_onnx.input_size)
    self.latent_dim = int(decoder.latent_dim)
    self.decoder_input_size = int(decoder_onnx.input_size)
    self.proprio_input_size = self.decoder_input_size - self.latent_dim
    if self.proprio_input_size <= 0:
      raise ValueError(
        "Decoder input size must be larger than latent_dim; got "
        f"decoder_input_size={self.decoder_input_size}, latent_dim={self.latent_dim}."
      )

    with torch.no_grad():
      latent = self.actor_onnx(torch.zeros(1, self.actor_input_size))
    if int(latent.shape[-1]) != self.latent_dim:
      raise ValueError(
        "Actor latent output dimension does not match decoder latent_dim: "
        f"actor={int(latent.shape[-1])}, decoder={self.latent_dim}."
      )

  def forward(self, obs: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    latent_actions = self.actor_onnx(obs)
    latent_actions = torch.clamp(
      latent_actions,
      min=-self.latent_action_clip,
      max=self.latent_action_clip,
    )
    decoder_input = torch.cat([proprio, latent_actions], dim=-1)
    return self.decoder_onnx(decoder_input)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return representative dummy inputs for ONNX tracing."""
    return (
      torch.zeros(1, self.actor_input_size),
      torch.zeros(1, self.proprio_input_size),
    )

  @property
  def input_names(self) -> list[str]:
    """Return ONNX input tensor names."""
    return ["obs", "proprio"]

  @property
  def output_names(self) -> list[str]:
    """Return ONNX output tensor names."""
    return ["actions"]


def export_actor_decoder_model_to_onnx(
  *,
  actor: nn.Module,
  decoder: nn.Module,
  checkpoint_path: str | Path,
  decoder_checkpoint_path: str | Path,
  task_id: str,
  obs_group: str,
  proprio_obs_group: str,
  latent_action_clip: float,
  output_name: str | Path | None = None,
  robot_name: str | None = None,
  overwrite: bool = False,
  verbose: bool = False,
) -> Path:
  """Export an already rebuilt latent actor and decoder as one ONNX model."""
  onnx_path = ensure_output_path_available(
    resolve_latent_policy_onnx_path(checkpoint_path, output_name=output_name),
    overwrite=overwrite,
  )
  if torch is None:
    raise ModuleNotFoundError("torch")
  model = LatentTrackingOnnxModel(
    actor=actor,
    decoder=decoder,
    latent_action_clip=latent_action_clip,
    verbose=verbose,
  )
  model.to("cpu")
  model.eval()

  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )

  from mjlab.rl.exporter_utils import attach_metadata_to_onnx

  attach_metadata_to_onnx(
    str(onnx_path),
    build_latent_policy_metadata(
      task_id=task_id,
      decoder_checkpoint_path=decoder_checkpoint_path,
      obs_group=obs_group,
      proprio_obs_group=proprio_obs_group,
      robot_name=robot_name,
    ),
  )
  return onnx_path


def _build_actor_and_decoder_from_checkpoints(
  *,
  checkpoint_path: str | Path,
  decoder_checkpoint_path: str | Path,
  task_id: str,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  latent_action_clip: float | None = None,
  device: str = "cpu",
) -> tuple[nn.Module, nn.Module, float, dict[str, Any]]:
  """Rebuild the latent actor and its frozen decoder from training checkpoints."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import tracking_bfm.tasks  # noqa: F401
  from tracking_bfm.tasks.latent_tracking.rl import LatentTrackingOnPolicyRunner

  env = None
  try:
    env_cfg = load_env_cfg(task_id, play=True)
    _apply_motion_source(
      env_cfg,
      motion_path=motion_path,
      motion_file=motion_file,
    )
    runner_cfg = asdict(load_rl_cfg(task_id))
    runner_cfg["latent_decoder_checkpoint_path"] = str(decoder_checkpoint_path)
    if latent_action_clip is not None:
      runner_cfg["latent_action_clip"] = float(latent_action_clip)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    if not issubclass(runner_cls, LatentTrackingOnPolicyRunner):
      raise ValueError(
        "Latent tracking export requires a LatentTrackingOnPolicyRunner task, "
        f"got {runner_cls.__name__} for task_id={task_id!r}."
      )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=device)
    runner.load(str(checkpoint_path), map_location=device)

    actor = runner.alg.get_policy()
    decoder = runner.env.decoder  # type: ignore[attr-defined]
    actor.to("cpu")
    actor.eval()
    decoder.to("cpu")
    decoder.eval()
    for param in decoder.parameters():
      param.requires_grad_(False)

    return actor, decoder, float(runner_cfg["latent_action_clip"]), runner_cfg
  finally:
    if env is not None:
      env.close()


def export_latent_tracking_checkpoint_to_onnx(
  *,
  checkpoint_path: str | Path,
  decoder_checkpoint_path: str | Path,
  task_id: str,
  obs_group: str = "actor",
  proprio_obs_group: str = "proprio_actor",
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  latent_action_clip: float | None = None,
  output_name: str | Path | None = None,
  robot_name: str | None = None,
  overwrite: bool = False,
  device: str = "cpu",
  verbose: bool = False,
) -> Path:
  """Export a latent tracking checkpoint plus decoder checkpoint to ONNX."""
  checkpoint_path = Path(checkpoint_path)
  ensure_output_path_available(
    resolve_latent_policy_onnx_path(checkpoint_path, output_name=output_name),
    overwrite=overwrite,
  )
  actor, decoder, resolved_clip, _ = _build_actor_and_decoder_from_checkpoints(
    checkpoint_path=checkpoint_path,
    decoder_checkpoint_path=decoder_checkpoint_path,
    task_id=task_id,
    motion_path=motion_path,
    motion_file=motion_file,
    latent_action_clip=latent_action_clip,
    device=device,
  )
  return export_actor_decoder_model_to_onnx(
    actor=actor,
    decoder=decoder,
    checkpoint_path=checkpoint_path,
    decoder_checkpoint_path=decoder_checkpoint_path,
    task_id=task_id,
    obs_group=obs_group,
    proprio_obs_group=proprio_obs_group,
    latent_action_clip=resolved_clip,
    output_name=output_name,
    robot_name=robot_name,
    overwrite=overwrite,
    verbose=verbose,
  )


__all__ = [
  "LatentTrackingOnnxModel",
  "export_actor_decoder_model_to_onnx",
  "export_latent_tracking_checkpoint_to_onnx",
  "resolve_latent_policy_onnx_path",
]
