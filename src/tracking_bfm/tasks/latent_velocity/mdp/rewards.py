"""Velocity reward wrappers needed by the latent velocity BFM task."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  penalize_z_velocity: bool = True,
) -> torch.Tensor:
  """Track commanded base linear velocity with optional z-velocity penalty."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  if penalize_z_velocity:
    z_error = torch.square(actual[:, 2])
  else:
    z_error = torch.zeros_like(xy_error)
  return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  penalize_xy_angular_velocity: bool = True,
) -> torch.Tensor:
  """Track commanded yaw velocity with optional roll/pitch velocity penalty."""
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  if not penalize_xy_angular_velocity:
    xy_error = torch.zeros_like(z_error)
  return torch.exp(-(z_error + xy_error) / std**2)
