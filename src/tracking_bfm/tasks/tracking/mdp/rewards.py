"""Tracking reward terms.

Common tracking rewards are re-exported from upstream ``mjlab``. This module
only implements BFM-specific reward additions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp.rewards import (
  motion_global_anchor_orientation_error_exp,
  motion_global_anchor_position_error_exp,
  motion_global_body_angular_velocity_error_exp,
  motion_global_body_linear_velocity_error_exp,
  motion_relative_body_orientation_error_exp,
  motion_relative_body_position_error_exp,
  self_collision_cost,
)
from mjlab.utils.lab_api.math import quat_error_magnitude, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def _pelvis_limb_ee_pose_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
  anchor_body_name: str = "pelvis",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  anchor_index = tuple(command.cfg.body_names).index(anchor_body_name)

  num_bodies = len(body_indexes)
  ref_anchor_pos_w = command.body_pos_w[:, anchor_index : anchor_index + 1, :].repeat(
    1, num_bodies, 1
  )
  ref_anchor_quat_w = command.body_quat_w[:, anchor_index : anchor_index + 1, :].repeat(
    1, num_bodies, 1
  )
  robot_anchor_pos_w = command.robot_body_pos_w[
    :, anchor_index : anchor_index + 1, :
  ].repeat(1, num_bodies, 1)
  robot_anchor_quat_w = command.robot_body_quat_w[
    :, anchor_index : anchor_index + 1, :
  ].repeat(1, num_bodies, 1)

  ref_pos_b, ref_quat_b = subtract_frame_transforms(
    ref_anchor_pos_w,
    ref_anchor_quat_w,
    command.body_pos_w[:, body_indexes],
    command.body_quat_w[:, body_indexes],
  )
  robot_pos_b, robot_quat_b = subtract_frame_transforms(
    robot_anchor_pos_w,
    robot_anchor_quat_w,
    command.robot_body_pos_w[:, body_indexes],
    command.robot_body_quat_w[:, body_indexes],
  )
  return ref_pos_b, ref_quat_b, robot_pos_b, robot_quat_b


def motion_pelvis_limb_ee_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  anchor_body_name: str = "pelvis",
) -> torch.Tensor:
  ref_pos_b, _, robot_pos_b, _ = _pelvis_limb_ee_pose_b(
    env,
    command_name=command_name,
    body_names=body_names,
    anchor_body_name=anchor_body_name,
  )
  pos_error = torch.sum(torch.square(ref_pos_b - robot_pos_b), dim=-1).mean(-1)
  return torch.exp(-pos_error / std**2)


def motion_pelvis_limb_ee_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...],
  anchor_body_name: str = "pelvis",
) -> torch.Tensor:
  _, ref_quat_b, _, robot_quat_b = _pelvis_limb_ee_pose_b(
    env,
    command_name=command_name,
    body_names=body_names,
    anchor_body_name=anchor_body_name,
  )
  ori_error = quat_error_magnitude(ref_quat_b, robot_quat_b).square().mean(-1)
  return torch.exp(-ori_error / std**2)


def motion_global_body_height_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_index = tuple(command.cfg.body_names).index(body_name)
  error = torch.square(
    command.body_pos_w[:, body_index, 2] - command.robot_body_pos_w[:, body_index, 2]
  )
  return torch.exp(-error / std**2)


def joint_action_rate_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Penalize raw action changes for selected joints in a joint action term."""
  action_manager = env.action_manager
  action_term = action_manager.get_term(action_name)
  if not hasattr(action_term, "target_ids"):
    raise ValueError(f"Action term '{action_name}' does not expose target_ids.")

  term_start = 0
  for term_name in action_manager.active_terms:
    term = action_manager.get_term(term_name)
    if term_name == action_name:
      break
    term_start += term.action_dim
  else:
    raise KeyError(f"Action term '{action_name}' not found.")

  target_ids = action_term.target_ids
  if isinstance(asset_cfg.joint_ids, slice):
    term_action_ids = torch.arange(
      action_term.action_dim, device=target_ids.device, dtype=torch.long
    )
  else:
    joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=target_ids.device)
    term_action_ids = torch.nonzero(torch.isin(target_ids, joint_ids), as_tuple=False)
    term_action_ids = term_action_ids.flatten()

  action_rate = action_manager.action - action_manager.prev_action
  action_ids = term_action_ids + term_start
  return torch.sum(torch.square(action_rate[:, action_ids]), dim=1)


__all__ = [
  "motion_global_anchor_position_error_exp",
  "motion_global_anchor_orientation_error_exp",
  "motion_relative_body_position_error_exp",
  "motion_relative_body_orientation_error_exp",
  "motion_global_body_linear_velocity_error_exp",
  "motion_global_body_angular_velocity_error_exp",
  "self_collision_cost",
  "motion_global_body_position_error_exp",
  "motion_global_body_orientation_error_exp",
  "motion_pelvis_limb_ee_position_error_exp",
  "motion_pelvis_limb_ee_orientation_error_exp",
  "motion_global_body_height_error_exp",
  "joint_action_rate_l2",
]
