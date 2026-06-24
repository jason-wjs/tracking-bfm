"""Observation terms for the G1 BFM wbteleop tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_LIMB_EE_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
_ANCHOR_BODY_NAME = "pelvis"


def _body_indices(command: MotionCommand, body_names: tuple[str, ...]) -> list[int]:
  command_body_names = tuple(command.cfg.body_names)
  return [command_body_names.index(name) for name in body_names]


def _body_index(command: MotionCommand, body_name: str) -> int:
  return tuple(command.cfg.body_names).index(body_name)


def _reference_time_window(
  command: MotionCommand,
  *,
  history_steps: int,
  future_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  if int(history_steps) == 0 and int(future_steps) == 1:
    return command.body_pos_w.unsqueeze(1), command.body_quat_w.unsqueeze(1)
  if not hasattr(command, "_gather_motion_field") or not hasattr(command, "time_steps"):
    raise NotImplementedError(
      "Reference limb ee pose observations with history/future require a motion "
      "command that supports reference field gathering."
    )

  history_steps = int(history_steps)
  future_steps = int(future_steps)
  offsets = list(range(-history_steps, 0))
  offsets.append(0)
  offsets.extend(range(1, future_steps))
  offsets_tensor = torch.tensor(offsets, device=command.time_steps.device, dtype=torch.long)
  reference_time_steps = command.time_steps.unsqueeze(1) + offsets_tensor.unsqueeze(0)
  body_pos_w = command._gather_motion_field(
    "body_pos_w", command.motion_idx, reference_time_steps
  )
  body_quat_w = command._gather_motion_field(
    "body_quat_w", command.motion_idx, reference_time_steps
  )
  if hasattr(command, "_env"):
    body_pos_w = body_pos_w + command._env.scene.env_origins[:, None, None, :]
  return body_pos_w, body_quat_w


def _limb_pose_in_anchor_frame(
  *,
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_names: tuple[str, ...],
  anchor_body_name: str,
) -> torch.Tensor:
  body_indexes = _body_indices(command, body_names)
  anchor_body_index = _body_index(command, anchor_body_name)
  anchor_pos_w = body_pos_w[:, :, anchor_body_index : anchor_body_index + 1, :].repeat(
    1, 1, len(body_indexes), 1
  )
  anchor_quat_w = body_quat_w[:, :, anchor_body_index : anchor_body_index + 1, :].repeat(
    1, 1, len(body_indexes), 1
  )
  limb_pos_w = body_pos_w[:, :, body_indexes, :]
  limb_quat_w = body_quat_w[:, :, body_indexes, :]

  pos_b, quat_b = subtract_frame_transforms(
    anchor_pos_w, anchor_quat_w, limb_pos_w, limb_quat_w
  )
  rot6d = matrix_from_quat(quat_b)[..., :2].reshape(
    env.num_envs, -1, len(body_indexes), 6
  )
  return torch.cat([pos_b, rot6d], dim=-1).reshape(env.num_envs, -1)


def motion_ref_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return reference anchor angular velocity from the motion command window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_ang_vel_w


def ref_limb_ee_pose_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] = _LIMB_EE_BODY_NAMES,
  anchor_body_name: str = _ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_pos_w, body_quat_w = _reference_time_window(
    command, history_steps=history_steps, future_steps=future_steps
  )
  return _limb_pose_in_anchor_frame(
    env=env,
    command=command,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_names=body_names,
    anchor_body_name=anchor_body_name,
  )


def robot_limb_ee_pose_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...] = _LIMB_EE_BODY_NAMES,
  anchor_body_name: str = _ANCHOR_BODY_NAME,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _limb_pose_in_anchor_frame(
    env=env,
    command=command,
    body_pos_w=command.robot_body_pos_w.unsqueeze(1),
    body_quat_w=command.robot_body_quat_w.unsqueeze(1),
    body_names=body_names,
    anchor_body_name=anchor_body_name,
  )
