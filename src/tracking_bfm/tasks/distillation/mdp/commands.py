from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_COMMAND_NAME = "motion"
_DEFAULT_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_DEFAULT_STUDENT_ANCHOR_BODY_NAME = "pelvis"
_LEFT_EE_COLOR = (0.18, 0.71, 0.98, 1.0)
_RIGHT_EE_COLOR = (1.0, 0.45, 0.24, 1.0)
_LEFT_EE_PELVIS_COLOR = (0.18, 0.71, 0.98, 0.45)
_RIGHT_EE_PELVIS_COLOR = (1.0, 0.45, 0.24, 0.45)
_BASE_LIN_VEL_COLOR = (0.16, 0.85, 0.66, 1.0)
_BASE_ANG_VEL_COLOR = (0.98, 0.71, 0.18, 1.0)
_BASE_HEIGHT_COLOR = (0.92, 0.89, 0.32, 0.85)


def _get_command(env: ManagerBasedRlEnv, command_name: str):
  return env.command_manager.get_term(command_name)


def _get_body_indexes(command, ee_body_names: tuple[str, str]) -> list[int]:
  body_names = tuple(command.cfg.body_names)
  return [body_names.index(name) for name in ee_body_names]


def _get_body_index(command, body_name: str) -> int:
  return tuple(command.cfg.body_names).index(body_name)


def _extract_current_step_velocity(command, values: torch.Tensor) -> torch.Tensor:
  history_steps = int(getattr(command.cfg, "history_steps", 0))
  future_steps = int(getattr(command.cfg, "future_steps", 1))
  num_steps_total = history_steps + 1 + max(0, future_steps - 1)
  if values.shape[-1] == 3 or num_steps_total <= 1:
    return values
  return values.reshape(values.shape[0], num_steps_total, 3)[:, history_steps, :]


def _get_student_command_offsets(
  history_steps: int,
  future_steps: int,
  device: torch.device | str,
) -> torch.Tensor:
  history_steps = int(history_steps)
  future_steps = int(future_steps)
  if history_steps < 0:
    raise ValueError(f"history_steps must be non-negative, got {history_steps}")
  if future_steps < 1:
    raise ValueError(f"future_steps must be at least 1, got {future_steps}")

  offsets = list(range(-history_steps, 0))
  offsets.append(0)
  offsets.extend(range(1, future_steps))
  return torch.tensor(offsets, device=device, dtype=torch.long)


def _gather_student_body_field(
  command,
  field_name: str,
  history_steps: int,
  future_steps: int,
) -> torch.Tensor:
  if int(history_steps) == 0 and int(future_steps) == 1:
    values = getattr(command, field_name)
    return values.unsqueeze(1)

  if not hasattr(command, "_gather_motion_field") or not hasattr(command, "motion_idx"):
    raise NotImplementedError(
      "Student multi-step command observations require a motion command that "
      "supports reference field gathering."
    )
  if not hasattr(command, "time_steps"):
    raise NotImplementedError(
      "Student multi-step command observations require a motion command that "
      "tracks per-env reference time steps."
    )

  offsets = _get_student_command_offsets(
    history_steps,
    future_steps,
    device=command.time_steps.device,
  )
  reference_time_steps = command.time_steps.unsqueeze(1) + offsets.unsqueeze(0)
  values = command._gather_motion_field(
    field_name, command.motion_idx, reference_time_steps
  )
  if field_name == "body_pos_w":
    values = values + command._env.scene.env_origins[:, None, None, :]
  return values


def student_ee_pose_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  ee_body_names: tuple[str, str],
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  body_indexes = _get_body_indexes(command, ee_body_names)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_pos_w = _gather_student_body_field(
    command,
    "body_pos_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  body_quat_w = _gather_student_body_field(
    command,
    "body_quat_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  anchor_pos = body_pos_w[:, :, anchor_body_index : anchor_body_index + 1, :].repeat(
    1, 1, len(body_indexes), 1
  )
  anchor_quat = body_quat_w[:, :, anchor_body_index : anchor_body_index + 1, :].repeat(
    1, 1, len(body_indexes), 1
  )
  body_pos = body_pos_w[:, :, body_indexes, :]
  body_quat = body_quat_w[:, :, body_indexes, :]

  pos_b, quat_b = subtract_frame_transforms(
    anchor_pos, anchor_quat, body_pos, body_quat
  )
  rot6d = matrix_from_quat(quat_b)[..., :2].reshape(
    env.num_envs, -1, len(body_indexes), 6
  )
  return torch.cat([pos_b, rot6d], dim=-1).reshape(env.num_envs, -1)


def student_base_lin_vel_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_lin_vel_w = _gather_student_body_field(
    command,
    "body_lin_vel_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  return body_lin_vel_w[:, :, anchor_body_index, :].reshape(env.num_envs, -1)


def student_base_ang_vel_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_ang_vel_w = _gather_student_body_field(
    command,
    "body_ang_vel_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  return body_ang_vel_w[:, :, anchor_body_index, :].reshape(env.num_envs, -1)


def student_base_lin_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_lin_vel_w = _gather_student_body_field(
    command,
    "body_lin_vel_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  body_quat_w = _gather_student_body_field(
    command,
    "body_quat_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  anchor_lin_vel_w = body_lin_vel_w[:, :, anchor_body_index, :]
  anchor_quat_w = body_quat_w[:, :, anchor_body_index, :]
  anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
  return anchor_lin_vel_b.reshape(env.num_envs, -1)


def student_base_ang_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_ang_vel_w = _gather_student_body_field(
    command,
    "body_ang_vel_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  body_quat_w = _gather_student_body_field(
    command,
    "body_quat_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  anchor_ang_vel_w = body_ang_vel_w[:, :, anchor_body_index, :]
  anchor_quat_w = body_quat_w[:, :, anchor_body_index, :]
  anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
  return anchor_ang_vel_b.reshape(env.num_envs, -1)


def student_anchor_height_w(
  env: ManagerBasedRlEnv,
  command_name: str,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  body_pos_w = _gather_student_body_field(
    command,
    "body_pos_w",
    history_steps=history_steps,
    future_steps=future_steps,
  )
  return body_pos_w[:, :, anchor_body_index, 2:3].reshape(env.num_envs, -1)


def student_sparse_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  ee_body_names: tuple[str, str],
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  history_steps: int = 0,
  future_steps: int = 1,
) -> torch.Tensor:
  return torch.cat(
    [
      student_ee_pose_b(
        env,
        command_name=command_name,
        ee_body_names=ee_body_names,
        anchor_body_name=anchor_body_name,
        history_steps=history_steps,
        future_steps=future_steps,
      ),
      student_base_lin_vel_w(
        env,
        command_name=command_name,
        anchor_body_name=anchor_body_name,
        history_steps=history_steps,
        future_steps=future_steps,
      ),
      student_base_ang_vel_w(
        env,
        command_name=command_name,
        anchor_body_name=anchor_body_name,
        history_steps=history_steps,
        future_steps=future_steps,
      ),
      student_anchor_height_w(
        env,
        command_name=command_name,
        anchor_body_name=anchor_body_name,
        history_steps=history_steps,
        future_steps=future_steps,
      ),
    ],
    dim=-1,
  )


def get_student_ee_reference_w(
  env: ManagerBasedRlEnv,
  command_name: str = _DEFAULT_COMMAND_NAME,
  ee_body_names: tuple[str, str] = _DEFAULT_EE_BODY_NAMES,
) -> torch.Tensor:
  command = _get_command(env, command_name)
  body_indexes = _get_body_indexes(command, ee_body_names)
  return command.body_pos_w[:, body_indexes, :]


def get_student_ee_reference_robot_pelvis_w(
  env: ManagerBasedRlEnv,
  command_name: str = _DEFAULT_COMMAND_NAME,
  ee_body_names: tuple[str, str] = _DEFAULT_EE_BODY_NAMES,
  pelvis_body_name: str = "pelvis",
) -> torch.Tensor:
  command = _get_command(env, command_name)
  pelvis_index = tuple(command.cfg.body_names).index(pelvis_body_name)
  ee_positions_w = get_student_ee_reference_w(
    env, command_name=command_name, ee_body_names=ee_body_names
  )
  ref_pelvis_pos_w = command.body_pos_w[:, pelvis_index : pelvis_index + 1, :]
  ref_pelvis_quat_w = command.body_quat_w[:, pelvis_index : pelvis_index + 1, :]
  ee_rel_pelvis, _ = subtract_frame_transforms(
    ref_pelvis_pos_w.repeat(1, ee_positions_w.shape[1], 1),
    ref_pelvis_quat_w.repeat(1, ee_positions_w.shape[1], 1),
    ee_positions_w,
    command.body_quat_w[:, _get_body_indexes(command, ee_body_names), :],
  )

  robot = env.scene["robot"]
  robot_pelvis_index = robot.body_names.index(pelvis_body_name)
  robot_pelvis_pos_w = robot.data.body_link_pos_w[
    :, robot_pelvis_index : robot_pelvis_index + 1, :
  ]
  robot_pelvis_quat_w = robot.data.body_link_quat_w[
    :, robot_pelvis_index : robot_pelvis_index + 1, :
  ]
  return robot_pelvis_pos_w + quat_apply(
    robot_pelvis_quat_w.repeat(1, ee_rel_pelvis.shape[1], 1),
    ee_rel_pelvis,
  )


def get_student_base_lin_vel_reference_w(
  env: ManagerBasedRlEnv,
  command_name: str = _DEFAULT_COMMAND_NAME,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
) -> torch.Tensor:
  return student_base_lin_vel_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )


def get_student_base_ang_vel_reference_w(
  env: ManagerBasedRlEnv,
  command_name: str = _DEFAULT_COMMAND_NAME,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
) -> torch.Tensor:
  return student_base_ang_vel_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )


def get_student_base_height_reference_w(
  env: ManagerBasedRlEnv,
  command_name: str = _DEFAULT_COMMAND_NAME,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
) -> tuple[torch.Tensor, torch.Tensor]:
  command = _get_command(env, command_name)
  anchor_body_index = _get_body_index(command, anchor_body_name)
  anchor_pos = command.body_pos_w[:, anchor_body_index, :]
  robot = env.scene["robot"]
  robot_anchor_index = robot.body_names.index(anchor_body_name)
  base_pos = robot.data.body_link_pos_w[:, robot_anchor_index, :].clone()
  base_pos[:, 2] = 0.0
  anchor_vis_pos = base_pos.clone()
  anchor_vis_pos[:, 2] = anchor_pos[:, 2]
  return base_pos, anchor_vis_pos


def draw_student_ee_reference_robot_pelvis(
  visualizer: DebugVisualizer,
  env_idx: int,
  ee_positions_w: torch.Tensor,
  radius: float,
) -> None:
  visualizer.add_sphere(
    center=ee_positions_w[0].detach().cpu().numpy(),
    radius=radius,
    color=_LEFT_EE_PELVIS_COLOR,
    label=f"student_ref_left_ee_robot_pelvis_{env_idx}",
  )
  visualizer.add_sphere(
    center=ee_positions_w[1].detach().cpu().numpy(),
    radius=radius,
    color=_RIGHT_EE_PELVIS_COLOR,
    label=f"student_ref_right_ee_robot_pelvis_{env_idx}",
  )


def draw_student_base_velocity_reference(
  visualizer: DebugVisualizer,
  env_idx: int,
  anchor_pos_w: torch.Tensor,
  base_lin_vel_w: torch.Tensor,
  base_ang_vel_w: torch.Tensor,
  lin_vel_scale: float,
  ang_vel_scale: float,
) -> None:
  start = anchor_pos_w.detach().cpu().numpy()
  visualizer.add_arrow(
    start=start,
    end=(anchor_pos_w + base_lin_vel_w * lin_vel_scale).detach().cpu().numpy(),
    color=_BASE_LIN_VEL_COLOR,
    width=0.01,
    label=f"student_ref_base_lin_vel_{env_idx}",
  )
  visualizer.add_arrow(
    start=start,
    end=(anchor_pos_w + base_ang_vel_w * ang_vel_scale).detach().cpu().numpy(),
    color=_BASE_ANG_VEL_COLOR,
    width=0.01,
    label=f"student_ref_base_ang_vel_{env_idx}",
  )


def draw_student_base_height_reference(
  visualizer: DebugVisualizer,
  env_idx: int,
  base_start_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  radius: float,
) -> None:
  visualizer.add_cylinder(
    start=base_start_w.detach().cpu().numpy(),
    end=anchor_pos_w.detach().cpu().numpy(),
    radius=radius,
    color=_BASE_HEIGHT_COLOR,
    label=f"student_ref_base_height_{env_idx}",
  )


def debug_vis_student_sparse_command(
  env: ManagerBasedRlEnv,
  visualizer: DebugVisualizer,
  command_name: str = _DEFAULT_COMMAND_NAME,
  ee_body_names: tuple[str, str] = _DEFAULT_EE_BODY_NAMES,
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME,
  ee_sphere_radius: float = 0.025,
  lin_vel_scale: float = 0.2,
  ang_vel_scale: float = 0.12,
  height_radius: float = 0.01,
) -> None:
  env_indices = visualizer.get_env_indices(env.num_envs)
  if not env_indices:
    return

  ee_positions_robot_pelvis_w = get_student_ee_reference_robot_pelvis_w(
    env, command_name=command_name, ee_body_names=ee_body_names
  )
  _, anchor_pos_w = get_student_base_height_reference_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )
  base_lin_vel_w = get_student_base_lin_vel_reference_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )
  base_ang_vel_w = get_student_base_ang_vel_reference_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )
  base_start_w, base_end_w = get_student_base_height_reference_w(
    env, command_name=command_name, anchor_body_name=anchor_body_name
  )

  for env_idx in env_indices:
    draw_student_ee_reference_robot_pelvis(
      visualizer=visualizer,
      env_idx=env_idx,
      ee_positions_w=ee_positions_robot_pelvis_w[env_idx],
      radius=ee_sphere_radius,
    )
    draw_student_base_velocity_reference(
      visualizer=visualizer,
      env_idx=env_idx,
      anchor_pos_w=anchor_pos_w[env_idx],
      base_lin_vel_w=base_lin_vel_w[env_idx],
      base_ang_vel_w=base_ang_vel_w[env_idx],
      lin_vel_scale=lin_vel_scale,
      ang_vel_scale=ang_vel_scale,
    )
    draw_student_base_height_reference(
      visualizer=visualizer,
      env_idx=env_idx,
      base_start_w=base_start_w[env_idx],
      anchor_pos_w=base_end_w[env_idx],
      radius=height_radius,
    )


class StudentSparseCommandVis(CommandTerm):
  cfg: "StudentSparseCommandVisCfg"

  def __init__(self, cfg: "StudentSparseCommandVisCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  @property
  def command(self) -> torch.Tensor:
    return student_sparse_command(
      self._env,
      command_name=self.cfg.command_name,
      ee_body_names=self.cfg.ee_body_names,
      anchor_body_name=self.cfg.anchor_body_name,
      history_steps=0,
      future_steps=1,
    )

  def _update_metrics(self) -> None:
    return None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids
    return None

  def _update_command(self) -> None:
    return None

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    debug_vis_student_sparse_command(
      self._env,
      visualizer=visualizer,
      command_name=self.cfg.command_name,
      ee_body_names=self.cfg.ee_body_names,
      anchor_body_name=self.cfg.anchor_body_name,
      ee_sphere_radius=self.cfg.ee_sphere_radius,
      lin_vel_scale=self.cfg.lin_vel_scale,
      ang_vel_scale=self.cfg.ang_vel_scale,
      height_radius=self.cfg.height_radius,
    )


@dataclass(kw_only=True)
class StudentSparseCommandVisCfg(CommandTermCfg):
  command_name: str = _DEFAULT_COMMAND_NAME
  ee_body_names: tuple[str, str] = _DEFAULT_EE_BODY_NAMES
  anchor_body_name: str = _DEFAULT_STUDENT_ANCHOR_BODY_NAME
  lin_vel_scale: float = 0.2
  ang_vel_scale: float = 0.12
  ee_sphere_radius: float = 0.025
  height_radius: float = 0.01

  @dataclass
  class VizCfg:
    reserved: str = "student_sparse"

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> StudentSparseCommandVis:
    return StudentSparseCommandVis(self, env)
