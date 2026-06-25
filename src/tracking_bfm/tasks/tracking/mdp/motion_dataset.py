from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

MotionType = Literal["isaaclab", "mujoco"]

_ISAACLAB_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]

_MUJOCO_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

_ISAACLAB_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

_MUJOCO_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
]

_ISAACLAB_TO_MUJOCO_JOINT_REINDEX = [
  _ISAACLAB_JOINT_NAMES.index(name) for name in _MUJOCO_JOINT_NAMES
]
_ISAACLAB_TO_MUJOCO_BODY_REINDEX = [
  _ISAACLAB_BODY_NAMES.index(name) for name in _MUJOCO_BODY_NAMES
]


def motion_reindex(
  motion_type: MotionType,
) -> tuple[list[int] | None, list[int] | None]:
  if motion_type == "isaaclab":
    return _ISAACLAB_TO_MUJOCO_JOINT_REINDEX, _ISAACLAB_TO_MUJOCO_BODY_REINDEX
  if motion_type == "mujoco":
    return None, None
  raise ValueError(f"Unsupported motion_type: {motion_type}")


class ReferenceMotionDataset:
  def __init__(
    self,
    motion_files: Sequence[str | Path],
    body_indexes: torch.Tensor,
    motion_type: MotionType = "isaaclab",
    device: str | torch.device = "cpu",
  ) -> None:
    if len(motion_files) == 0:
      raise ValueError("motion_files cannot be empty")
    self.device = device
    self._body_indexes = body_indexes.to(device=device, dtype=torch.long)
    joint_reindex, body_reindex = motion_reindex(motion_type)

    self.num_files = len(motion_files)
    self.fps_list: list[float] = []
    file_lengths: list[int] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_lin_vel_w_list: list[torch.Tensor] = []
    body_ang_vel_w_list: list[torch.Tensor] = []

    for motion_file in motion_files:
      motion_path = Path(motion_file)
      if not motion_path.is_file():
        raise FileNotFoundError(f"Invalid motion file: {motion_path}")

      with np.load(motion_path) as data:
        self.fps_list.append(float(np.asarray(data["fps"]).item()))
        joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        body_pos_w = torch.tensor(
          data["body_pos_w"], dtype=torch.float32, device=device
        )
        body_quat_w = torch.tensor(
          data["body_quat_w"], dtype=torch.float32, device=device
        )
        body_lin_vel_w = torch.tensor(
          data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        body_ang_vel_w = torch.tensor(
          data["body_ang_vel_w"], dtype=torch.float32, device=device
        )

      if joint_reindex is not None:
        joint_pos = joint_pos[:, joint_reindex]
        joint_vel = joint_vel[:, joint_reindex]
      if body_reindex is not None:
        body_pos_w = body_pos_w[:, body_reindex, :]
        body_quat_w = body_quat_w[:, body_reindex, :]
        body_lin_vel_w = body_lin_vel_w[:, body_reindex, :]
        body_ang_vel_w = body_ang_vel_w[:, body_reindex, :]

      body_pos_w = body_pos_w[:, self._body_indexes, :]
      body_quat_w = body_quat_w[:, self._body_indexes, :]
      body_lin_vel_w = body_lin_vel_w[:, self._body_indexes, :]
      body_ang_vel_w = body_ang_vel_w[:, self._body_indexes, :]

      joint_pos_list.append(joint_pos)
      joint_vel_list.append(joint_vel)
      body_pos_w_list.append(body_pos_w)
      body_quat_w_list.append(body_quat_w)
      body_lin_vel_w_list.append(body_lin_vel_w)
      body_ang_vel_w_list.append(body_ang_vel_w)
      file_lengths.append(joint_pos.shape[0])

    self.file_lengths = torch.tensor(file_lengths, dtype=torch.long, device=device)
    self.fps = self.fps_list[0]
    self.length_starts = torch.cat(
      [
        torch.zeros(1, dtype=torch.long, device=device),
        self.file_lengths[:-1].cumsum(dim=0),
      ]
    )
    self.joint_pos = torch.cat(joint_pos_list, dim=0)
    self.joint_vel = torch.cat(joint_vel_list, dim=0)
    self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
    self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
    self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
    self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
    self.time_step_total = int(self.joint_pos.shape[0])

  def clamp_time_steps(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    max_time_steps = self.file_lengths[motion_ids] - 1
    if time_steps.ndim > 1:
      max_time_steps = max_time_steps.unsqueeze(-1)
    clamped_time_steps = torch.clamp_min(time_steps, 0)
    return torch.minimum(clamped_time_steps, max_time_steps)

  def frame_indices(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    clamped_time_steps = self.clamp_time_steps(motion_ids, time_steps)
    frame_starts = self.length_starts[motion_ids]
    if clamped_time_steps.ndim > 1:
      frame_starts = frame_starts.unsqueeze(-1)
    return frame_starts + clamped_time_steps

  def gather(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return getattr(self, field_name)[self.frame_indices(motion_ids, time_steps)]

  def get_motion_data_batch(
    self, motion_idx: int, time_steps_start: torch.Tensor, time_steps_end: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    time_steps_tensor = torch.arange(
      time_steps_start.item(),
      time_steps_end.item(),
      device=self.file_lengths.device,
      dtype=torch.long,
    )
    time_steps_tensor = torch.clamp(
      time_steps_tensor,
      torch.tensor(0, device=self.file_lengths.device),
      self.file_lengths[motion_idx] - 1,
    )
    frame_indices = self.length_starts[motion_idx] + time_steps_tensor
    return {
      "joint_pos": self.joint_pos[frame_indices],
      "joint_vel": self.joint_vel[frame_indices],
      "body_pos_w": self.body_pos_w[frame_indices],
      "body_quat_w": self.body_quat_w[frame_indices],
      "body_lin_vel_w": self.body_lin_vel_w[frame_indices],
      "body_ang_vel_w": self.body_ang_vel_w[frame_indices],
    }


__all__ = [
  "MotionType",
  "ReferenceMotionDataset",
  "motion_reindex",
]
