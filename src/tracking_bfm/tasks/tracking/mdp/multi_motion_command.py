from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from tracking_bfm.tasks.tracking.mdp.motion_dataset import ReferenceMotionDataset

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MultiMotionCommand(CommandTerm):
  cfg: "MultiMotionCommandCfg"
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: "MultiMotionCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    motion_files = self._resolve_motion_files()
    self.motion = ReferenceMotionDataset(
      motion_files,
      self.body_indexes,
      motion_type=self.cfg.motion_type,
      device=self.device,
    )

    # 初始化状态变量
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Adaptive sampling bins are tracked per-motion on a shared global bin axis.
    # Each motion only uses the prefix indicated by bin_valid_mask.
    max_motion_length = self.motion.file_lengths.max().item()
    if self.cfg.adaptive_bin_width_steps is not None:
      self.bin_width_steps = max(int(self.cfg.adaptive_bin_width_steps), 1)
    else:
      self.bin_width_steps = max(
        int(round(float(self.cfg.adaptive_bin_width_s) / env.step_dt)), 1
      )
    self.bin_count = int(max_motion_length // self.bin_width_steps) + 1
    self.motion_bin_counts = torch.clamp(
      torch.div(
        self.motion.file_lengths + self.bin_width_steps - 1,
        self.bin_width_steps,
        rounding_mode="floor",
      ),
      min=1,
    )
    bin_indices = torch.arange(self.bin_count, device=self.device)
    self.bin_valid_mask = bin_indices.unsqueeze(0) < self.motion_bin_counts.unsqueeze(1)
    self.valid_motion_ids, self.valid_bin_ids = torch.where(self.bin_valid_mask)
    self.num_valid_motion_bins = max(int(self.valid_motion_ids.numel()), 1)
    bin_starts = bin_indices.unsqueeze(0) * self.bin_width_steps
    remaining_lengths = (self.motion.file_lengths.unsqueeze(1) - bin_starts).clamp(
      min=0
    )
    self.bin_lengths = torch.minimum(
      remaining_lengths,
      torch.full_like(remaining_lengths, self.bin_width_steps),
    )
    self.bin_lengths.masked_fill_(~self.bin_valid_mask, 0)

    valid_bin_lengths = self.bin_lengths[self.bin_valid_mask].float()
    mean_bin_length = torch.clamp(valid_bin_lengths.mean(), min=1.0)
    self.bin_weights = self.bin_lengths.float() / mean_bin_length
    if self.cfg.adaptive_sequence_length_agnostic:
      self.bin_weights = self.bin_weights / self.motion_bin_counts.unsqueeze(1).float()
    self.bin_weights.masked_fill_(~self.bin_valid_mask, 0.0)

    init_count = float(self.cfg.adaptive_init_num_failures)
    self.bin_episode_count = torch.full(
      (self.motion.num_files, self.bin_count),
      init_count,
      dtype=torch.float,
      device=self.device,
    )
    self.bin_failure_count = torch.full_like(self.bin_episode_count, init_count)
    self.bin_episode_count.masked_fill_(~self.bin_valid_mask, 0.0)
    self.bin_failure_count.masked_fill_(~self.bin_valid_mask, 0.0)
    self._init_adaptive_sampling_window()
    self._adaptive_sampling_phase = "idle"
    self._skip_current_adaptive_episode_count = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    if self.cfg.if_log_metrics:
      self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_anchor_lin_vel"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["error_anchor_ang_vel"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["sampling_uniform_prob"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_top1_prob"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_top1_ratio"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_failure_rate_mean"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_failure_rate_max"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_effective_num_bins"] = torch.zeros(
        self.num_envs, device=self.device
      )
      self.metrics["sampling_num_concentrated_bins"] = torch.zeros(
        self.num_envs, device=self.device
      )

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _resolve_motion_files(self) -> list[str]:
    """Resolve multi-motion inputs from ``motion_path`` or a single ``motion_file``."""
    motion_path = os.fspath(self.cfg.motion_path)
    motion_file = os.fspath(self.cfg.motion_file)
    if motion_path and motion_file:
      raise ValueError(
        "Provide either motion_path for multi-motion input or motion_file for a "
        "single motion, but not both."
      )

    if motion_path:
      if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Invalid motion path: {motion_path}")
      if not os.path.isdir(motion_path):
        raise ValueError(
          f"motion_path must point to a directory containing .npz files: {motion_path}"
        )
      resolved_motion_files = []
      for root, _, files in os.walk(motion_path):
        for filename in files:
          if filename.lower().endswith(".npz"):
            resolved_motion_files.append(os.path.join(root, filename))
      resolved_motion_files.sort()
    elif motion_file:
      if not os.path.exists(motion_file):
        raise FileNotFoundError(f"Invalid motion file: {motion_file}")
      if not os.path.isfile(motion_file):
        raise ValueError(f"motion_file must point to a .npz file: {motion_file}")
      resolved_motion_files = [motion_file]
    else:
      resolved_motion_files = []

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and len(resolved_motion_files) > 1:
      rank = int(os.environ.get("RANK", "0"))
      resolved_motion_files = resolved_motion_files[rank::world_size]

    if len(resolved_motion_files) == 0:
      raise ValueError(
        "No motion files found. Provide either:\n"
        "  - motion_path: path to a directory containing .npz files\n"
        "  - motion_file: path to a single .npz file"
      )
    return resolved_motion_files

  def _compute_motion_bin_indices(
    self, time_steps: torch.Tensor, motion_indices: torch.Tensor
  ) -> torch.Tensor:
    raw_bin_indices = torch.div(time_steps, self.bin_width_steps, rounding_mode="floor")
    max_bin_indices = self.motion_bin_counts[motion_indices] - 1
    return torch.minimum(raw_bin_indices, max_bin_indices)

  def _compute_failure_rate(self) -> torch.Tensor:
    failure_rate = self.bin_failure_count / torch.clamp(
      self.bin_episode_count, min=1e-12
    )
    return failure_rate.masked_fill(~self.bin_valid_mask, 0.0)

  def _init_adaptive_sampling_window(self) -> None:
    window_iterations = getattr(
      self.cfg, "adaptive_failure_rate_window_iterations", None
    )
    self._adaptive_window_episode_chunks: torch.Tensor | None = None
    self._adaptive_window_failure_chunks: torch.Tensor | None = None
    self._adaptive_window_chunk_size = 0
    self._adaptive_window_current_chunk = 0
    self._adaptive_window_base_iteration: int | None = None
    self._adaptive_window_last_logical_chunk = 0

    if window_iterations is None or int(window_iterations) <= 0:
      return

    window_iterations = max(int(window_iterations), 1)
    num_chunks = max(
      int(getattr(self.cfg, "adaptive_failure_rate_window_chunks", 40)), 1
    )
    num_chunks = min(num_chunks, window_iterations)
    self._adaptive_window_chunk_size = max(
      int(math.ceil(window_iterations / num_chunks)), 1
    )
    chunk_shape = (num_chunks, *self.bin_episode_count.shape)
    self._adaptive_window_episode_chunks = torch.zeros(
      chunk_shape,
      dtype=self.bin_episode_count.dtype,
      device=self.bin_episode_count.device,
    )
    self._adaptive_window_failure_chunks = torch.zeros_like(
      self._adaptive_window_episode_chunks
    )
    self._adaptive_window_episode_chunks[0].copy_(self.bin_episode_count)
    self._adaptive_window_failure_chunks[0].copy_(self.bin_failure_count)

  def begin_adaptive_sampling_iteration(self, iteration: int) -> None:
    if (
      self.cfg.sampling_mode != "adaptive"
      or self._adaptive_window_episode_chunks is None
      or self._adaptive_window_failure_chunks is None
    ):
      return

    if self._adaptive_window_base_iteration is None:
      self._adaptive_window_base_iteration = int(iteration)
      return

    chunk_size = max(self._adaptive_window_chunk_size, 1)
    logical_chunk = max(
      (int(iteration) - self._adaptive_window_base_iteration) // chunk_size,
      0,
    )
    if logical_chunk <= self._adaptive_window_last_logical_chunk:
      return

    num_chunks = self._adaptive_window_episode_chunks.shape[0]
    for next_logical_chunk in range(
      self._adaptive_window_last_logical_chunk + 1,
      logical_chunk + 1,
    ):
      chunk_index = next_logical_chunk % num_chunks
      self.bin_episode_count -= self._adaptive_window_episode_chunks[chunk_index]
      self.bin_failure_count -= self._adaptive_window_failure_chunks[chunk_index]
      self._adaptive_window_episode_chunks[chunk_index].zero_()
      self._adaptive_window_failure_chunks[chunk_index].zero_()
      self._adaptive_window_current_chunk = chunk_index

    self.bin_episode_count.clamp_(min=0.0)
    self.bin_failure_count.clamp_(min=0.0)
    self._adaptive_window_last_logical_chunk = logical_chunk

  def _record_adaptive_sampling_window_increments(
    self, episode_increments: torch.Tensor, failure_increments: torch.Tensor
  ) -> None:
    episode_chunks = getattr(self, "_adaptive_window_episode_chunks", None)
    failure_chunks = getattr(self, "_adaptive_window_failure_chunks", None)
    if episode_chunks is None or failure_chunks is None:
      return

    episode_chunks[self._adaptive_window_current_chunk] += episode_increments
    failure_chunks[self._adaptive_window_current_chunk] += failure_increments

  def _accumulate_adaptive_sampling_stats(
    self,
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    failure_mask: torch.Tensor | None,
  ) -> None:
    if motion_ids.numel() == 0:
      return

    current_bin_indices = self._compute_motion_bin_indices(time_steps, motion_ids)
    linear_indices = motion_ids * self.bin_count + current_bin_indices
    current_counts = torch.bincount(
      linear_indices, minlength=self.motion.num_files * self.bin_count
    ).view(self.motion.num_files, self.bin_count)
    episode_increments = current_counts.float() / torch.clamp(
      self.bin_lengths.float(), min=1.0
    )
    self.bin_episode_count += episode_increments

    failure_increments = torch.zeros_like(self.bin_failure_count)
    if failure_mask is None or not failure_mask.any():
      self._record_adaptive_sampling_window_increments(
        episode_increments, failure_increments
      )
      return

    failed_linear_indices = linear_indices[failure_mask]
    failed_counts = torch.bincount(
      failed_linear_indices, minlength=self.motion.num_files * self.bin_count
    ).view(self.motion.num_files, self.bin_count)
    failure_increments = failed_counts.float()
    self.bin_failure_count += failure_increments
    self._record_adaptive_sampling_window_increments(
      episode_increments, failure_increments
    )

  def _stage_pre_resample_adaptive_stats(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode != "adaptive" or env_ids.numel() == 0:
      return
    if self._adaptive_sampling_phase != "idle":
      return

    active_env_ids = env_ids[self._env.episode_length_buf[env_ids] > 0]
    if active_env_ids.numel() == 0:
      return

    failure_mask = self._env.termination_manager.terminated[active_env_ids]
    self._accumulate_adaptive_sampling_stats(
      self.motion_idx[active_env_ids],
      self.time_steps[active_env_ids],
      failure_mask,
    )
    self._skip_current_adaptive_episode_count[active_env_ids] = True

  def _accumulate_current_adaptive_sampling_stats(self) -> None:
    active_env_ids = torch.where(~self._skip_current_adaptive_episode_count)[0]
    self._skip_current_adaptive_episode_count.zero_()
    if active_env_ids.numel() == 0:
      return
    self._accumulate_adaptive_sampling_stats(
      self.motion_idx[active_env_ids],
      self.time_steps[active_env_ids],
      failure_mask=None,
    )

  def _resolve_probability_cap(
    self, value: float | Literal["auto"] | None, count: int
  ) -> float | None:
    if value is None:
      return None
    if value == "auto":
      if count <= 0:
        return 1.0
      return float(self.cfg.adaptive_failure_rate_max_over_mean) / float(count)
    resolved = float(value)
    if resolved <= 0.0:
      return None
    return resolved

  def _clamp_motion_time_steps(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.clamp_time_steps(motion_ids, time_steps)

  def _get_frame_indices(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.frame_indices(motion_ids, time_steps)

  def gather_reference_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.gather(field_name, motion_ids, time_steps)

  def _gather_motion_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.gather_reference_field(field_name, motion_ids, time_steps)

  def _get_reference_time_steps(self) -> torch.Tensor:
    offsets = []
    if self.cfg.history_steps > 0:
      offsets.extend(range(-self.cfg.history_steps, 0))
    offsets.append(0)
    if self.cfg.future_steps > 1:
      offsets.extend(range(1, self.cfg.future_steps))
    offset_tensor = torch.tensor(offsets, device=self.device, dtype=torch.long)
    return self.time_steps.unsqueeze(1) + offset_tensor.unsqueeze(0)

  def _apply_max_probability_constraints(
    self,
    probabilities: torch.Tensor,
    valid_motion_ids: torch.Tensor,
    num_motions: int,
  ) -> torch.Tensor:
    constrained = probabilities
    max_prob_per_bin = self._resolve_probability_cap(
      self.cfg.adaptive_max_prob_per_bin, len(probabilities)
    )
    if max_prob_per_bin is not None and len(probabilities) > 1.0 / max_prob_per_bin:
      constrained = torch.clamp(constrained, max=max_prob_per_bin)
      constrained = constrained / torch.clamp(constrained.sum(), min=1e-12)

    max_prob_per_motion = self._resolve_probability_cap(
      self.cfg.adaptive_max_prob_per_motion, num_motions
    )
    if max_prob_per_motion is not None and num_motions > 1.0 / max_prob_per_motion:
      motion_probabilities = torch.zeros(
        self.motion.num_files, dtype=constrained.dtype, device=self.device
      )
      motion_probabilities.scatter_add_(0, valid_motion_ids, constrained)
      motion_scale = torch.ones_like(motion_probabilities)
      oversized = motion_probabilities > max_prob_per_motion
      motion_scale[oversized] = max_prob_per_motion / torch.clamp(
        motion_probabilities[oversized], min=1e-12
      )
      constrained = constrained * motion_scale[valid_motion_ids]
      constrained = constrained / torch.clamp(constrained.sum(), min=1e-12)

    return constrained

  def _compute_pair_sampling_probabilities(
    self,
    valid_motion_ids: torch.Tensor,
    valid_bin_ids: torch.Tensor,
    num_motions: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    failure_rate = self._compute_failure_rate()
    valid_failure_rate = failure_rate[valid_motion_ids, valid_bin_ids]
    failure_rate_mean = valid_failure_rate.mean()
    failure_rate_upper_bound = failure_rate_mean * float(
      self.cfg.adaptive_failure_rate_max_over_mean
    )
    clipped_failure_rate = torch.clamp(
      valid_failure_rate, 0.0, failure_rate_upper_bound
    )

    clipped_sum = clipped_failure_rate.sum()
    if clipped_sum <= 0.0:
      failure_based_probabilities = torch.full(
        (len(valid_motion_ids),),
        1.0 / float(max(len(valid_motion_ids), 1)),
        dtype=torch.float,
        device=self.device,
      )
    else:
      failure_based_probabilities = clipped_failure_rate / clipped_sum

    uniform_probabilities = torch.full_like(
      failure_based_probabilities, 1.0 / float(max(len(valid_motion_ids), 1))
    )
    uniform_ratio = float(max(0.0, min(1.0, self.cfg.adaptive_uniform_ratio)))
    probabilities = (
      1.0 - uniform_ratio
    ) * failure_based_probabilities + uniform_ratio * uniform_probabilities
    probabilities = probabilities * self.bin_weights[valid_motion_ids, valid_bin_ids]
    probabilities = probabilities / torch.clamp(probabilities.sum(), min=1e-12)
    probabilities = self._apply_max_probability_constraints(
      probabilities, valid_motion_ids, num_motions
    )
    return probabilities, valid_failure_rate

  def _uniform_baseline_probabilities(
    self, motion_indices: torch.Tensor
  ) -> torch.Tensor:
    return torch.full(
      (len(motion_indices),),
      1.0 / float(self.num_valid_motion_bins),
      dtype=torch.float,
      device=self.device,
    )

  @property
  def command(self) -> torch.Tensor:
    cmd = torch.cat([self.motion_joint_pos, self.motion_joint_vel], dim=1)
    return cmd

  @property
  def command_joint_pos(self) -> torch.Tensor:
    return self.motion_joint_pos

  @property
  def command_joint_vel(self) -> torch.Tensor:
    return self.motion_joint_vel

  @property
  def command_current_joint_pos(self) -> torch.Tensor:
    return self.current_motion_joint_pos

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._gather_motion_field("joint_pos", self.motion_idx, self.time_steps)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._gather_motion_field("joint_vel", self.motion_idx, self.time_steps)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self._gather_motion_field("body_pos_w", self.motion_idx, self.time_steps)
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._gather_motion_field("body_quat_w", self.motion_idx, self.time_steps)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._gather_motion_field("body_lin_vel_w", self.motion_idx, self.time_steps)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._gather_motion_field("body_ang_vel_w", self.motion_idx, self.time_steps)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    """Anchor linear velocities with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_lin_vel = self._gather_motion_field(
      "body_lin_vel_w", self.motion_idx, reference_time_steps
    )[:, :, self.motion_anchor_body_index]
    return reference_lin_vel.reshape(self.num_envs, -1)

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    """Anchor angular velocities with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_ang_vel = self._gather_motion_field(
      "body_ang_vel_w", self.motion_idx, reference_time_steps
    )[:, :, self.motion_anchor_body_index]
    return reference_ang_vel.reshape(self.num_envs, -1)

  @property
  def anchor_projected_gravity(self) -> torch.Tensor:
    """Anchor projected gravity with history and future steps.

    Converts anchor quaternions to projected gravity vectors using the formula:
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    Shape: (num_envs, num_steps * 3) where num_steps = history_steps + 1 + (future_steps - 1)
    """
    reference_time_steps = self._get_reference_time_steps()
    anchor_quat = self._gather_motion_field(
      "body_quat_w", self.motion_idx, reference_time_steps
    )[:, :, self.motion_anchor_body_index]

    # Extract quaternion components: (w, x, y, z) format
    qw = anchor_quat[..., 0]  # (num_envs, num_steps)
    qx = anchor_quat[..., 1]  # (num_envs, num_steps)
    qy = anchor_quat[..., 2]  # (num_envs, num_steps)
    qz = anchor_quat[..., 3]  # (num_envs, num_steps)

    # Compute projected gravity for each step
    gravity_x = 2 * (-qz * qx + qw * qy)
    gravity_y = -2 * (qz * qy + qw * qx)
    gravity_z = 1 - 2 * (qw * qw + qz * qz)

    # Stack to (num_envs, num_steps, 3)
    projected_gravity = torch.stack([gravity_x, gravity_y, gravity_z], dim=-1)

    # Reshape to (num_envs, num_steps * 3)
    return projected_gravity.reshape(self.num_envs, -1)

  # Motion reference properties with history and future steps
  @property
  def motion_joint_pos(self) -> torch.Tensor:
    """Joint positions reference with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_joint_pos = self._gather_motion_field(
      "joint_pos", self.motion_idx, reference_time_steps
    )
    return reference_joint_pos.reshape(self.num_envs, -1)

  @property
  def current_motion_joint_pos(self) -> torch.Tensor:
    """Joint positions reference at current step only."""
    return self.joint_pos

  @property
  def motion_joint_vel(self) -> torch.Tensor:
    """Joint velocities reference with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_joint_vel = self._gather_motion_field(
      "joint_vel", self.motion_idx, reference_time_steps
    )
    return reference_joint_vel.reshape(self.num_envs, -1)

  @property
  def motion_anchor_pos(self) -> torch.Tensor:
    """Anchor positions reference with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_anchor_pos = self._gather_motion_field(
      "body_pos_w", self.motion_idx, reference_time_steps
    )[:, :, self.motion_anchor_body_index]
    reference_anchor_pos = (
      reference_anchor_pos + self._env.scene.env_origins[:, None, :]
    )
    return reference_anchor_pos.reshape(self.num_envs, -1)

  @property
  def motion_anchor_quat(self) -> torch.Tensor:
    """Anchor quaternions reference with history and future steps.

    Returns concatenated [history_steps, current, future_steps] if both are enabled,
    or just the enabled steps. Order: [past, current, future].
    """
    reference_time_steps = self._get_reference_time_steps()
    reference_anchor_quat = self._gather_motion_field(
      "body_quat_w", self.motion_idx, reference_time_steps
    )[:, :, self.motion_anchor_body_index]
    return reference_anchor_quat.reshape(self.num_envs, -1)

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    if not self.cfg.if_log_metrics:
      return

    # Extract current step data from multi-step properties
    # anchor_lin_vel_w and anchor_ang_vel_w contain [history_steps, current, future_steps]
    # Current step is at index: history_steps
    # Calculate total number of steps: history_steps + 1 (current) + (future_steps - 1)
    num_steps_total = self.cfg.history_steps + 1 + max(0, self.cfg.future_steps - 1)
    current_step_idx = self.cfg.history_steps

    # For anchor_lin_vel_w and anchor_ang_vel_w, extract current step
    # Reshape from (num_envs, num_steps * 3) to (num_envs, num_steps, 3) and extract current step
    if num_steps_total > 1:
      anchor_lin_vel_current = self.anchor_lin_vel_w.reshape(
        self.num_envs, num_steps_total, 3
      )[:, current_step_idx, :]
      anchor_ang_vel_current = self.anchor_ang_vel_w.reshape(
        self.num_envs, num_steps_total, 3
      )[:, current_step_idx, :]
    else:
      # No history/future, use directly (shape is already (num_envs, 3))
      anchor_lin_vel_current = self.anchor_lin_vel_w
      anchor_ang_vel_current = self.anchor_ang_vel_w

    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      anchor_lin_vel_current - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      anchor_ang_vel_current - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    sampling_probabilities, valid_failure_rate = (
      self._compute_pair_sampling_probabilities(
        self.valid_motion_ids,
        self.valid_bin_ids,
        self.motion.num_files,
      )
    )
    sampled_pair_indices = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    sampled_motion_indices = self.valid_motion_ids[sampled_pair_indices]
    sampled_bin_indices = self.valid_bin_ids[sampled_pair_indices]

    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    denom = (
      math.log(self.num_valid_motion_bins) if self.num_valid_motion_bins > 1 else 1.0
    )
    H_norm = H / denom if self.num_valid_motion_bins > 1 else 0.0
    pmax, _ = sampling_probabilities.max(dim=0)
    uniform_prob = 1.0 / float(self.num_valid_motion_bins)
    effective_num_bins = 1.0 / torch.clamp((sampling_probabilities**2).sum(), min=1e-12)
    num_concentrated_bins = (sampling_probabilities > 10.0 * uniform_prob).sum().float()
    if self.cfg.if_log_metrics:
      self.metrics["sampling_entropy"][env_ids] = H_norm
      self.metrics["sampling_uniform_prob"][env_ids] = uniform_prob
      self.metrics["sampling_top1_prob"][env_ids] = pmax
      self.metrics["sampling_top1_ratio"][env_ids] = pmax / uniform_prob
      self.metrics["sampling_failure_rate_mean"][env_ids] = valid_failure_rate.mean()
      self.metrics["sampling_failure_rate_max"][env_ids] = valid_failure_rate.max()
      self.metrics["sampling_effective_num_bins"][env_ids] = effective_num_bins
      self.metrics["sampling_num_concentrated_bins"][env_ids] = num_concentrated_bins

    self.motion_idx[env_ids] = sampled_motion_indices
    self.motion_length[env_ids] = self.motion.file_lengths[sampled_motion_indices]

    bin_starts = sampled_bin_indices * self.bin_width_steps
    bin_ends = torch.minimum(
      bin_starts + self.bin_width_steps, self.motion_length[env_ids]
    )
    bin_lengths = torch.clamp(bin_ends - bin_starts, min=1)
    offsets = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      * bin_lengths.float()
    ).long()
    self.time_steps[env_ids] = torch.minimum(
      bin_starts + offsets, self.motion_length[env_ids] - 1
    )
    if self.cfg.adaptive_pre_failure_sample_window_steps > 0:
      pre_failure_offsets = torch.randint(
        self.cfg.adaptive_pre_failure_sample_window_steps,
        (len(env_ids),),
        device=self.device,
      )
      self.time_steps[env_ids] = (
        self.time_steps[env_ids] - pre_failure_offsets
      ).clamp_min(0)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = (
      sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      * self.motion_length[env_ids]
    ).long()
    if self.cfg.if_log_metrics:
      uniform_probabilities = self._uniform_baseline_probabilities(
        self.motion_idx[env_ids]
      )
      self.metrics["sampling_entropy"][env_ids] = 1.0  # Maximum entropy for uniform.
      self.metrics["sampling_uniform_prob"][env_ids] = uniform_probabilities[
        : len(env_ids)
      ]
      self.metrics["sampling_top1_prob"][env_ids] = uniform_probabilities[
        : len(env_ids)
      ]
      self.metrics["sampling_top1_ratio"][env_ids] = 1.0
      self.metrics["sampling_failure_rate_mean"][env_ids] = 0.0
      self.metrics["sampling_failure_rate_max"][env_ids] = 0.0
      self.metrics["sampling_effective_num_bins"][env_ids] = float(
        self.num_valid_motion_bins
      )
      self.metrics["sampling_num_concentrated_bins"][env_ids] = 0.0

  def _resample_command(self, env_ids: torch.Tensor):
    if len(env_ids) == 0:
      return
    self._stage_pre_resample_adaptive_stats(env_ids)
    motion_indices = torch.randint(
      0,
      self.motion.num_files,
      (len(env_ids),),
      device=self.device,
    )
    if self.cfg.sampling_mode == "start":
      self.motion_idx[env_ids] = motion_indices
      self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
      self.time_steps[env_ids] = 0
      print(
        " ************** [FOR DEBUG] WARNING: All envs time steps is set to start initialization ! ************** "
      )

    elif self.cfg.sampling_mode == "uniform":
      self.motion_idx[env_ids] = motion_indices
      self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()
    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )

    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(
      torch.cat(
        [
          root_pos[env_ids],
          root_ori[env_ids],
          root_lin_vel[env_ids],
          root_ang_vel[env_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    if self.cfg.sampling_mode == "adaptive":
      self._adaptive_sampling_phase = "updating"
      self._accumulate_current_adaptive_sampling_stats()

    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_length)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )
    if self.cfg.sampling_mode == "adaptive":
      self._adaptive_sampling_phase = "idle"

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  """Configuration for the motion command."""

  entity_name: str
  motion_path: str = ""
  motion_file: str = ""
  motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
  anchor_body_name: str
  body_names: tuple[str, ...]
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)

  # Ref Motion: Future/History steps configuration for N-step lookahead
  future_steps: int = 5  # 1
  history_steps: int = 5  # 0

  adaptive_uniform_ratio: float = 0.1
  adaptive_bin_width_s: float = 1.0
  adaptive_bin_width_steps: int | None = None
  adaptive_init_num_failures: float = 1.0
  adaptive_failure_rate_window_iterations: int | None = None
  adaptive_failure_rate_window_chunks: int = 40
  adaptive_failure_rate_max_over_mean: float = 200.0
  adaptive_sequence_length_agnostic: bool = True
  adaptive_max_prob_per_bin: float | Literal["auto"] | None = "auto"
  adaptive_max_prob_per_motion: float | Literal["auto"] | None = "auto"
  adaptive_pre_failure_sample_window_steps: int = 200
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  # for downstream task training
  if_log_metrics: bool = True

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)


# Keep the public interface aligned with the single-motion module.
MotionCommand = MultiMotionCommand
MotionCommandCfg = MultiMotionCommandCfg
