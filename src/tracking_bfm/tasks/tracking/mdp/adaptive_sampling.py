from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, kw_only=True)
class AdaptiveSamplingConfig:
  bin_width_steps: int
  uniform_ratio: float = 0.1
  init_num_failures: float = 1.0
  failure_rate_window_iterations: int | None = None
  failure_rate_window_chunks: int = 40
  failure_rate_max_over_mean: float = 200.0
  sequence_length_agnostic: bool = True
  max_prob_per_bin: float | Literal["auto"] | None = "auto"
  max_prob_per_motion: float | Literal["auto"] | None = "auto"


class AdaptiveSamplingState:
  def __init__(
    self,
    motion_lengths: torch.Tensor,
    cfg: AdaptiveSamplingConfig,
  ) -> None:
    self.cfg = cfg
    self.motion_lengths = motion_lengths.to(dtype=torch.long)
    self.num_motions = int(self.motion_lengths.numel())
    if self.num_motions <= 0:
      raise ValueError("Adaptive sampling requires at least one motion length.")

    self.bin_width_steps = max(int(cfg.bin_width_steps), 1)
    max_motion_length = self.motion_lengths.max().item()
    self.bin_count = int(max_motion_length // self.bin_width_steps) + 1
    self.motion_bin_counts = torch.clamp(
      torch.div(
        self.motion_lengths + self.bin_width_steps - 1,
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
    remaining_lengths = (self.motion_lengths.unsqueeze(1) - bin_starts).clamp(min=0)
    self.bin_lengths = torch.minimum(
      remaining_lengths,
      torch.full_like(remaining_lengths, self.bin_width_steps),
    )
    self.bin_lengths.masked_fill_(~self.bin_valid_mask, 0)

    valid_bin_lengths = self.bin_lengths[self.bin_valid_mask].float()
    mean_bin_length = torch.clamp(valid_bin_lengths.mean(), min=1.0)
    self.bin_weights = self.bin_lengths.float() / mean_bin_length
    if cfg.sequence_length_agnostic:
      self.bin_weights = self.bin_weights / self.motion_bin_counts.unsqueeze(1).float()
    self.bin_weights.masked_fill_(~self.bin_valid_mask, 0.0)

    init_count = float(cfg.init_num_failures)
    self.bin_episode_count = torch.full(
      (self.num_motions, self.bin_count),
      init_count,
      dtype=torch.float,
      device=self.device,
    )
    self.bin_failure_count = torch.full_like(self.bin_episode_count, init_count)
    self.bin_episode_count.masked_fill_(~self.bin_valid_mask, 0.0)
    self.bin_failure_count.masked_fill_(~self.bin_valid_mask, 0.0)

    self._init_window()

  @property
  def device(self) -> torch.device:
    return self.motion_lengths.device

  def compute_motion_bin_indices(
    self, time_steps: torch.Tensor, motion_ids: torch.Tensor
  ) -> torch.Tensor:
    raw_bin_indices = torch.div(time_steps, self.bin_width_steps, rounding_mode="floor")
    max_bin_indices = self.motion_bin_counts[motion_ids] - 1
    return torch.minimum(raw_bin_indices, max_bin_indices)

  def failure_rate(self) -> torch.Tensor:
    failure_rate = self.bin_failure_count / torch.clamp(
      self.bin_episode_count, min=1e-12
    )
    return failure_rate.masked_fill(~self.bin_valid_mask, 0.0)

  def begin_iteration(self, iteration: int) -> None:
    if (
      self._window_episode_chunks is None
      or self._window_failure_chunks is None
    ):
      return

    if self._window_base_iteration is None:
      self._window_base_iteration = int(iteration)
      return

    chunk_size = max(self._window_chunk_size, 1)
    logical_chunk = max(
      (int(iteration) - self._window_base_iteration) // chunk_size,
      0,
    )
    if logical_chunk <= self._window_last_logical_chunk:
      return

    num_chunks = self._window_episode_chunks.shape[0]
    for next_logical_chunk in range(
      self._window_last_logical_chunk + 1,
      logical_chunk + 1,
    ):
      chunk_index = next_logical_chunk % num_chunks
      self.bin_episode_count -= self._window_episode_chunks[chunk_index]
      self.bin_failure_count -= self._window_failure_chunks[chunk_index]
      self._window_episode_chunks[chunk_index].zero_()
      self._window_failure_chunks[chunk_index].zero_()
      self._window_current_chunk = chunk_index

    self.bin_episode_count.clamp_(min=0.0)
    self.bin_failure_count.clamp_(min=0.0)
    self._window_last_logical_chunk = logical_chunk

  def record(
    self,
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
    failure_mask: torch.Tensor | None,
  ) -> None:
    if motion_ids.numel() == 0:
      return

    current_bin_indices = self.compute_motion_bin_indices(time_steps, motion_ids)
    linear_indices = motion_ids * self.bin_count + current_bin_indices
    current_counts = torch.bincount(
      linear_indices,
      minlength=self.num_motions * self.bin_count,
    ).view(self.num_motions, self.bin_count)
    episode_increments = current_counts.float() / torch.clamp(
      self.bin_lengths.float(), min=1.0
    )
    self.bin_episode_count += episode_increments

    failure_increments = torch.zeros_like(self.bin_failure_count)
    if failure_mask is None or not bool(failure_mask.any()):
      self._record_window_increments(episode_increments, failure_increments)
      return

    failed_linear_indices = linear_indices[failure_mask]
    failed_counts = torch.bincount(
      failed_linear_indices,
      minlength=self.num_motions * self.bin_count,
    ).view(self.num_motions, self.bin_count)
    failure_increments = failed_counts.float()
    self.bin_failure_count += failure_increments
    self._record_window_increments(episode_increments, failure_increments)

  def sampling_probabilities(
    self,
    valid_motion_ids: torch.Tensor | None = None,
    valid_bin_ids: torch.Tensor | None = None,
    num_motions: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probabilities for all valid bins or a caller-selected valid subset."""
    if valid_motion_ids is None:
      valid_motion_ids = self.valid_motion_ids
    if valid_bin_ids is None:
      valid_bin_ids = self.valid_bin_ids
    if num_motions is None:
      num_motions = self.num_motions

    failure_rate = self.failure_rate()
    valid_failure_rate = failure_rate[valid_motion_ids, valid_bin_ids]
    failure_rate_mean = valid_failure_rate.mean()
    failure_rate_upper_bound = failure_rate_mean * float(
      self.cfg.failure_rate_max_over_mean
    )
    clipped_failure_rate = torch.clamp(
      valid_failure_rate, 0.0, failure_rate_upper_bound
    )

    clipped_sum = clipped_failure_rate.sum()
    if bool(clipped_sum <= 0.0):
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
    uniform_ratio = float(max(0.0, min(1.0, self.cfg.uniform_ratio)))
    probabilities = (
      1.0 - uniform_ratio
    ) * failure_based_probabilities + uniform_ratio * uniform_probabilities
    probabilities = probabilities * self.bin_weights[valid_motion_ids, valid_bin_ids]
    probabilities = probabilities / torch.clamp(probabilities.sum(), min=1e-12)
    probabilities = self._apply_max_probability_constraints(
      probabilities, valid_motion_ids, num_motions
    )
    return probabilities, valid_failure_rate

  def uniform_baseline_probabilities(
    self, motion_indices: torch.Tensor
  ) -> torch.Tensor:
    return torch.full(
      (len(motion_indices),),
      1.0 / float(self.num_valid_motion_bins),
      dtype=torch.float,
      device=self.device,
    )

  def _init_window(self) -> None:
    window_iterations = self.cfg.failure_rate_window_iterations
    self._window_episode_chunks: torch.Tensor | None = None
    self._window_failure_chunks: torch.Tensor | None = None
    self._window_chunk_size = 0
    self._window_current_chunk = 0
    self._window_base_iteration: int | None = None
    self._window_last_logical_chunk = 0

    if window_iterations is None or int(window_iterations) <= 0:
      return

    window_iterations = max(int(window_iterations), 1)
    num_chunks = max(int(self.cfg.failure_rate_window_chunks), 1)
    num_chunks = min(num_chunks, window_iterations)
    self._window_chunk_size = max(int(math.ceil(window_iterations / num_chunks)), 1)
    chunk_shape = (num_chunks, *self.bin_episode_count.shape)
    self._window_episode_chunks = torch.zeros(
      chunk_shape,
      dtype=self.bin_episode_count.dtype,
      device=self.bin_episode_count.device,
    )
    self._window_failure_chunks = torch.zeros_like(self._window_episode_chunks)
    self._window_episode_chunks[0].copy_(self.bin_episode_count)
    self._window_failure_chunks[0].copy_(self.bin_failure_count)

  def _record_window_increments(
    self, episode_increments: torch.Tensor, failure_increments: torch.Tensor
  ) -> None:
    if (
      self._window_episode_chunks is None
      or self._window_failure_chunks is None
    ):
      return

    self._window_episode_chunks[self._window_current_chunk] += episode_increments
    self._window_failure_chunks[self._window_current_chunk] += failure_increments

  def _resolve_probability_cap(
    self, value: float | Literal["auto"] | None, count: int
  ) -> float | None:
    if value is None:
      return None
    if value == "auto":
      if count <= 0:
        return 1.0
      return float(self.cfg.failure_rate_max_over_mean) / float(count)
    resolved = float(value)
    if resolved <= 0.0:
      return None
    return resolved

  def _apply_max_probability_constraints(
    self,
    probabilities: torch.Tensor,
    valid_motion_ids: torch.Tensor,
    num_motions: int,
  ) -> torch.Tensor:
    constrained = probabilities
    max_prob_per_bin = self._resolve_probability_cap(
      self.cfg.max_prob_per_bin, len(probabilities)
    )
    if max_prob_per_bin is not None and len(probabilities) > 1.0 / max_prob_per_bin:
      constrained = torch.clamp(constrained, max=max_prob_per_bin)
      constrained = constrained / torch.clamp(constrained.sum(), min=1e-12)

    max_prob_per_motion = self._resolve_probability_cap(
      self.cfg.max_prob_per_motion, num_motions
    )
    if max_prob_per_motion is not None and num_motions > 1.0 / max_prob_per_motion:
      motion_probabilities = torch.zeros(
        self.num_motions, dtype=constrained.dtype, device=self.device
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
