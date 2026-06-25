from __future__ import annotations

import torch

from tracking_bfm.tasks.tracking.mdp.adaptive_sampling import (
  AdaptiveSamplingConfig,
  AdaptiveSamplingState,
)


def test_variable_motion_lengths_create_valid_bins() -> None:
  state = AdaptiveSamplingState(
    torch.tensor([3, 8], dtype=torch.long),
    AdaptiveSamplingConfig(
      bin_width_steps=4,
      init_num_failures=0.0,
    ),
  )

  assert state.bin_width_steps == 4
  assert state.bin_count == 3
  assert state.motion_bin_counts.tolist() == [1, 2]
  assert state.valid_motion_ids.tolist() == [0, 1, 1]
  assert state.valid_bin_ids.tolist() == [0, 0, 1]
  assert state.num_valid_motion_bins == 3
  torch.testing.assert_close(
    state.bin_valid_mask,
    torch.tensor(
      [
        [True, False, False],
        [True, True, False],
      ],
    ),
  )
  torch.testing.assert_close(
    state.bin_lengths,
    torch.tensor(
      [
        [3, 0, 0],
        [4, 4, 0],
      ],
    ),
  )


def test_record_updates_episode_and_failure_counts() -> None:
  state = AdaptiveSamplingState(
    torch.tensor([4, 6], dtype=torch.long),
    AdaptiveSamplingConfig(
      bin_width_steps=3,
      init_num_failures=0.0,
    ),
  )

  state.record(
    motion_ids=torch.tensor([0, 0, 1], dtype=torch.long),
    time_steps=torch.tensor([0, 3, 5], dtype=torch.long),
    failure_mask=torch.tensor([True, False, True]),
  )

  torch.testing.assert_close(
    state.bin_episode_count,
    torch.tensor(
      [
        [1.0 / 3.0, 1.0, 0.0],
        [0.0, 1.0 / 3.0, 0.0],
      ],
    ),
  )
  torch.testing.assert_close(
    state.bin_failure_count,
    torch.tensor(
      [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
      ],
    ),
  )


def test_begin_iteration_advances_sliding_window_and_expires_old_chunks() -> None:
  state = AdaptiveSamplingState(
    torch.tensor([4], dtype=torch.long),
    AdaptiveSamplingConfig(
      bin_width_steps=2,
      init_num_failures=0.0,
      failure_rate_window_iterations=2,
      failure_rate_window_chunks=2,
    ),
  )

  state.begin_iteration(10)
  state.record(
    motion_ids=torch.tensor([0], dtype=torch.long),
    time_steps=torch.tensor([0], dtype=torch.long),
    failure_mask=torch.tensor([True]),
  )
  state.begin_iteration(11)
  state.record(
    motion_ids=torch.tensor([0], dtype=torch.long),
    time_steps=torch.tensor([2], dtype=torch.long),
    failure_mask=torch.tensor([False]),
  )

  state.begin_iteration(12)

  torch.testing.assert_close(
    state.bin_episode_count,
    torch.tensor([[0.0, 0.5, 0.0]]),
  )
  torch.testing.assert_close(
    state.bin_failure_count,
    torch.tensor([[0.0, 0.0, 0.0]]),
  )


def _state_with_failure_rates(
  *,
  max_prob_per_bin: float | None,
  max_prob_per_motion: float | None,
) -> AdaptiveSamplingState:
  state = AdaptiveSamplingState(
    torch.tensor([10, 10], dtype=torch.long),
    AdaptiveSamplingConfig(
      bin_width_steps=5,
      init_num_failures=0.0,
      failure_rate_max_over_mean=10.0,
      sequence_length_agnostic=False,
      uniform_ratio=0.0,
      max_prob_per_bin=max_prob_per_bin,
      max_prob_per_motion=max_prob_per_motion,
    ),
  )
  state.bin_episode_count[state.bin_valid_mask] = 1.0
  state.bin_failure_count[state.bin_valid_mask] = torch.tensor([80.0, 7.0, 3.0, 2.0])
  return state


def test_sampling_probabilities_preserve_one_pass_soft_cap_semantics() -> None:
  uncapped_state = _state_with_failure_rates(
    max_prob_per_bin=None,
    max_prob_per_motion=None,
  )
  capped_state = _state_with_failure_rates(
    max_prob_per_bin=0.45,
    max_prob_per_motion=0.8,
  )

  uncapped_probabilities, _ = uncapped_state.sampling_probabilities()
  probabilities, valid_failure_rate = capped_state.sampling_probabilities()

  torch.testing.assert_close(
    probabilities,
    torch.tensor([0.76574785, 0.12947428, 0.06286673, 0.04191115]),
    rtol=1e-5,
    atol=1e-6,
  )
  torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
  assert probabilities.max().item() > 0.45
  assert probabilities.max().item() < uncapped_probabilities.max().item()
  motion_probabilities = torch.zeros(2, dtype=probabilities.dtype)
  motion_probabilities.scatter_add_(0, capped_state.valid_motion_ids, probabilities)
  uncapped_motion_probabilities = torch.zeros(2, dtype=uncapped_probabilities.dtype)
  uncapped_motion_probabilities.scatter_add_(
    0, uncapped_state.valid_motion_ids, uncapped_probabilities
  )
  assert motion_probabilities.max().item() < uncapped_motion_probabilities.max().item()
  torch.testing.assert_close(valid_failure_rate, torch.tensor([80.0, 7.0, 3.0, 2.0]))
