"""Teacher policy adapters for distillation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDict


class TeacherPolicyAdapter:
  """Thin wrapper around a deterministic teacher policy callable."""

  uses_deterministic_mean_action = True

  def __init__(
    self,
    policy_fn: Callable[[torch.Tensor | TensorDict], torch.Tensor],
    obs_group: str | None = None,
    policy_input_key: str = "actor",
  ):
    self._policy_fn = policy_fn
    self._obs_group = obs_group
    self._policy_input_key = policy_input_key

  def act_mean(self, obs: torch.Tensor | TensorDict) -> torch.Tensor:
    if isinstance(obs, TensorDict) and self._obs_group is not None:
      obs = TensorDict(
        {self._policy_input_key: obs[self._obs_group]},
        batch_size=list(obs.batch_size),
        device=obs.device,
      )
    with torch.no_grad():
      return self._policy_fn(obs)
