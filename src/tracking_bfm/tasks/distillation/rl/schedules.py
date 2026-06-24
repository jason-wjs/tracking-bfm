"""Teacher rollout mixing schedules for distillation."""

from __future__ import annotations


class LinearTeacherMixSchedule:
  """Linearly decay the teacher rollout probability."""

  def __init__(self, beta_start: float, beta_end: float, decay_steps: int):
    if decay_steps <= 0:
      raise ValueError("decay_steps must be positive")
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.decay_steps = decay_steps

  def __call__(self, iteration: int) -> float:
    if iteration <= 0:
      return self.beta_start
    if iteration >= self.decay_steps:
      return self.beta_end
    alpha = iteration / self.decay_steps
    return self.beta_start + alpha * (self.beta_end - self.beta_start)
