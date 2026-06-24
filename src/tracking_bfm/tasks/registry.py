"""Small helpers around the mjlab task registry."""

from __future__ import annotations

from collections.abc import Iterable

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.tasks.registry import register_mjlab_task


def register_task_with_aliases(
  *,
  primary_id: str,
  aliases: Iterable[str] = (),
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlBaseRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register a primary task ID plus legacy aliases using one config set."""
  register_mjlab_task(
    task_id=primary_id,
    env_cfg=env_cfg,
    play_env_cfg=play_env_cfg,
    rl_cfg=rl_cfg,
    runner_cls=runner_cls,
  )
  for alias in aliases:
    register_mjlab_task(
      task_id=alias,
      env_cfg=env_cfg,
      play_env_cfg=play_env_cfg,
      rl_cfg=rl_cfg,
      runner_cls=runner_cls,
    )
