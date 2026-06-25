"""Unitree G1 latent velocity task registration."""

from tracking_bfm.tasks.latent_velocity.rl import LatentVelocityOnPolicyRunner
from tracking_bfm.tasks.registry import register_task_with_aliases

from .env_cfgs import (
  unitree_g1_flat_latent_rl_env_cfg,
  unitree_g1_rough_latent_rl_env_cfg,
)
from .rl_cfg import unitree_g1_latent_velocity_ppo_runner_cfg


def register_tasks() -> None:
  register_task_with_aliases(
    primary_id="Mjlab-LatentVelocityBFM-Flat-Unitree-G1",
    aliases=("Mjlab-LatentRL-Flat-Unitree-G1",),
    env_cfg=unitree_g1_flat_latent_rl_env_cfg(),
    play_env_cfg=unitree_g1_flat_latent_rl_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_velocity_ppo_runner_cfg(),
    runner_cls=LatentVelocityOnPolicyRunner,
  )
  register_task_with_aliases(
    primary_id="Mjlab-LatentVelocityBFM-Rough-Unitree-G1",
    aliases=("Mjlab-LatentRL-Rough-Unitree-G1",),
    env_cfg=unitree_g1_rough_latent_rl_env_cfg(),
    play_env_cfg=unitree_g1_rough_latent_rl_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_velocity_ppo_runner_cfg(),
    runner_cls=LatentVelocityOnPolicyRunner,
  )


register_tasks()
