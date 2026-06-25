"""Unitree G1 BFM distillation task registrations."""

from tracking_bfm.tasks.distillation.rl import DistillationRunner
from tracking_bfm.tasks.registry import register_task_with_aliases

from .env_cfgs import (
  unitree_g1_flat_distillation_env_cfg,
  unitree_g1_flat_distillation_wbteleop_obs_env_cfg,
)
from .rl_cfg import (
  unitree_g1_distillation_runner_cfg,
  unitree_g1_latent_distillation_runner_cfg,
)


def register_tasks() -> None:
  register_task_with_aliases(
    primary_id="Mjlab-DistillationBFM-Flat-Unitree-G1",
    aliases=("Mjlab-Distillation-Flat-Unitree-G1",),
    env_cfg=unitree_g1_flat_distillation_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
    rl_cfg=unitree_g1_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )
  register_task_with_aliases(
    primary_id="Mjlab-LatentDistillationBFM-Flat-Unitree-G1",
    aliases=("Mjlab-LatentDistillation-Flat-Unitree-G1",),
    env_cfg=unitree_g1_flat_distillation_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )
  register_task_with_aliases(
    primary_id="Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs",
    aliases=("Mjlab-DistillationWbteleopObs-Flat-Unitree-G1",),
    env_cfg=unitree_g1_flat_distillation_wbteleop_obs_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_wbteleop_obs_env_cfg(play=True),
    rl_cfg=unitree_g1_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )


register_tasks()
