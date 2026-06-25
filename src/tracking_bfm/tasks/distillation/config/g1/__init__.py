"""Unitree G1 BFM distillation task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from tracking_bfm.tasks.distillation.rl import DistillationRunner

from .env_cfgs import (
  unitree_g1_flat_distillation_env_cfg,
  unitree_g1_flat_distillation_wbteleop_obs_env_cfg,
)
from .rl_cfg import (
  unitree_g1_distillation_runner_cfg,
  unitree_g1_latent_distillation_runner_cfg,
)


def register_tasks() -> None:
  register_mjlab_task(
    task_id="Mjlab-DistillationBFM-Flat-Unitree-G1",
    env_cfg=unitree_g1_flat_distillation_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
    rl_cfg=unitree_g1_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )
  register_mjlab_task(
    task_id="Mjlab-LatentDistillationBFM-Flat-Unitree-G1",
    env_cfg=unitree_g1_flat_distillation_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )
  register_mjlab_task(
    task_id="Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs",
    env_cfg=unitree_g1_flat_distillation_wbteleop_obs_env_cfg(),
    play_env_cfg=unitree_g1_flat_distillation_wbteleop_obs_env_cfg(play=True),
    rl_cfg=unitree_g1_distillation_runner_cfg(),
    runner_cls=DistillationRunner,
  )


register_tasks()
