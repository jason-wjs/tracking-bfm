"""Unitree G1 latent BFM tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from tracking_bfm.tasks.latent_tracking.rl import LatentTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg
from .rl_cfg import unitree_g1_latent_trackingbfm_ppo_runner_cfg


def register_tasks() -> None:
  register_mjlab_task(
    task_id="Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
    env_cfg=unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg(),
    play_env_cfg=unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_trackingbfm_ppo_runner_cfg(),
    runner_cls=LatentTrackingOnPolicyRunner,
  )


register_tasks()
