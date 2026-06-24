"""Unitree G1 latent BFM tracking task registration."""

from tracking_bfm.tasks.latent_tracking.rl import LatentTrackingOnPolicyRunner
from tracking_bfm.tasks.registry import register_task_with_aliases

from .env_cfgs import unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg
from .rl_cfg import unitree_g1_latent_trackingbfm_ppo_runner_cfg


def register_tasks() -> None:
  register_task_with_aliases(
    primary_id="Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
    aliases=("Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",),
    env_cfg=unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg(),
    play_env_cfg=unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg(play=True),
    rl_cfg=unitree_g1_latent_trackingbfm_ppo_runner_cfg(),
    runner_cls=LatentTrackingOnPolicyRunner,
  )


def _auto_register_tasks() -> None:
  try:
    register_tasks()
  except ModuleNotFoundError as exc:
    if (exc.name or "").startswith("tracking_bfm.tasks.tracking"):
      return
    raise


_auto_register_tasks()
