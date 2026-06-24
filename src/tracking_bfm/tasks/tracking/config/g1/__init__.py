"""Unitree G1 BFM tracking task registrations."""

from tracking_bfm.tasks.registry import register_task_with_aliases
from tracking_bfm.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_tracking_bfm_1stage_env_cfg,
  unitree_g1_flat_tracking_bfm_env_cfg,
)
from .rl_cfg import unitree_g1_trackingbfm_ppo_runner_cfg

register_task_with_aliases(
  primary_id="Mjlab-TrackingBFM-Flat-Unitree-G1",
  aliases=("Mjlab-Trackingbfm-Flat-Unitree-G1",),
  env_cfg=unitree_g1_flat_tracking_bfm_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_task_with_aliases(
  primary_id="Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage",
  aliases=("Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",),
  env_cfg=unitree_g1_flat_tracking_bfm_1stage_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_1stage_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
