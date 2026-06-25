"""G1 BFM wbteleop tracking task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import unitree_g1_flat_tracking_bfm_wbteleop_env_cfg
from .rl_cfg import unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg
from .runner import WbTeleopTrackingRunner

register_mjlab_task(
  task_id="Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop",
  env_cfg=unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg(),
  runner_cls=WbTeleopTrackingRunner,
)
