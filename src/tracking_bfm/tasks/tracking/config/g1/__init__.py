"""Unitree G1 BFM tracking task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from tracking_bfm.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_tracking_bfm_1stage_env_cfg,
  unitree_g1_flat_tracking_bfm_env_cfg,
  unitree_g1_flat_tracking_bfm_test_optimal_env_cfg,
)
from .rl_cfg import unitree_g1_trackingbfm_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-TrackingBFM-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_bfm_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage",
  env_cfg=unitree_g1_flat_tracking_bfm_1stage_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_1stage_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal",
  env_cfg=unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(play=True),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
  env_cfg=unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(disable_reg_and_dr=True),
  play_env_cfg=unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(
    play=True,
    disable_reg_and_dr=True,
  ),
  rl_cfg=unitree_g1_trackingbfm_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
