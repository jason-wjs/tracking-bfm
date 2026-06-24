"""Unitree G1 distillation environment configurations."""

from tracking_bfm.tasks.distillation.distillation_env_cfg import (
  make_distillation_env_cfg,
  make_distillation_wbteleop_obs_env_cfg,
)


def unitree_g1_flat_distillation_env_cfg(play: bool = False):
  return make_distillation_env_cfg(play=play)


def unitree_g1_flat_distillation_wbteleop_obs_env_cfg(play: bool = False):
  return make_distillation_wbteleop_obs_env_cfg(play=play)
