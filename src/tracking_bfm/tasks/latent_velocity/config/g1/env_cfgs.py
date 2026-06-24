"""Unitree G1 latent velocity RL environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)

from tracking_bfm.tasks.distillation.mdp.observations import build_proprio_actor_terms
from tracking_bfm.tasks.latent_velocity.mdp import rewards as latent_velocity_rewards

_LATENT_REMOVED_REWARDS = (
  # "upright",
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "air_time",
  "foot_clearance",
  "foot_swing_height",
  "foot_slip",
)


def _make_unitree_g1_latent_rl_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
) -> ManagerBasedRlEnvCfg:
  """Adapt a Unitree G1 velocity config for latent-space RL."""
  for reward_name in _LATENT_REMOVED_REWARDS:
    cfg.rewards.pop(reward_name, None)
  cfg.rewards["track_linear_velocity"].func = (
    latent_velocity_rewards.track_linear_velocity
  )
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_linear_velocity"].params["penalize_z_velocity"] = False
  cfg.rewards["track_angular_velocity"].func = (
    latent_velocity_rewards.track_angular_velocity
  )
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["penalize_xy_angular_velocity"] = False
  cfg.rewards["action_rate_l2"].weight = -0.2
  cfg.rewards["waist_joint_vel_l2"] = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "waist_yaw_joint",
          "waist_roll_joint",
          "waist_pitch_joint",
        ),
      )
    },
  )
  # cfg.rewards["action_l2"].weight = -0.01
  # cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.observations["proprio_actor"] = ObservationGroupCfg(
    terms=build_proprio_actor_terms(history_steps=0),
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def unitree_g1_flat_latent_rl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat velocity config for latent-space RL."""
  return _make_unitree_g1_latent_rl_env_cfg(unitree_g1_flat_env_cfg(play=play))


def unitree_g1_rough_latent_rl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity config for latent-space RL."""
  return _make_unitree_g1_latent_rl_env_cfg(unitree_g1_rough_env_cfg(play=play))
