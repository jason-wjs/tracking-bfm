"""Unitree G1 latent tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp as upstream_tracking_mdp

from tracking_bfm.tasks.distillation.mdp.observations import build_proprio_actor_terms
from tracking_bfm.tasks.tracking import mdp as bfm_mdp

_SPARSE_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_SPARSE_ROOT_BODY_NAME = "pelvis"


def _make_sparse_tracking_rewards() -> dict[str, RewardTermCfg]:
  return {
    "sparse_ee_pos": RewardTermCfg(
      func=upstream_tracking_mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.3,
        "body_names": _SPARSE_EE_BODY_NAMES,
      },
    ),
    "sparse_ee_ori": RewardTermCfg(
      func=upstream_tracking_mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.4,
        "body_names": _SPARSE_EE_BODY_NAMES,
      },
    ),
    "sparse_root_lin_vel": RewardTermCfg(
      func=upstream_tracking_mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 1.0,
        "body_names": (_SPARSE_ROOT_BODY_NAME,),
      },
    ),
    "sparse_root_ang_vel": RewardTermCfg(
      func=upstream_tracking_mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 3.14,
        "body_names": (_SPARSE_ROOT_BODY_NAME,),
      },
    ),
    "sparse_root_height": RewardTermCfg(
      func=bfm_mdp.motion_global_body_height_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.3,
        "body_name": _SPARSE_ROOT_BODY_NAME,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=env_mdp.action_rate_l2, weight=-1e-1),
    "waist_action_rate_l2": RewardTermCfg(
      func=bfm_mdp.joint_action_rate_l2,
      weight=-5e-2,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
          ),
        ),
        "action_name": "joint_pos",
      },
    ),
    "joint_limit": RewardTermCfg(
      func=env_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=upstream_tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def _make_tracking_bfm_1stage_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  from tracking_bfm.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_bfm_1stage_env_cfg,
  )

  return unitree_g1_flat_tracking_bfm_1stage_env_cfg(play=play)


def unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create latent-action sparse-observation Unitree G1 tracking config."""
  cfg = _make_tracking_bfm_1stage_env_cfg(play=play)
  cfg.rewards = _make_sparse_tracking_rewards()
  cfg.observations["proprio_actor"] = ObservationGroupCfg(
    terms=build_proprio_actor_terms(history_steps=0),
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg
