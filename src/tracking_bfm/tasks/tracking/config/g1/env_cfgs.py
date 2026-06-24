"""Unitree G1 flat tracking environment configurations."""

import math
from copy import deepcopy

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from tracking_bfm.tasks.distillation.mdp import commands as distill_commands
from tracking_bfm.tasks.tracking import mdp
from tracking_bfm.tasks.tracking.mdp import MotionCommandCfg as SingleMotionCommandCfg
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from tracking_bfm.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_STUDENT_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
_STUDENT_ANCHOR_BODY_NAME = "pelvis"
_BASE_INERTIA_ALPHA_RANGE = (0.5 * math.log(0.92), 0.5 * math.log(1.08))
_BODY_INERTIA_ALPHA_RANGE = (0.5 * math.log(0.95), 0.5 * math.log(1.05))
_REGULARIZATION_REWARD_NAMES = (
  "action_rate_l2",
  "waist_action_rate_l2",
  "joint_limit",
  "self_collisions",
)


def _robot_state_history_length(history_steps: int) -> int:
  history_steps = int(history_steps)
  if history_steps <= 0:
    return 0
  return history_steps + 1


def _unitree_g1_sparse_actor_cfg(
  *,
  history_steps: int,
  future_steps: int,
  enable_corruption: bool,
) -> ObservationGroupCfg:
  robot_history_length = _robot_state_history_length(history_steps)
  return ObservationGroupCfg(
    terms={
      "ee_pose": ObservationTermCfg(
        func=distill_commands.student_ee_pose_b,
        params={
          "command_name": "motion",
          "ee_body_names": _STUDENT_EE_BODY_NAMES,
          "anchor_body_name": _STUDENT_ANCHOR_BODY_NAME,
          "history_steps": history_steps,
          "future_steps": future_steps,
        },
      ),
      "base_lin_vel_b": ObservationTermCfg(
        func=distill_commands.student_base_lin_vel_b,
        params={
          "command_name": "motion",
          "anchor_body_name": _STUDENT_ANCHOR_BODY_NAME,
          "history_steps": history_steps,
          "future_steps": future_steps,
        },
      ),
      "base_ang_vel_b": ObservationTermCfg(
        func=distill_commands.student_base_ang_vel_b,
        params={
          "command_name": "motion",
          "anchor_body_name": _STUDENT_ANCHOR_BODY_NAME,
          "history_steps": history_steps,
          "future_steps": future_steps,
        },
      ),
      "anchor_height_w": ObservationTermCfg(
        func=distill_commands.student_anchor_height_w,
        params={
          "command_name": "motion",
          "anchor_body_name": _STUDENT_ANCHOR_BODY_NAME,
          "history_steps": history_steps,
          "future_steps": future_steps,
        },
      ),
      "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=robot_history_length,
      ),
      "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=robot_history_length,
      ),
      "joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=robot_history_length,
      ),
      "joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        history_length=robot_history_length,
      ),
      "actions": ObservationTermCfg(
        func=mdp.last_action,
        history_length=robot_history_length,
      ),
    },
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def add_base_inertia_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add torso mass, inertia, and COM randomization to a G1 tracking config."""
  cfg.events["base_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "alpha_range": _BASE_INERTIA_ALPHA_RANGE,
      "t_range": (-0.02, 0.02),
    },
  )


def add_body_inertia_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add non-torso mass, inertia, and COM randomization to a G1 tracking config."""
  cfg.events["body_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=r"^(?!torso_link$).+"),
      "alpha_range": _BODY_INERTIA_ALPHA_RANGE,
      "t_range": (-0.01, 0.01),
    },
  )


def _use_full_critic_actor_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Give the policy the same uncorrupted observation terms used by the critic."""
  critic_obs = cfg.observations["critic"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=deepcopy(critic_obs.terms),
    concatenate_terms=True,
    enable_corruption=False,
  )


def _use_global_body_pose_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards["motion_body_pos"].func = mdp.motion_global_body_position_error_exp
  cfg.rewards["motion_body_ori"].func = mdp.motion_global_body_orientation_error_exp


def _disable_tracking_regularization_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  for reward_name in _REGULARIZATION_REWARD_NAMES:
    cfg.rewards.pop(reward_name, None)


def _disable_tracking_domain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events.clear()

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, (SingleMotionCommandCfg, MultiMotionCommandCfg))
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)


def _unitree_g1_flat_tracking_env_cfg(
  motion_command_cfg_cls,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg(motion_command_cfg_cls=motion_command_cfg_cls)

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, (SingleMotionCommandCfg, MultiMotionCommandCfg))
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.075, 0.075),
    1: (-0.075, 0.075),
    2: (-0.075, 0.075),
  }
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "add",
      "ranges": (-1.0, 1.0),
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

    ##termination
    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("ee_body_pos", None)
  return cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the single-motion Unitree G1 flat terrain tracking configuration."""
  return _unitree_g1_flat_tracking_env_cfg(
    SingleMotionCommandCfg,
    has_state_estimation=has_state_estimation,
    play=play,
  )


def unitree_g1_flat_tracking_bfm_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the multi-motion Unitree G1 flat terrain tracking configuration."""
  cfg = _unitree_g1_flat_tracking_env_cfg(
    MultiMotionCommandCfg,
    has_state_estimation=has_state_estimation,
    play=play,
  )
  # Keep torso COM randomization from the base G1 config. Mass and inertia
  # randomization are disabled for now.
  # add_body_inertia_randomization(cfg)
  return cfg


def unitree_g1_flat_tracking_bfm_action_trunk_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the multi-motion Unitree G1 tracking task with 4-slice action trunk."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)
  cfg.action_trunk_len = cfg.decimation
  return cfg


def unitree_g1_flat_tracking_bfm_1stage_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the sparse-observation 1-stage Unitree G1 flat tracking configuration."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.history_steps = 0
  motion_cmd.future_steps = 1

  cfg.observations["actor"] = _unitree_g1_sparse_actor_cfg(
    history_steps=motion_cmd.history_steps,
    future_steps=motion_cmd.future_steps,
    enable_corruption=not play,
  )
  return cfg


def unitree_g1_flat_tracking_bfm_test_optimal_env_cfg(
  play: bool = False,
  disable_reg_and_dr: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a full-state, global-body-reward task for optimality probes."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)

  _use_full_critic_actor_observations(cfg)
  _use_global_body_pose_rewards(cfg)

  if disable_reg_and_dr:
    _disable_tracking_regularization_rewards(cfg)
    _disable_tracking_domain_randomization(cfg)

  return cfg
