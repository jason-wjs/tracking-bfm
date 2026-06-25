"""Environment config for G1 BFM wbteleop tracking."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from tracking_bfm.tasks.tracking import mdp as bfm_mdp
from tracking_bfm.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_bfm_env_cfg,
)
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommandCfg

from . import observations as wbteleop_observations

_LIMB_EE_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
_PELVIS_BODY_NAME = "pelvis"


def _robot_history_length(history_steps: int) -> int:
  history_steps = int(history_steps)
  if history_steps <= 0:
    return 0
  return history_steps + 1


def _history_kwargs(history_steps: int) -> dict[str, int]:
  history_length = _robot_history_length(history_steps)
  if history_length <= 0:
    return {}
  return {"history_length": history_length}


def wbteleop_actor_cfg(
  *,
  history_steps: int,
  future_steps: int,
  enable_corruption: bool,
) -> ObservationGroupCfg:
  robot_history = _history_kwargs(history_steps)
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(
        func=env_mdp.generated_commands,
        params={"command_name": "motion"},
      ),
      "ref_limb_ee_pose_b": ObservationTermCfg(
        func=wbteleop_observations.ref_limb_ee_pose_b,
        params={
          "command_name": "motion",
          "body_names": _LIMB_EE_BODY_NAMES,
          "anchor_body_name": _PELVIS_BODY_NAME,
          "history_steps": history_steps,
          "future_steps": future_steps,
        },
      ),
      "motion_ref_ang_vel": ObservationTermCfg(
        func=wbteleop_observations.motion_ref_ang_vel,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.05, n_max=0.05),
      ),
      "robot_limb_ee_pose_b": ObservationTermCfg(
        func=wbteleop_observations.robot_limb_ee_pose_b,
        params={
          "command_name": "motion",
          "body_names": _LIMB_EE_BODY_NAMES,
          "anchor_body_name": _PELVIS_BODY_NAME,
        },
        **robot_history,
      ),
      "projected_gravity": ObservationTermCfg(
        func=env_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        **robot_history,
      ),
      "base_ang_vel": ObservationTermCfg(
        func=env_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        **robot_history,
      ),
      "joint_pos": ObservationTermCfg(
        func=env_mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        **robot_history,
      ),
      "joint_vel": ObservationTermCfg(
        func=env_mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        **robot_history,
      ),
      "actions": ObservationTermCfg(
        func=env_mdp.last_action,
        **robot_history,
      ),
    },
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def unitree_g1_flat_tracking_bfm_wbteleop_env_cfg(
  *,
  history_steps: int = 0,
  future_steps: int = 1,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 BFM wbteleop tracking environment config."""
  cfg = unitree_g1_flat_tracking_bfm_env_cfg(play=play)

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.history_steps = int(history_steps)
  motion_cmd.future_steps = int(future_steps)

  teacher_actor = deepcopy(cfg.observations["actor"])
  teacher_actor.enable_corruption = False
  cfg.observations["teacher_actor"] = teacher_actor
  cfg.observations["actor"] = wbteleop_actor_cfg(
    history_steps=motion_cmd.history_steps,
    future_steps=motion_cmd.future_steps,
    enable_corruption=not play,
  )
  cfg.rewards["pelvis_limb_ee_pos"] = RewardTermCfg(
    func=bfm_mdp.motion_pelvis_limb_ee_position_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.3,
      "body_names": _LIMB_EE_BODY_NAMES,
      "anchor_body_name": _PELVIS_BODY_NAME,
    },
  )
  cfg.rewards["pelvis_limb_ee_ori"] = RewardTermCfg(
    func=bfm_mdp.motion_pelvis_limb_ee_orientation_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "std": 0.4,
      "body_names": _LIMB_EE_BODY_NAMES,
      "anchor_body_name": _PELVIS_BODY_NAME,
    },
  )
  return cfg
