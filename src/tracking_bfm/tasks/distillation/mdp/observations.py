from __future__ import annotations

from mjlab.envs import mdp as env_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from . import commands as distill_commands


def _robot_state_history_length(history_steps: int) -> int:
  history_steps = int(history_steps)
  if history_steps <= 0:
    return 0
  return history_steps + 1


def build_student_actor_terms(
  *,
  command_name: str,
  ee_body_names: tuple[str, str],
  anchor_body_name: str,
  history_steps: int = 0,
  future_steps: int = 1,
) -> dict[str, ObservationTermCfg]:
  robot_history_length = _robot_state_history_length(history_steps)
  return {
    "ee_pose": ObservationTermCfg(
      func=distill_commands.student_ee_pose_b,
      params={
        "command_name": command_name,
        "ee_body_names": ee_body_names,
        "anchor_body_name": anchor_body_name,
        "history_steps": history_steps,
        "future_steps": future_steps,
      },
    ),
    # "base_lin_vel_w": ObservationTermCfg(
    #   func=distill_commands.student_base_lin_vel_w,
    #   params={
    #     "command_name": command_name,
    #     "anchor_body_name": anchor_body_name,
    #     "history_steps": history_steps,
    #     "future_steps": future_steps,
    #   },
    # ),
    "base_lin_vel_b": ObservationTermCfg(
      func=distill_commands.student_base_lin_vel_b,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
        "history_steps": history_steps,
        "future_steps": future_steps,
      },
    ),
    # "base_ang_vel_w": ObservationTermCfg(
    #   func=distill_commands.student_base_ang_vel_w,
    #   params={
    #     "command_name": command_name,
    #     "anchor_body_name": anchor_body_name,
    #     "history_steps": history_steps,
    #     "future_steps": future_steps,
    #   },
    # ),
    "base_ang_vel_b": ObservationTermCfg(
      func=distill_commands.student_base_ang_vel_b,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
        "history_steps": history_steps,
        "future_steps": future_steps,
      },
    ),
    "anchor_height_w": ObservationTermCfg(
      func=distill_commands.student_anchor_height_w,
      params={
        "command_name": command_name,
        "anchor_body_name": anchor_body_name,
        "history_steps": history_steps,
        "future_steps": future_steps,
      },
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=robot_history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=robot_history_length,
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=robot_history_length,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=robot_history_length,
    ),
    "actions": ObservationTermCfg(
      func=env_mdp.last_action,
      history_length=robot_history_length,
    ),
  }


def build_proprio_actor_terms(history_steps: int = 0) -> dict[str, ObservationTermCfg]:
  """Build decoder-only robot proprioception terms."""
  robot_history_length = _robot_state_history_length(history_steps)
  return {
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=robot_history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=robot_history_length,
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=robot_history_length,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=robot_history_length,
    ),
    "actions": ObservationTermCfg(
      func=env_mdp.last_action,
      history_length=robot_history_length,
    ),
  }
