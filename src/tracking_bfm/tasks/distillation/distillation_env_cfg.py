"""Common environment composition helpers for distillation tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from mjlab.managers.observation_manager import ObservationGroupCfg

from tracking_bfm.tasks.distillation.mdp.commands import StudentSparseCommandVisCfg
from tracking_bfm.tasks.distillation.mdp.observations import (
  build_proprio_actor_terms,
  build_student_actor_terms,
)


@dataclass(frozen=True)
class StudentCommandConfig:
  command_name: str = "motion"
  ee_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
  anchor_body_name: str = "pelvis"
  history_steps: int = 0
  future_steps: int = 1


def _make_tracking_bfm_env_cfg(*, play: bool = False):
  from tracking_bfm.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_bfm_env_cfg,
  )

  return unitree_g1_flat_tracking_bfm_env_cfg(play=play)


def _make_wbteleop_actor_cfg(
  *,
  history_steps: int,
  future_steps: int,
  enable_corruption: bool,
):
  from tracking_bfm.tasks.tracking.wbteleop.env_cfg import wbteleop_actor_cfg

  return wbteleop_actor_cfg(
    history_steps=history_steps,
    future_steps=future_steps,
    enable_corruption=enable_corruption,
  )


def make_distillation_env_cfg(play: bool = False):
  """Build a distillation env cfg by composing the tracking BFM env cfg."""
  cfg = _make_tracking_bfm_env_cfg(play=play)
  student_command_cfg = StudentCommandConfig()

  teacher_actor = deepcopy(cfg.observations["actor"])
  teacher_actor.enable_corruption = False
  student_actor = ObservationGroupCfg(
    terms=build_student_actor_terms(
      command_name=student_command_cfg.command_name,
      ee_body_names=student_command_cfg.ee_body_names,
      anchor_body_name=student_command_cfg.anchor_body_name,
      history_steps=student_command_cfg.history_steps,
      future_steps=student_command_cfg.future_steps,
    ),
    concatenate_terms=True,
    enable_corruption=False,
  )
  proprio_actor = ObservationGroupCfg(
    terms=build_proprio_actor_terms(history_steps=student_command_cfg.history_steps),
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["teacher_actor"] = teacher_actor
  cfg.observations["student_actor"] = student_actor
  cfg.observations["proprio_actor"] = proprio_actor

  if play:
    cfg.commands["motion"].debug_vis = True
    cfg.commands["student_sparse_vis"] = StudentSparseCommandVisCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      command_name=student_command_cfg.command_name,
      ee_body_names=student_command_cfg.ee_body_names,
      anchor_body_name=student_command_cfg.anchor_body_name,
    )
  return cfg


def make_distillation_wbteleop_obs_env_cfg(play: bool = False):
  """Build distillation env cfg with wbteleop actor observations for the student."""
  cfg = make_distillation_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  cfg.observations["student_actor"] = _make_wbteleop_actor_cfg(
    history_steps=motion_cmd.history_steps,
    future_steps=motion_cmd.future_steps,
    enable_corruption=False,
  )
  return cfg
