from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import tracking_bfm  # noqa: F401
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommandCfg
from tracking_bfm.tasks.tracking.wbteleop.runner import WbTeleopTrackingRunner

PRIMARY_TRACKING_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1"
PRIMARY_1STAGE_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage"
PRIMARY_WBTELEOP_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop"


def _term_names(task_id: str, group: str = "actor") -> set[str]:
  return set(load_env_cfg(task_id).observations[group].terms)


def test_tracking_tasks_register_primary_ids_and_legacy_aliases() -> None:
  task_ids = set(list_tasks())

  assert {
    PRIMARY_TRACKING_ID,
    "Mjlab-Trackingbfm-Flat-Unitree-G1",
    PRIMARY_1STAGE_ID,
    "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",
    PRIMARY_WBTELEOP_ID,
    "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  }.issubset(task_ids)


def test_tracking_bfm_uses_multi_motion_command_and_full_reference_actor() -> None:
  cfg = load_env_cfg(PRIMARY_TRACKING_ID)

  assert isinstance(cfg.commands["motion"], MotionCommandCfg)
  assert {"command", "motion_anchor_pos_b", "motion_anchor_ori_b", "body_pos", "body_ori"}.issubset(
    _term_names(PRIMARY_TRACKING_ID)
  )


def test_tracking_bfm_1stage_uses_sparse_student_actor_terms() -> None:
  cfg = load_env_cfg(PRIMARY_1STAGE_ID)
  motion_cmd = cfg.commands["motion"]

  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.history_steps == 0
  assert motion_cmd.future_steps == 1
  assert {
    "ee_pose",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "anchor_height_w",
    "projected_gravity",
  }.issubset(_term_names(PRIMARY_1STAGE_ID))


def test_tracking_bfm_wbteleop_exposes_teacher_and_limb_reference_student_actor() -> None:
  cfg = load_env_cfg(PRIMARY_WBTELEOP_ID)

  assert "teacher_actor" in cfg.observations
  assert {"body_pos", "body_ori"}.issubset(cfg.observations["teacher_actor"].terms)
  assert {
    "ref_limb_ee_pose_b",
    "motion_ref_ang_vel",
    "robot_limb_ee_pose_b",
  }.issubset(_term_names(PRIMARY_WBTELEOP_ID))
  assert {"body_pos", "body_ori"}.isdisjoint(_term_names(PRIMARY_WBTELEOP_ID))


def test_tracking_bfm_wbteleop_uses_standalone_runner_and_algorithm() -> None:
  rl_cfg = load_rl_cfg(PRIMARY_WBTELEOP_ID)

  assert load_runner_cls(PRIMARY_WBTELEOP_ID) is WbTeleopTrackingRunner
  assert (
    rl_cfg.algorithm.class_name
    == "tracking_bfm.tasks.tracking.wbteleop.algorithm:WbTeleopPPO"
  )
  assert rl_cfg.algorithm.teacher_task_id == PRIMARY_TRACKING_ID
