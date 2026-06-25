from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import tracking_bfm  # noqa: F401
from tracking_bfm.tasks.tracking.mdp import multi_commands as legacy_multi_commands
from tracking_bfm.tasks.tracking.mdp import multi_motion_command
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommandCfg
from tracking_bfm.tasks.tracking.wbteleop.runner import WbTeleopTrackingRunner

PRIMARY_TRACKING_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1"
PRIMARY_1STAGE_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage"
PRIMARY_WBTELEOP_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop"
PRIMARY_TEST_OPTIMAL_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal"
PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID = (
  "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR"
)
REMOVED_TRACKING_LEGACY_IDS = {
  "Mjlab-TrackingBFM-Flat-Unitree-G1-ActionTrunk",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk",
}


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
    PRIMARY_TEST_OPTIMAL_ID,
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal",
    PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID,
    "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
  }.issubset(task_ids)


def test_legacy_multi_commands_import_preserves_motion_command_identity() -> None:
  assert legacy_multi_commands.MotionCommandCfg is multi_motion_command.MotionCommandCfg
  assert (
    legacy_multi_commands.MultiMotionCommandCfg
    is multi_motion_command.MultiMotionCommandCfg
  )
  assert (
    legacy_multi_commands.MultiMotionCommand is multi_motion_command.MultiMotionCommand
  )


@pytest.mark.parametrize(
  ("primary_id", "legacy_alias"),
  [
    (PRIMARY_TRACKING_ID, "Mjlab-Trackingbfm-Flat-Unitree-G1"),
    (PRIMARY_1STAGE_ID, "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage"),
    (PRIMARY_WBTELEOP_ID, "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop"),
    (PRIMARY_TEST_OPTIMAL_ID, "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal"),
    (
      PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID,
      "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
    ),
  ],
)
def test_tracking_legacy_aliases_resolve_to_matching_task_surface(
  primary_id: str,
  legacy_alias: str,
) -> None:
  primary_env = load_env_cfg(primary_id)
  alias_env = load_env_cfg(legacy_alias)
  primary_play_env = load_env_cfg(primary_id, play=True)
  alias_play_env = load_env_cfg(legacy_alias, play=True)
  primary_rl = load_rl_cfg(primary_id)
  alias_rl = load_rl_cfg(legacy_alias)

  assert type(alias_env.commands["motion"]) is type(primary_env.commands["motion"])
  assert type(alias_play_env.commands["motion"]) is type(
    primary_play_env.commands["motion"]
  )
  assert load_runner_cls(legacy_alias) is load_runner_cls(primary_id)
  assert alias_rl.algorithm.class_name == primary_rl.algorithm.class_name
  assert alias_rl.experiment_name == primary_rl.experiment_name


def test_removed_action_trunk_is_not_registered_or_exposed() -> None:
  from tracking_bfm.tasks.tracking.config.g1 import env_cfgs, rl_cfg

  task_ids = set(list_tasks())

  assert REMOVED_TRACKING_LEGACY_IDS.isdisjoint(task_ids)
  assert not hasattr(env_cfgs, "unitree_g1_flat_tracking_bfm_action_trunk_env_cfg")
  assert not hasattr(rl_cfg, "unitree_g1_trackingbfm_action_trunk_ppo_runner_cfg")


def test_tracking_bfm_test_optimal_uses_full_critic_actor_observations() -> None:
  cfg = load_env_cfg(PRIMARY_TEST_OPTIMAL_ID)

  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  assert set(actor.terms) == set(critic.terms)
  assert actor.enable_corruption is False
  assert critic.enable_corruption is False
  assert all(term.noise is None for term in actor.terms.values())


def test_tracking_bfm_test_optimal_uses_global_body_pose_rewards() -> None:
  from tracking_bfm.tasks.tracking import mdp

  cfg = load_env_cfg(PRIMARY_TEST_OPTIMAL_ID)

  assert cfg.rewards["motion_body_pos"].func is (
    mdp.motion_global_body_position_error_exp
  )
  assert cfg.rewards["motion_body_ori"].func is (
    mdp.motion_global_body_orientation_error_exp
  )


def test_tracking_bfm_test_optimal_no_reg_no_dr_removes_interference() -> None:
  cfg = load_env_cfg(PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID)

  assert cfg.events == {}
  assert "action_rate_l2" not in cfg.rewards
  assert "waist_action_rate_l2" not in cfg.rewards
  assert "joint_limit" not in cfg.rewards
  assert "self_collisions" not in cfg.rewards

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.joint_position_range == (0.0, 0.0)


def test_tracking_reuses_upstream_observation_and_termination_terms() -> None:
  from mjlab.tasks.tracking.mdp import observations as upstream_observations
  from mjlab.tasks.tracking.mdp import terminations as upstream_terminations

  from tracking_bfm.tasks.tracking.mdp import observations, terminations

  assert observations.__all__ == [
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "robot_body_ori_b",
    "robot_body_pos_b",
  ]
  assert terminations.__all__ == [
    "bad_anchor_ori",
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
  ]

  for name in (
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "robot_body_pos_b",
    "robot_body_ori_b",
  ):
    assert getattr(observations, name) is getattr(upstream_observations, name)

  for name in (
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_anchor_ori",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
  ):
    assert getattr(terminations, name) is getattr(upstream_terminations, name)


def test_tracking_mdp_namespace_keeps_local_motion_command_exports() -> None:
  from tracking_bfm.tasks.tracking import mdp
  from tracking_bfm.tasks.tracking.mdp.commands import MotionCommand
  from tracking_bfm.tasks.tracking.mdp.commands import MotionCommandCfg as SingleCfg
  from tracking_bfm.tasks.tracking.mdp.multi_motion_command import (
    MotionCommandCfg as MultiCfg,
  )

  assert mdp.MotionCommand is MotionCommand
  assert mdp.MotionCommandCfg is SingleCfg
  assert mdp.MotionCommandCfg is not MultiCfg


def test_tracking_bfm_uses_multi_motion_command_and_full_reference_actor() -> None:
  cfg = load_env_cfg(PRIMARY_TRACKING_ID)

  assert isinstance(cfg.commands["motion"], MotionCommandCfg)
  assert {
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
  }.issubset(_term_names(PRIMARY_TRACKING_ID))


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


def test_tracking_bfm_wbteleop_exposes_teacher_and_limb_reference_student_actor() -> (
  None
):
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
