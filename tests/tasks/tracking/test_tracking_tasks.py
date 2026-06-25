from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from importlib.util import find_spec

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import tracking_bfm  # noqa: F401
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommandCfg
from tracking_bfm.tasks.tracking.wbteleop.runner import WbTeleopTrackingRunner

PRIMARY_TRACKING_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1"
PRIMARY_1STAGE_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage"
PRIMARY_WBTELEOP_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop"
PRIMARY_TEST_OPTIMAL_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal"
PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID = (
  "Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR"
)
PRIMARY_TRACKING_IDS = {
  PRIMARY_TRACKING_ID,
  PRIMARY_1STAGE_ID,
  PRIMARY_WBTELEOP_ID,
  PRIMARY_TEST_OPTIMAL_ID,
  PRIMARY_TEST_OPTIMAL_NO_REG_NO_DR_ID,
}
REMOVED_TRACKING_LEGACY_ALIASES = {
  "Mjlab-Trackingbfm-Flat-Unitree-G1",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR",
}
REMOVED_TRACKING_LEGACY_IDS = {
  "Mjlab-TrackingBFM-Flat-Unitree-G1-ActionTrunk",
  "Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk",
}


def _term_names(task_id: str, group: str = "actor") -> set[str]:
  return set(load_env_cfg(task_id).observations[group].terms)


def test_tracking_tasks_register_primary_ids_without_legacy_aliases() -> None:
  task_ids = set(list_tasks())

  assert PRIMARY_TRACKING_IDS.issubset(task_ids)
  assert REMOVED_TRACKING_LEGACY_ALIASES.isdisjoint(task_ids)


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


def test_historical_tracking_mdp_shims_are_removed() -> None:
  for module_name in (
    "tracking_bfm.tasks.tracking.mdp.multi_commands",
    "tracking_bfm.tasks.tracking.mdp.observations",
    "tracking_bfm.tasks.tracking.mdp.terminations",
    "tracking_bfm.tasks.tracking.mdp.metrics",
  ):
    assert find_spec(module_name) is None


def test_tracking_cfg_uses_upstream_terms_and_local_bfm_rewards() -> None:
  from mjlab.envs import mdp as env_mdp
  from mjlab.tasks.tracking import mdp as upstream_tracking_mdp

  from tracking_bfm.tasks.tracking import mdp as bfm_mdp

  cfg = load_env_cfg(PRIMARY_TRACKING_ID)

  assert cfg.observations["actor"].terms["command"].func is env_mdp.generated_commands
  assert (
    cfg.observations["actor"].terms["motion_anchor_pos_b"].func
    is upstream_tracking_mdp.motion_anchor_pos_b
  )
  assert (
    cfg.observations["actor"].terms["body_pos"].func
    is upstream_tracking_mdp.robot_body_pos_b
  )
  assert (
    cfg.rewards["motion_global_root_pos"].func
    is upstream_tracking_mdp.motion_global_anchor_position_error_exp
  )
  assert (
    cfg.rewards["motion_body_pos"].func
    is upstream_tracking_mdp.motion_relative_body_position_error_exp
  )
  assert cfg.rewards["action_rate_l2"].func is env_mdp.action_rate_l2
  assert cfg.rewards["waist_action_rate_l2"].func is bfm_mdp.joint_action_rate_l2
  assert (
    cfg.terminations["anchor_pos"].func
    is upstream_tracking_mdp.bad_anchor_pos_z_only
  )


def test_tracking_env_cfg_composes_upstream_base_cfg(monkeypatch) -> None:
  from mjlab.tasks.tracking import tracking_env_cfg as upstream_tracking_env_cfg

  from tracking_bfm.tasks.tracking import tracking_env_cfg

  calls = []
  real_make_tracking_env_cfg = upstream_tracking_env_cfg.make_tracking_env_cfg

  def fake_make_tracking_env_cfg():
    calls.append("called")
    return real_make_tracking_env_cfg()

  monkeypatch.setattr(
    upstream_tracking_env_cfg,
    "make_tracking_env_cfg",
    fake_make_tracking_env_cfg,
  )

  cfg = tracking_env_cfg.make_tracking_env_cfg()

  assert calls == ["called"]
  assert {"body_pos", "body_ori"}.issubset(cfg.observations["actor"].terms)
  assert "biased" not in cfg.observations["actor"].terms["joint_pos"].params
  assert cfg.rewards["motion_global_root_pos"].weight == 1.0
  assert "waist_action_rate_l2" in cfg.rewards
  assert cfg.events["foot_friction"].params["ranges"] == (0.3, 2.0)
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.5


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


def test_wbteleop_reference_observations_use_public_reference_gather() -> None:
  from types import SimpleNamespace

  import torch

  from tracking_bfm.tasks.tracking.wbteleop import observations

  class FakeCommand:
    cfg = SimpleNamespace(
      body_names=(
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      )
    )
    time_steps = torch.tensor([5], dtype=torch.long)
    motion_idx = torch.tensor([0], dtype=torch.long)
    _env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))

    def gather_reference_field(self, field_name, motion_ids, time_steps):
      assert field_name in {"body_pos_w", "body_quat_w"}
      if field_name == "body_pos_w":
        return torch.zeros(1, 3, 5, 3)
      quat = torch.zeros(1, 3, 5, 4)
      quat[..., 0] = 1.0
      return quat

  env = SimpleNamespace(
    num_envs=1,
    command_manager=SimpleNamespace(get_term=lambda _: FakeCommand()),
  )

  obs = observations.ref_limb_ee_pose_b(
    env,
    "motion",
    history_steps=1,
    future_steps=2,
  )

  assert obs.shape == (1, 108)
