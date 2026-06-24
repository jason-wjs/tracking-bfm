from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg


def _noop_obs(_env):
  return None


def _fake_tracking_cfg(*, play: bool = False):
  del play
  return SimpleNamespace(
    observations={
      "actor": ObservationGroupCfg(
        terms={"full_reference": ObservationTermCfg(func=_noop_obs)},
        concatenate_terms=True,
        enable_corruption=True,
      )
    },
    commands={"motion": SimpleNamespace(debug_vis=False)},
  )


def test_distillation_env_cfg_adds_teacher_student_and_proprio_groups(monkeypatch):
  from tracking_bfm.tasks.distillation import distillation_env_cfg

  monkeypatch.setattr(
    distillation_env_cfg,
    "_make_tracking_bfm_env_cfg",
    lambda *, play=False: deepcopy(_fake_tracking_cfg(play=play)),
  )

  cfg = distillation_env_cfg.make_distillation_env_cfg()

  assert set(cfg.observations) == {
    "actor",
    "teacher_actor",
    "student_actor",
    "proprio_actor",
  }
  assert cfg.observations["teacher_actor"].terms == {
    "full_reference": cfg.observations["actor"].terms["full_reference"]
  }
  assert cfg.observations["teacher_actor"].enable_corruption is False

  student_terms = cfg.observations["student_actor"].terms
  assert {
    "ee_pose",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "anchor_height_w",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  }.issubset(student_terms)

  proprio_terms = cfg.observations["proprio_actor"].terms
  assert {"projected_gravity", "joint_pos", "joint_vel", "actions"}.issubset(
    proprio_terms
  )
  assert "ee_pose" not in proprio_terms


def test_distillation_play_cfg_adds_student_sparse_visualizer(monkeypatch):
  from tracking_bfm.tasks.distillation import distillation_env_cfg

  monkeypatch.setattr(
    distillation_env_cfg,
    "_make_tracking_bfm_env_cfg",
    lambda *, play=False: deepcopy(_fake_tracking_cfg(play=play)),
  )

  cfg = distillation_env_cfg.make_distillation_env_cfg(play=True)

  assert cfg.commands["motion"].debug_vis is True
  assert cfg.commands["student_sparse_vis"].debug_vis is True
  assert cfg.commands["student_sparse_vis"].command_name == "motion"
  assert cfg.commands["student_sparse_vis"].anchor_body_name == "pelvis"


def test_distillation_runner_configs_use_bfm_primary_teacher_and_latent_fields():
  from tracking_bfm.tasks.distillation.config.g1.rl_cfg import (
    unitree_g1_distillation_runner_cfg,
    unitree_g1_latent_distillation_runner_cfg,
  )

  cfg = unitree_g1_distillation_runner_cfg()
  assert cfg.class_name == "DistillationRunner"
  assert cfg.teacher_task_id == "Mjlab-TrackingBFM-Flat-Unitree-G1"
  assert cfg.teacher_obs_group == "teacher_actor"
  assert cfg.student_obs_group == "student_actor"
  assert cfg.student_model_type == "mlp"

  latent_cfg = unitree_g1_latent_distillation_runner_cfg()
  assert latent_cfg.student_model_type == "latent"
  assert latent_cfg.encoder_obs_group == "teacher_actor"
  assert latent_cfg.decoder_obs_group == "proprio_actor"
  assert latent_cfg.latent_dim == 64
  assert latent_cfg.latent_regularization == "kl"


def test_distillation_registration_uses_primary_ids_and_legacy_aliases(monkeypatch):
  from tracking_bfm.tasks.distillation.config.g1 import register_tasks

  calls = []
  monkeypatch.setattr(
    "tracking_bfm.tasks.distillation.config.g1.register_task_with_aliases",
    lambda **kwargs: calls.append(kwargs),
  )
  monkeypatch.setattr(
    "tracking_bfm.tasks.distillation.config.g1.unitree_g1_flat_distillation_env_cfg",
    lambda play=False: f"dist-env-play-{play}",
  )

  register_tasks()

  assert [call["primary_id"] for call in calls] == [
    "Mjlab-DistillationBFM-Flat-Unitree-G1",
    "Mjlab-LatentDistillationBFM-Flat-Unitree-G1",
  ]
  assert calls[0]["aliases"] == ("Mjlab-Distillation-Flat-Unitree-G1",)
  assert calls[1]["aliases"] == ("Mjlab-LatentDistillation-Flat-Unitree-G1",)
  assert calls[0]["runner_cls"].__name__ == "DistillationRunner"
  assert calls[1]["runner_cls"].__name__ == "DistillationRunner"
