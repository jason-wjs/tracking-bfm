from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import tracking_bfm  # noqa: E402, F401

PRIMARY_LATENT_IDS = (
  "Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentVelocityBFM-Flat-Unitree-G1",
  "Mjlab-LatentVelocityBFM-Rough-Unitree-G1",
)
REMOVED_LATENT_LEGACY_ALIASES = {
  "Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage",
  "Mjlab-LatentRL-Flat-Unitree-G1",
  "Mjlab-LatentRL-Rough-Unitree-G1",
}


def _reward(weight: float = 0.0, params: dict | None = None):
  return SimpleNamespace(weight=weight, params={} if params is None else dict(params))


def _fake_tracking_cfg(*, play: bool = False):
  del play
  return SimpleNamespace(observations={}, rewards={})


def _fake_velocity_cfg(*, play: bool = False):
  del play
  return SimpleNamespace(
    observations={},
    rewards={
      "pose": _reward(),
      "body_ang_vel": _reward(),
      "track_linear_velocity": _reward(params={"penalize_z_velocity": True}),
      "track_angular_velocity": _reward(params={"penalize_xy_angular_velocity": True}),
      "action_rate_l2": _reward(weight=-0.01),
    },
  )


def test_latent_tracking_env_cfg_adds_sparse_rewards_and_proprio_group(monkeypatch):
  from tracking_bfm.tasks.latent_tracking.config.g1 import env_cfgs

  monkeypatch.setattr(
    env_cfgs,
    "_make_tracking_bfm_1stage_env_cfg",
    lambda *, play=False: deepcopy(_fake_tracking_cfg(play=play)),
  )

  cfg = env_cfgs.unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg()

  assert {
    "sparse_ee_pos",
    "sparse_ee_ori",
    "sparse_root_lin_vel",
    "sparse_root_ang_vel",
    "sparse_root_height",
    "action_rate_l2",
    "waist_action_rate_l2",
    "joint_limit",
    "self_collisions",
  }.issubset(cfg.rewards)
  assert {"projected_gravity", "joint_pos", "joint_vel", "actions"}.issubset(
    cfg.observations["proprio_actor"].terms
  )


def test_latent_velocity_env_cfg_updates_rewards_and_proprio_group(monkeypatch):
  from tracking_bfm.tasks.latent_velocity.config.g1 import env_cfgs

  monkeypatch.setattr(
    env_cfgs,
    "unitree_g1_flat_env_cfg",
    lambda play=False: deepcopy(_fake_velocity_cfg(play=play)),
  )

  cfg = env_cfgs.unitree_g1_flat_latent_rl_env_cfg()

  assert "pose" not in cfg.rewards
  assert "body_ang_vel" not in cfg.rewards
  assert cfg.rewards["track_linear_velocity"].weight == 3.0
  assert cfg.rewards["track_linear_velocity"].params["penalize_z_velocity"] is False
  assert cfg.rewards["track_angular_velocity"].weight == 3.0
  assert (
    cfg.rewards["track_angular_velocity"].params["penalize_xy_angular_velocity"]
    is False
  )
  assert cfg.rewards["action_rate_l2"].weight == -0.2
  assert "waist_joint_vel_l2" in cfg.rewards
  assert {"projected_gravity", "joint_pos", "joint_vel", "actions"}.issubset(
    cfg.observations["proprio_actor"].terms
  )


def test_latent_runner_configs_expose_decoder_checkpoint_fields():
  from tracking_bfm.tasks.latent_tracking.config.g1.rl_cfg import (
    LatentTrackingPpoRunnerCfg,
    unitree_g1_latent_trackingbfm_ppo_runner_cfg,
  )
  from tracking_bfm.tasks.latent_velocity.config.g1.rl_cfg import (
    LatentVelocityPpoRunnerCfg,
    unitree_g1_latent_velocity_ppo_runner_cfg,
  )

  tracking_cfg = unitree_g1_latent_trackingbfm_ppo_runner_cfg()
  assert isinstance(tracking_cfg, LatentTrackingPpoRunnerCfg)
  assert tracking_cfg.latent_decoder_checkpoint_path == ""
  assert tracking_cfg.latent_dim == 64
  assert tracking_cfg.latent_action_clip == 6.0
  assert tracking_cfg.proprio_obs_group == "proprio_actor"
  assert tracking_cfg.clip_actions is None

  velocity_cfg = unitree_g1_latent_velocity_ppo_runner_cfg()
  assert isinstance(velocity_cfg, LatentVelocityPpoRunnerCfg)
  assert velocity_cfg.latent_decoder_checkpoint_path == ""
  assert velocity_cfg.latent_dim == 64
  assert velocity_cfg.latent_action_clip == 6.0
  assert velocity_cfg.proprio_obs_group == "proprio_actor"
  assert velocity_cfg.clip_actions is None


def test_latent_decoder_helpers_validate_checkpoint_shapes_and_cfg(tmp_path):
  from tracking_bfm.tasks.latent_velocity.rl.decoder import (
    infer_hidden_dims,
    infer_mlp_input_dim,
    load_latent_decoder,
  )

  state_dict = {
    "encoder.mlp.0.weight": torch.zeros(8, 10),
    "encoder.mlp.2.weight": torch.zeros(4, 8),
    "decoder.mlp.0.weight": torch.zeros(16, 20),
    "decoder.mlp.2.weight": torch.zeros(12, 16),
    "decoder.mlp.4.weight": torch.zeros(23, 12),
  }
  assert infer_mlp_input_dim(state_dict, "encoder") == 10
  assert infer_hidden_dims(state_dict, "encoder") == (8,)
  assert infer_hidden_dims(state_dict, "decoder") == (16, 12)

  checkpoint_path = tmp_path / "latent_decoder.pt"
  torch.save(
    {
      "model_type": "latent",
      "policy_state_dict": {},
      "latent_cfg": {"latent_dim": 32},
    },
    checkpoint_path,
  )

  with pytest.raises(ValueError, match="latent_dim does not match"):
    load_latent_decoder(
      env=object(),
      train_cfg={
        "latent_decoder_checkpoint_path": str(checkpoint_path),
        "latent_dim": 64,
      },
      device="cpu",
    )


def test_latent_registration_uses_primary_ids(monkeypatch):
  from tracking_bfm.tasks.latent_tracking.config.g1 import (
    register_tasks as register_latent_tracking_tasks,
  )
  from tracking_bfm.tasks.latent_velocity.config.g1 import (
    register_tasks as register_latent_velocity_tasks,
  )

  calls = []
  monkeypatch.setattr(
    "tracking_bfm.tasks.latent_tracking.config.g1.register_mjlab_task",
    lambda **kwargs: calls.append(kwargs),
  )
  monkeypatch.setattr(
    "tracking_bfm.tasks.latent_velocity.config.g1.register_mjlab_task",
    lambda **kwargs: calls.append(kwargs),
  )
  monkeypatch.setattr(
    "tracking_bfm.tasks.latent_tracking.config.g1."
    "unitree_g1_flat_latent_tracking_bfm_1stage_env_cfg",
    lambda play=False: f"latent-tracking-env-play-{play}",
  )
  monkeypatch.setattr(
    "tracking_bfm.tasks.latent_velocity.config.g1.unitree_g1_flat_latent_rl_env_cfg",
    lambda play=False: f"latent-velocity-env-play-{play}",
  )
  monkeypatch.setattr(
    "tracking_bfm.tasks.latent_velocity.config.g1.unitree_g1_rough_latent_rl_env_cfg",
    lambda play=False: f"rough-latent-velocity-env-play-{play}",
  )

  register_latent_tracking_tasks()
  register_latent_velocity_tasks()

  assert [call["task_id"] for call in calls] == list(PRIMARY_LATENT_IDS)
  assert all("aliases" not in call for call in calls)
  assert calls[0]["runner_cls"].__name__ == "LatentTrackingOnPolicyRunner"
  assert calls[1]["runner_cls"].__name__ == "LatentVelocityOnPolicyRunner"
  assert calls[2]["runner_cls"].__name__ == "LatentVelocityOnPolicyRunner"


def test_latent_tasks_are_registered_without_legacy_aliases() -> None:
  task_ids = set(list_tasks())

  assert set(PRIMARY_LATENT_IDS).issubset(task_ids)
  assert REMOVED_LATENT_LEGACY_ALIASES.isdisjoint(task_ids)

  cfg = load_env_cfg("Mjlab-LatentVelocityBFM-Rough-Unitree-G1")

  assert cfg.scene.terrain is not None
  assert "pose" not in cfg.rewards
  assert cfg.rewards["track_linear_velocity"].weight == 3.0
