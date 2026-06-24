"""RL config for G1 BFM wbteleop tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class WbTeleopPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "tracking_bfm.tasks.tracking.wbteleop.algorithm:WbTeleopPPO"
  teacher_task_id: str = "Mjlab-TrackingBFM-Flat-Unitree-G1"
  teacher_checkpoint_path: str = ""
  teacher_obs_group: str = "teacher_actor"
  bc_weight_start: float = 0.5
  bc_weight_end: float = 0.1
  bc_decay_steps: int = 10_000
  pure_bc_enabled: bool = False
  pure_bc_weight: float = 1.0
  pure_bc_rollout: Literal["student", "teacher"] = "student"
  bc_actor_checkpoint_path: str = ""
  init_actor_std_from_teacher: bool = False
  init_critic_from_teacher: bool = True
  strict_init: bool = True


def unitree_g1_trackingbfm_wbteleop_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the runner config for the G1 BFM wbteleop task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(2048, 2048, 1024, 1024, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=WbTeleopPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="g1_tracking_wbteleop",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=300_000,
  )
