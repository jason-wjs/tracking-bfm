"""RL configuration for the Unitree G1 distillation task."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  teacher_task_id: str = "Mjlab-TrackingBFM-Flat-Unitree-G1"
  teacher_checkpoint_path: str = ""
  teacher_obs_group: str = "teacher_actor"
  student_obs_group: str = "student_actor"
  student_model_type: Literal["mlp", "latent"] = "mlp"
  encoder_obs_group: str = "teacher_actor"
  decoder_obs_group: str = "proprio_actor"
  beta_schedule: Literal["linear"] = "linear"
  beta_start: float = 1.0
  beta_end: float = 0.0
  beta_decay_steps: int = 2_00
  student_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (4096,2048,2048,1024, 1024, 1024, 512, 256, 128))
  student_activation: str = "elu"
  latent_dim: int = 64
  encoder_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (2048,2048,1024, 512, 256))
  decoder_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (2048,2048,1024, 1024, 512, 256, 128))
  latent_activation: str = "elu"
  latent_regularization: Literal["kl", "wae_mmd", "bfmzero_sphere"] = "kl"
  kl_weight: float = 1.0e-4
  kl_warmup_iterations: int = 2_000
  free_nats_per_dim: float = 0.02
  mmd_weight: float = 0.0
  mmd_kernel_scales: tuple[float, ...] = field(default_factory=lambda: (0.5, 1.0, 2.0, 4.0))
  mmd_max_samples: int = 1024
  latent_smooth_weight: float = 1.0e-3
  latent_smooth_max_pairs: int = 2048
  sphere_radius: float = -1.0
  sphere_orthonormal_weight: float = 0.0
  sphere_knn_smooth_weight: float = 0.0
  sphere_knn_k: int = 4
  sphere_knn_max_samples: int = 2048
  sphere_eps: float = 1.0e-6
  latent_log_std_min: float = -5.0
  latent_log_std_max: float = 2.0
  learning_rate: float = 1.0e-3
  num_learning_epochs: int = 5
  num_mini_batches: int = 4


def unitree_g1_distillation_runner_cfg() -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    experiment_name="g1_distillation",
    save_interval=5000,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_latent_distillation_runner_cfg() -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    experiment_name="g1_latent_distillation",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
    student_model_type="latent",
    encoder_obs_group="teacher_actor",
    decoder_obs_group="proprio_actor",
  )
