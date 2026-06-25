"""RL configuration for Unitree G1 latent velocity task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class LatentVelocityPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
  latent_decoder_checkpoint_path: str = ""
  latent_dim: int = 64
  latent_action_clip: float = 6.0
  proprio_obs_group: str = "proprio_actor"


def unitree_g1_latent_velocity_ppo_runner_cfg() -> LatentVelocityPpoRunnerCfg:
  """Create PPO runner configuration for Unitree G1 latent velocity RL."""
  return LatentVelocityPpoRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(1024, 1024, 512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 1024, 512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_latent_velocity",
    run_name="latent_rl_flat_g1",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=30_000,
    clip_actions=None,
  )
