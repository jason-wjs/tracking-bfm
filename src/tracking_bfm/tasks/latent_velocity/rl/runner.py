"""Runner for latent velocity RL tasks."""

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from tracking_bfm.tasks.latent_velocity.rl.decoder import load_latent_decoder
from tracking_bfm.tasks.latent_velocity.rl.latent_decoder_wrapper import (
  LatentDecoderVecEnvWrapper,
)


class LatentVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """PPO runner for direct latent velocity RL."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    decoder = load_latent_decoder(env, train_cfg, device)
    latent_env = LatentDecoderVecEnvWrapper(
      env,
      decoder=decoder,
      latent_dim=int(train_cfg["latent_dim"]),
      proprio_obs_group=train_cfg.get("proprio_obs_group", "proprio_actor"),
      latent_action_clip=float(train_cfg["latent_action_clip"]),
    )
    super().__init__(latent_env, train_cfg, log_dir, device)

  @staticmethod
  def _load_decoder(env, train_cfg: dict, device: str) -> torch.nn.Module:
    return load_latent_decoder(env, train_cfg, device)
