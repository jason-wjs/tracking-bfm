"""Runner for latent-action tracking tasks."""

from rsl_rl.env.vec_env import VecEnv

from tracking_bfm.tasks.latent_velocity.rl import LatentDecoderVecEnvWrapper
from tracking_bfm.tasks.latent_velocity.rl.decoder import load_latent_decoder
from tracking_bfm.tasks.tracking.rl import MotionTrackingOnPolicyRunner


class LatentTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """PPO runner for motion tracking policies that output latent actions."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    decoder = load_latent_decoder(env, train_cfg, device)
    latent_env = LatentDecoderVecEnvWrapper(
      env,
      decoder=decoder,
      latent_dim=int(train_cfg["latent_dim"]),
      proprio_obs_group=train_cfg.get("proprio_obs_group", "proprio_actor"),
      latent_action_clip=float(train_cfg["latent_action_clip"]),
    )
    super().__init__(
      latent_env,
      train_cfg,
      log_dir,
      device,
      registry_name=registry_name,
    )
