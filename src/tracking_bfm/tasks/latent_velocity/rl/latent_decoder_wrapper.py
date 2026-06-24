"""Latent-action VecEnv wrapper for frozen decoder policies."""

from __future__ import annotations

import torch
from mjlab.utils.spaces import Box, batch_space
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class LatentDecoderVecEnvWrapper(VecEnv):
  """Expose latent actions to RSL-RL and decode them to joint actions."""

  def __init__(
    self,
    env: VecEnv,
    decoder: torch.nn.Module,
    latent_dim: int,
    proprio_obs_group: str,
    latent_action_clip: float = 6.0,
  ) -> None:
    self.env = env
    self.decoder = decoder
    self.latent_dim = int(latent_dim)
    self.proprio_obs_group = proprio_obs_group
    self.latent_action_clip = float(latent_action_clip)
    self.decoder.eval()
    for param in self.decoder.parameters():
      param.requires_grad_(False)

    self.num_envs = env.num_envs
    self.num_actions = self.latent_dim
    self.device = env.device
    self.max_episode_length = env.max_episode_length
    self.single_action_space = Box(
      shape=(self.latent_dim,),
      low=-self.latent_action_clip,
      high=self.latent_action_clip,
    )
    self.action_space = batch_space(self.single_action_space, self.num_envs)
    self._last_obs = env.get_observations()

  @property
  def cfg(self):
    return self.env.cfg

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self._action_space

  @action_space.setter
  def action_space(self, value) -> None:
    self._action_space = value

  def seed(self, seed: int = -1) -> int:
    return self.env.seed(seed)

  def get_observations(self) -> TensorDict:
    self._last_obs = self.env.get_observations()
    return self._last_obs

  def reset(self):
    obs, extras = self.env.reset()
    self._last_obs = obs
    return obs, extras

  @torch.no_grad()
  def step(self, latent_actions: torch.Tensor):
    latent_actions = torch.clamp(
      latent_actions,
      min=-self.latent_action_clip,
      max=self.latent_action_clip,
    )
    decoder_obs = TensorDict(
      {self.proprio_obs_group: self._last_obs[self.proprio_obs_group]},
      batch_size=list(self._last_obs.batch_size),
      device=self._last_obs.device,
    )
    joint_actions = self.decoder.decode(decoder_obs, latent_actions)
    obs, rewards, dones, extras = self.env.step(joint_actions.to(self.env.device))
    self._last_obs = obs

    extras = dict(extras)
    log_extras = dict(extras.get("log", {}))
    log_extras["latent/norm_mean"] = latent_actions.norm(dim=-1).mean()
    log_extras["latent/abs_max"] = latent_actions.abs().max()
    log_extras["latent/decoded_action_norm_mean"] = joint_actions.norm(dim=-1).mean()
    extras["log"] = log_extras
    return obs, rewards, dones, extras

  def close(self) -> None:
    self.env.close()
