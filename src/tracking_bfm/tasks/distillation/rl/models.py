"""Student model builders for distillation."""

from __future__ import annotations

import math

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class LatentDistillationModel(torch.nn.Module):
  """Encoder/decoder latent student for first-stage action distillation."""

  def __init__(
    self,
    obs: TensorDict,
    encoder_obs_group: str,
    decoder_obs_group: str,
    action_dim: int,
    latent_dim: int,
    encoder_hidden_dims: tuple[int, ...],
    decoder_hidden_dims: tuple[int, ...],
    activation: str,
    obs_normalization: bool = True,
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
    latent_mode: str = "gaussian",
    sphere_radius: float = -1.0,
    sphere_eps: float = 1.0e-6,
  ):
    super().__init__()
    self.encoder_obs_group = encoder_obs_group
    self.decoder_obs_group = decoder_obs_group
    self.latent_dim = int(latent_dim)
    self.action_dim = int(action_dim)
    self.log_std_min = float(log_std_min)
    self.log_std_max = float(log_std_max)
    self.latent_mode = latent_mode
    self.sphere_radius = float(sphere_radius)
    self.sphere_eps = float(sphere_eps)
    self.decoder_input_group = "_latent_decoder_input"

    self.encoder = MLPModel(
      obs=obs,
      obs_groups={"encoder": [encoder_obs_group]},
      obs_set="encoder",
      output_dim=2 * self.latent_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
    )
    decoder_obs_dim = obs[decoder_obs_group].shape[-1]
    decoder_template = TensorDict(
      {
        self.decoder_input_group: torch.zeros(
          *obs.batch_size,
          decoder_obs_dim + self.latent_dim,
          dtype=obs[decoder_obs_group].dtype,
          device=obs[decoder_obs_group].device,
        )
      },
      batch_size=list(obs.batch_size),
      device=obs.device,
    )
    self.decoder = MLPModel(
      obs=decoder_template,
      obs_groups={"decoder": [self.decoder_input_group]},
      obs_set="decoder",
      output_dim=action_dim,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
    )

  def encode(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_out = self.encoder(obs)
    mu, log_std = torch.chunk(encoder_out, chunks=2, dim=-1)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
    return mu, log_std

  @staticmethod
  def sample(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(log_std)

  @staticmethod
  def spherical_project(
    z: torch.Tensor,
    radius: float = -1.0,
    eps: float = 1.0e-6,
  ) -> torch.Tensor:
    radius_value = math.sqrt(z.shape[-1]) if radius < 0.0 else float(radius)
    norm = z.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(z)
    fallback[..., 0] = 1.0
    unit = torch.where(norm > float(eps), z / norm.clamp_min(float(eps)), fallback)
    return unit * radius_value

  @staticmethod
  def slerp(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: float | torch.Tensor,
    radius: float = -1.0,
    eps: float = 1.0e-6,
  ) -> torch.Tensor:
    radius_value = math.sqrt(z0.shape[-1]) if radius < 0.0 else float(radius)
    u0 = LatentDistillationModel.spherical_project(z0, radius=1.0, eps=eps)
    u1 = LatentDistillationModel.spherical_project(z1, radius=1.0, eps=eps)
    dot = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    t_tensor = torch.as_tensor(t, dtype=z0.dtype, device=z0.device)
    while t_tensor.ndim < z0.ndim:
      t_tensor = t_tensor.unsqueeze(-1)
    lerp = (1.0 - t_tensor) * u0 + t_tensor * u1
    slerp = (
      torch.sin((1.0 - t_tensor) * omega) / sin_omega * u0
      + torch.sin(t_tensor * omega) / sin_omega * u1
    )
    near_collinear = sin_omega.abs() < eps
    interpolated = torch.where(near_collinear, lerp, slerp)
    return LatentDistillationModel.spherical_project(
      interpolated, radius=radius_value, eps=eps
    )

  def decode(self, obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
    decoder_obs = self._decoder_obs(obs, z)
    return self.decoder(decoder_obs)

  def forward(
    self,
    obs: TensorDict,
    deterministic: bool = False,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mu, log_std = self.encode(obs)
    z_raw = mu if deterministic else self.sample(mu, log_std)
    z = self._decode_latent(z_raw)
    actions = self.decode(obs, z)
    latent = {"mu": mu, "log_std": log_std, "z": z, "z_raw": z_raw}
    if self.latent_mode == "bfmzero_sphere":
      latent["z_sphere"] = z
    return actions, latent

  def act(self, obs: TensorDict, deterministic: bool = True) -> torch.Tensor:
    actions, _ = self.forward(obs, deterministic=deterministic)
    return actions

  def update_normalization(self, obs: TensorDict) -> None:
    self.encoder.update_normalization(obs)
    with torch.no_grad():
      mu, _ = self.encode(obs)
    decoder_obs = self._decoder_obs(obs, self._decode_latent(mu).detach())
    self.decoder.update_normalization(decoder_obs)

  def latent_cfg(self) -> dict[str, int | float | str]:
    return {
      "encoder_obs_group": self.encoder_obs_group,
      "decoder_obs_group": self.decoder_obs_group,
      "latent_dim": self.latent_dim,
      "action_dim": self.action_dim,
      "log_std_min": self.log_std_min,
      "log_std_max": self.log_std_max,
      "latent_mode": self.latent_mode,
      "sphere_radius": self.sphere_radius,
      "sphere_eps": self.sphere_eps,
    }

  def _decode_latent(self, z_raw: torch.Tensor) -> torch.Tensor:
    if self.latent_mode == "bfmzero_sphere":
      return self.spherical_project(
        z_raw,
        radius=self.sphere_radius,
        eps=self.sphere_eps,
      )
    return z_raw

  def _decoder_obs(self, obs: TensorDict, z: torch.Tensor) -> TensorDict:
    decoder_input = torch.cat([obs[self.decoder_obs_group], z], dim=-1)
    return TensorDict(
      {self.decoder_input_group: decoder_input},
      batch_size=list(obs.batch_size),
      device=obs.device,
    )


def build_student_model(
  obs: TensorDict,
  student_obs_group: str,
  action_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str,
  obs_normalization: bool = True,
) -> MLPModel:
  """Build the default MLP student policy."""
  return MLPModel(
    obs=obs,
    obs_groups={"actor": [student_obs_group]},
    obs_set="actor",
    output_dim=action_dim,
    hidden_dims=hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
  )


def build_latent_student_model(
  obs: TensorDict,
  encoder_obs_group: str,
  decoder_obs_group: str,
  action_dim: int,
  latent_dim: int,
  encoder_hidden_dims: tuple[int, ...],
  decoder_hidden_dims: tuple[int, ...],
  activation: str,
  obs_normalization: bool = True,
  log_std_min: float = -5.0,
  log_std_max: float = 2.0,
  latent_mode: str = "gaussian",
  sphere_radius: float = -1.0,
  sphere_eps: float = 1.0e-6,
) -> LatentDistillationModel:
  """Build the latent encoder/decoder student policy."""
  return LatentDistillationModel(
    obs=obs,
    encoder_obs_group=encoder_obs_group,
    decoder_obs_group=decoder_obs_group,
    action_dim=action_dim,
    latent_dim=latent_dim,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    activation=activation,
    obs_normalization=obs_normalization,
    log_std_min=log_std_min,
    log_std_max=log_std_max,
    latent_mode=latent_mode,
    sphere_radius=sphere_radius,
    sphere_eps=sphere_eps,
  )
