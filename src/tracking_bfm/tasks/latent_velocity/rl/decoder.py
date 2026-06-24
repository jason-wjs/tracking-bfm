"""Utilities for loading frozen latent decoder policies."""

import re
from pathlib import Path

import torch
from tensordict import TensorDict

from tracking_bfm.tasks.distillation.rl.models import build_latent_student_model


def load_latent_decoder(env, train_cfg: dict, device: str) -> torch.nn.Module:
  """Load a latent distillation checkpoint as a frozen decoder model."""
  checkpoint_path = train_cfg.get("latent_decoder_checkpoint_path", "")
  if not checkpoint_path:
    raise ValueError("latent_decoder_checkpoint_path must be provided.")
  if not Path(checkpoint_path).exists():
    raise FileNotFoundError(f"latent decoder checkpoint not found: {checkpoint_path}")

  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  if checkpoint.get("model_type") != "latent":
    raise ValueError("latent decoder checkpoint must be a latent distillation checkpoint.")
  state_dict = checkpoint.get("policy_state_dict")
  if state_dict is None:
    raise ValueError("latent decoder checkpoint is missing policy_state_dict.")

  latent_dim = int(train_cfg["latent_dim"])
  checkpoint_latent_cfg = checkpoint.get("latent_cfg", {})
  if checkpoint_latent_cfg and int(checkpoint_latent_cfg["latent_dim"]) != latent_dim:
    raise ValueError(
      "latent_dim does not match latent decoder checkpoint: "
      f"cfg={latent_dim}, checkpoint={checkpoint_latent_cfg['latent_dim']}"
    )

  obs = env.get_observations().to(device)
  proprio_obs_group = train_cfg.get("proprio_obs_group", "proprio_actor")
  encoder_obs_dim = infer_mlp_input_dim(state_dict, "encoder")
  model_obs = TensorDict(
    {
      "teacher_actor": torch.zeros(
        obs.batch_size[0],
        encoder_obs_dim,
        dtype=obs[proprio_obs_group].dtype,
        device=device,
      ),
      proprio_obs_group: obs[proprio_obs_group],
    },
    batch_size=list(obs.batch_size),
    device=device,
  )
  model = build_latent_student_model(
    obs=model_obs,
    encoder_obs_group="teacher_actor",
    decoder_obs_group=proprio_obs_group,
    action_dim=env.num_actions,
    latent_dim=latent_dim,
    encoder_hidden_dims=infer_hidden_dims(state_dict, "encoder"),
    decoder_hidden_dims=infer_hidden_dims(state_dict, "decoder"),
    activation="elu",
    obs_normalization=True,
  )
  model.load_state_dict(state_dict, strict=True)
  model.to(device)
  model.eval()
  return model


def infer_mlp_input_dim(state_dict: dict[str, torch.Tensor], module_name: str) -> int:
  key = f"{module_name}.mlp.0.weight"
  if key not in state_dict:
    raise ValueError(f"latent decoder checkpoint is missing {key}.")
  return int(state_dict[key].shape[1])


def infer_hidden_dims(
  state_dict: dict[str, torch.Tensor],
  module_name: str,
) -> tuple[int, ...]:
  linear_layers: list[tuple[int, torch.Tensor]] = []
  pattern = re.compile(rf"^{re.escape(module_name)}\.mlp\.(\d+)\.weight$")
  for key, value in state_dict.items():
    match = pattern.match(key)
    if match is not None:
      linear_layers.append((int(match.group(1)), value))
  if len(linear_layers) < 2:
    raise ValueError(
      f"latent decoder checkpoint has no MLP hidden layers for {module_name}."
    )
  linear_layers.sort(key=lambda item: item[0])
  return tuple(int(weight.shape[0]) for _, weight in linear_layers[:-1])
