"""Collect and visualize latent distributions from latent distillation checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from tracking_bfm.scripts.cli_helpers import maybe_print_top_level_help
from tracking_bfm.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)


@dataclass(frozen=True)
class LatentSpaceAnalysisConfig:
  checkpoint_file: str
  output_dir: str
  motion_path: str | None = None
  num_envs: int = 512
  num_points: int = 50_000
  device: str | None = None
  deterministic: bool = True
  sampling_mode: str | None = "uniform"
  motion_history_steps: int | None = None
  motion_future_steps: int | None = None
  proprio_history_length: int | None = None
  sim_njmax: int | None = 600
  """Override analysis env constraint buffer size. Set None to keep task default."""
  sim_nconmax: int | None = None
  """Override analysis env contact buffer size. Set None to keep task default."""
  max_plot_points: int = 10_000
  plot_range: tuple[float, float] | None = (-20.0, 20.0)
  """Fixed plotting range for latent visualizations; set to None for autoscale."""


def _latent_policy_obs(policy, obs: TensorDict) -> TensorDict:
  return TensorDict(
    {
      policy.encoder_obs_group: obs[policy.encoder_obs_group],
      policy.decoder_obs_group: obs[policy.decoder_obs_group],
    },
    batch_size=list(obs.batch_size),
    device=obs.device,
  )


def _apply_latent_analysis_overrides(env_cfg, cfg: LatentSpaceAnalysisConfig) -> None:
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.sim_njmax is not None:
    env_cfg.sim.njmax = int(cfg.sim_njmax)
  if cfg.sim_nconmax is not None:
    env_cfg.sim.nconmax = int(cfg.sim_nconmax)

  motion_cfg = env_cfg.commands.get("motion")
  motion_overrides = (
    cfg.motion_path is not None
    or cfg.sampling_mode is not None
    or cfg.motion_history_steps is not None
    or cfg.motion_future_steps is not None
  )
  if motion_overrides and motion_cfg is None:
    raise ValueError("Latent analysis expects an env command named 'motion'.")
  if motion_cfg is not None:
    if cfg.motion_path is not None:
      motion_cfg.motion_path = cfg.motion_path
    if cfg.sampling_mode is not None:
      motion_cfg.sampling_mode = cfg.sampling_mode
    if cfg.motion_history_steps is not None:
      motion_cfg.history_steps = cfg.motion_history_steps
    if cfg.motion_future_steps is not None:
      motion_cfg.future_steps = cfg.motion_future_steps

  if cfg.proprio_history_length is not None:
    proprio_group = env_cfg.observations.get("proprio_actor")
    if proprio_group is None:
      raise ValueError(
        "Latent analysis expects an observation group named 'proprio_actor'."
      )
    for term_name in (
      "projected_gravity",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ):
      term = proprio_group.terms.get(term_name)
      if term is None:
        raise ValueError(f"proprio_actor is missing term {term_name!r}.")
      term.history_length = cfg.proprio_history_length


@torch.no_grad()
def collect_latent_batches(
  *,
  env,
  policy,
  num_points: int,
  deterministic: bool,
  device: torch.device,
) -> dict[str, torch.Tensor]:
  """Collect latent samples while stepping parallel auto-reset environments."""
  policy.eval()
  obs = env.get_observations().to(device)
  collected: dict[str, list[torch.Tensor]] = {
    "mu": [],
    "log_std": [],
    "z": [],
    "dones": [],
    "step": [],
  }
  collected_points = 0
  step = 0

  while collected_points < num_points:
    policy_obs = _latent_policy_obs(policy, obs)
    mu, log_std = policy.encode(policy_obs)
    z = mu if deterministic else mu + torch.randn_like(mu) * torch.exp(log_std)
    if hasattr(policy, "decode"):
      actions = policy.decode(policy_obs, z)
    else:
      actions = policy.act(policy_obs, deterministic=deterministic)
    next_obs, _, dones, _ = env.step(actions.to(env.device))

    remaining = num_points - collected_points
    take = min(mu.shape[0], remaining)
    collected["mu"].append(mu[:take].detach().cpu())
    collected["log_std"].append(log_std[:take].detach().cpu())
    collected["z"].append(z[:take].detach().cpu())
    collected["dones"].append(dones[:take].detach().cpu().bool())
    collected["step"].append(torch.full((take,), step, dtype=torch.long))

    collected_points += take
    step += 1
    obs = next_obs.to(device)

  return {key: torch.cat(value, dim=0) for key, value in collected.items()}


def _normalize_to_unit_sphere(
  samples: torch.Tensor, eps: float = 1.0e-12
) -> torch.Tensor:
  norms = torch.linalg.vector_norm(samples, dim=-1, keepdim=True)
  return samples / torch.clamp(norms, min=eps)


def _tsne_embedding(
  samples: torch.Tensor,
  n_components: int,
  *,
  random_state: int = 0,
  perplexity: float = 30.0,
) -> np.ndarray:
  if samples.shape[0] < 2:
    return np.zeros((samples.shape[0], n_components), dtype=np.float32)

  try:
    from sklearn.manifold import TSNE
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "t-SNE plotting requires scikit-learn. Install it with `uv add scikit-learn` "
      "or add scikit-learn to the project dependencies."
    ) from exc

  effective_perplexity = min(float(perplexity), max(1.0, (samples.shape[0] - 1) / 3.0))
  tsne = TSNE(
    n_components=n_components,
    perplexity=effective_perplexity,
    init="random",
    learning_rate="auto",
    random_state=random_state,
    method="barnes_hut",
  )
  return tsne.fit_transform(samples.detach().cpu().numpy())


def _effective_rank(eigenvalues: torch.Tensor) -> float:
  eigenvalues = torch.clamp(eigenvalues, min=0.0)
  total = eigenvalues.sum()
  if total <= 0:
    return 0.0
  probs = eigenvalues / total
  entropy = -(probs * torch.log(probs + 1.0e-12)).sum()
  return float(torch.exp(entropy).item())


def _safe_covariance_eigenvalues(cov: torch.Tensor) -> torch.Tensor:
  if cov.numel() == 0:
    return cov.new_zeros((0,))
  cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
  cov = 0.5 * (cov + cov.T)
  cov64 = cov.to(dtype=torch.float64)
  try:
    eigenvalues = torch.linalg.eigvalsh(cov64)
  except torch.linalg.LinAlgError:
    jitter = torch.eye(cov64.shape[0], dtype=cov64.dtype, device=cov64.device) * 1.0e-8
    try:
      eigenvalues = torch.linalg.eigvalsh(cov64 + jitter)
    except torch.linalg.LinAlgError:
      return cov.new_zeros((cov.shape[0],))
  return eigenvalues.to(dtype=cov.dtype).flip(0)


def _finite_latent_rows(*tensors: torch.Tensor) -> torch.Tensor:
  masks = []
  for tensor in tensors:
    masks.append(torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1))
  finite = masks[0]
  for mask in masks[1:]:
    finite &= mask
  return finite


def summarize_latents(latents: dict[str, torch.Tensor]) -> dict[str, Any]:
  z = latents["z"].float()
  mu = latents["mu"].float()
  log_std = latents["log_std"].float()
  finite_rows = _finite_latent_rows(z, mu, log_std)
  finite_count = int(finite_rows.count_nonzero().item())
  if finite_count > 0:
    z_stats = z[finite_rows]
    mu_stats = mu[finite_rows]
    log_std_stats = log_std[finite_rows]
  else:
    z_stats = torch.zeros((1, z.shape[-1]), dtype=z.dtype, device=z.device)
    mu_stats = torch.zeros_like(z_stats)
    log_std_stats = torch.zeros_like(z_stats)
  radius = z_stats.norm(dim=-1)
  centered = z_stats - z_stats.mean(dim=0, keepdim=True)
  cov = centered.T @ centered / max(z_stats.shape[0] - 1, 1)
  diag_mask = torch.eye(cov.shape[0], dtype=torch.bool)
  offdiag = cov[~diag_mask]
  eigenvalues = _safe_covariance_eigenvalues(cov)
  prior = torch.randn_like(z_stats)
  return {
    "num_points": int(z.shape[0]),
    "finite_num_points": finite_count,
    "nonfinite_num_points": int(z.shape[0] - finite_count),
    "latent_dim": int(z.shape[-1]),
    "mu_mean_norm": float(mu_stats.mean(dim=0).norm().item()),
    "z_mean_norm": float(z_stats.mean(dim=0).norm().item()),
    "z_std_mean": float(z_stats.std(dim=0, unbiased=False).mean().item()),
    "z_std_min": float(z_stats.std(dim=0, unbiased=False).min().item()),
    "z_std_max": float(z_stats.std(dim=0, unbiased=False).max().item()),
    "latent_std_mean": float(torch.exp(log_std_stats).mean().item()),
    "cov_offdiag_mean_abs": float(offdiag.abs().mean().item())
    if offdiag.numel()
    else 0.0,
    "radius_mean": float(radius.mean().item()),
    "radius_std": float(radius.std(unbiased=False).item()),
    "radius_q05": float(torch.quantile(radius, 0.05).item()),
    "radius_q50": float(torch.quantile(radius, 0.50).item()),
    "radius_q95": float(torch.quantile(radius, 0.95).item()),
    "effective_rank": _effective_rank(eigenvalues),
    "prior_radius_mean": float(prior.norm(dim=-1).mean().item()),
    "done_ratio": float(latents["dones"].float().mean().item()),
  }


def _coerce_plot_range(
  plot_range: tuple[float, float] | None,
) -> tuple[float, float] | None:
  if plot_range is None:
    return None
  lo, hi = float(plot_range[0]), float(plot_range[1])
  if not lo < hi:
    raise ValueError(f"plot_range must be increasing, got {plot_range}")
  return lo, hi


def save_latent_plots(
  latents: dict[str, torch.Tensor],
  output_dir: Path,
  max_plot_points: int,
  plot_range: tuple[float, float] | None = (-20.0, 20.0),
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output_dir.mkdir(parents=True, exist_ok=True)
  z = latents["z"].float()
  axis_range = _coerce_plot_range(plot_range)
  plot_count = min(max_plot_points, z.shape[0])
  plot_idx = torch.linspace(0, z.shape[0] - 1, plot_count).long()
  sphere_z = _normalize_to_unit_sphere(z[plot_idx])
  colors = latents.get("step", torch.arange(z.shape[0]))[plot_idx].cpu().numpy()

  for old_plot_name in (
    "pca_z_vs_prior.png",
    "pca_mu.png",
    "radius_hist.png",
    "cov_heatmap.png",
    "pca_spectrum.png",
  ):
    (output_dir / old_plot_name).unlink(missing_ok=True)

  tsne_2d = _tsne_embedding(sphere_z, n_components=2)

  plt.figure(figsize=(7, 6))
  scatter = plt.scatter(
    tsne_2d[:, 0],
    tsne_2d[:, 1],
    c=colors,
    cmap="viridis",
    s=3,
    alpha=0.65,
  )
  plt.xlabel("t-SNE 1")
  plt.ylabel("t-SNE 2")
  if axis_range is not None:
    plt.xlim(axis_range)
    plt.ylim(axis_range)
  plt.colorbar(scatter, label="rollout step")
  plt.tight_layout()
  plt.savefig(output_dir / "tsne_sphere_z_2d.png", dpi=180)
  plt.close()

  tsne_3d = _tsne_embedding(sphere_z, n_components=3)
  fig = plt.figure(figsize=(8, 7))
  ax = fig.add_subplot(111, projection="3d")
  scatter = ax.scatter(
    tsne_3d[:, 0],
    tsne_3d[:, 1],
    tsne_3d[:, 2],
    c=colors,
    cmap="viridis",
    s=3,
    alpha=0.65,
  )
  ax.set_xlabel("t-SNE 1")
  ax.set_ylabel("t-SNE 2")
  ax.set_zlabel("t-SNE 3")
  if axis_range is not None:
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_zlim(axis_range)
  fig.colorbar(scatter, ax=ax, label="rollout step", shrink=0.75)
  plt.tight_layout()
  plt.savefig(output_dir / "tsne_sphere_z_3d.png", dpi=180)
  plt.close()


def run_analysis(task_id: str, cfg: LatentSpaceAnalysisConfig) -> Path:
  configure_torch_backends()
  device = torch.device(
    cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  )

  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  _apply_latent_analysis_overrides(env_cfg, cfg)
  motion_cfg = env_cfg.commands.get("motion")
  if cfg.motion_path is not None and not isinstance(motion_cfg, MultiMotionCommandCfg):
    raise ValueError("Latent analysis currently expects a multi-motion tracking task.")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    raise ValueError(f"Task {task_id} does not define a runner class")
  runner = runner_cls(wrapped_env, asdict(agent_cfg), log_dir=None, device=str(device))
  runner.load(
    cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=str(device)
  )
  policy = runner.student_policy
  if not all(
    hasattr(policy, name)
    for name in ("encode", "decode", "encoder_obs_group", "decoder_obs_group")
  ):
    raise ValueError("Checkpoint policy is not a latent distillation policy")

  try:
    latents = collect_latent_batches(
      env=wrapped_env,
      policy=policy,
      num_points=cfg.num_points,
      deterministic=cfg.deterministic,
      device=device,
    )
  finally:
    env.close()

  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_dir / "latents.npz", **{k: v.numpy() for k, v in latents.items()}
  )
  summary = summarize_latents(latents)
  (output_dir / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
  )
  save_latent_plots(
    latents,
    output_dir,
    max_plot_points=cfg.max_plot_points,
    plot_range=cfg.plot_range,
  )
  print(json.dumps(summary, indent=2))
  print(f"[INFO] Latent analysis written to {output_dir}")
  return output_dir


def main() -> None:
  maybe_print_top_level_help("tracking-bfm-analyze-latent-space")
  import tracking_bfm.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(
    LatentSpaceAnalysisConfig,
    args=remaining_args,
    prog=f"analyze-latent-space {task_id}",
  )
  if cfg.device is not None:
    os.environ.setdefault("MUJOCO_GL", "egl")
  run_analysis(task_id, cfg)


if __name__ == "__main__":
  main()
