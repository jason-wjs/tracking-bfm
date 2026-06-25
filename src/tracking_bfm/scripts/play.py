"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from tracking_bfm.motion_source import (
  AppliedMotionSource,
  MotionSourceSpec,
  apply_motion_source_to_command,
  is_motion_command_cfg,
  resolve_motion_source,
)
from tracking_bfm.scripts.cli_helpers import maybe_print_top_level_help


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  stochastic_policy: bool = False
  """Use stochastic action sampling for trained policies instead of deterministic inference."""
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  motion_path: str | None = None
  motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  show_reference_motion: bool = True
  """For distillation play, keep sparse student refs visible but optionally hide full reference motion."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


@dataclass(frozen=True, kw_only=True)
class PlayCliConfig(PlayConfig):
  """CLI-facing play config that also exposes env overrides."""

  env: ManagerBasedRlEnvCfg
  rl: RslRlBaseRunnerCfg

  @staticmethod
  def from_task(task_id: str) -> "PlayCliConfig":
    return PlayCliConfig(env=load_env_cfg(task_id, play=True), rl=load_rl_cfg(task_id))


def _configure_distillation_play_visualization(
  env_cfg, show_reference_motion: bool
) -> None:
  motion_cfg = env_cfg.commands.get("motion")
  student_sparse_vis_cfg = env_cfg.commands.get("student_sparse_vis")
  if motion_cfg is None or student_sparse_vis_cfg is None:
    return
  motion_cfg.debug_vis = show_reference_motion
  student_sparse_vis_cfg.debug_vis = True


def _motion_source_spec_for_play(cfg: PlayConfig) -> MotionSourceSpec:
  explicit_sources = [
    source
    for source in (cfg.motion_file, cfg.motion_path, cfg.registry_name)
    if source is not None
  ]
  if len(explicit_sources) > 1:
    raise ValueError(
      "Provide only one explicit motion source: motion_file, motion_path, or registry_name."
    )
  if cfg.motion_file is not None:
    return MotionSourceSpec(motion_file=cfg.motion_file)
  if cfg.motion_path is not None:
    return MotionSourceSpec(motion_path=cfg.motion_path)
  if cfg.registry_name is not None:
    return MotionSourceSpec(wandb_registry_name=cfg.registry_name)
  if cfg.wandb_run_path is not None:
    return MotionSourceSpec(wandb_run_path=cfg.wandb_run_path)
  return MotionSourceSpec()


def _has_motion_source(spec: MotionSourceSpec) -> bool:
  return any(
    source is not None
    for source in (
      spec.motion_file,
      spec.motion_path,
      spec.wandb_run_path,
      spec.wandb_registry_name,
    )
  )


def _apply_tracking_motion_source(
  env_cfg: Any,
  cfg: PlayConfig,
  *,
  dummy_mode: bool,
) -> AppliedMotionSource | None:
  motion_cmd = env_cfg.commands.get("motion")
  if not is_motion_command_cfg(motion_cmd):
    return None

  motion_cmd.motion_type = cfg.motion_type
  if cfg._demo_mode and hasattr(motion_cmd, "sampling_mode"):
    motion_cmd.sampling_mode = "uniform"

  source_spec = _motion_source_spec_for_play(cfg)
  if (dummy_mode or cfg.checkpoint_file is not None) and not _has_motion_source(
    source_spec
  ):
    raise ValueError("Tracking tasks require a motion source.")

  source = resolve_motion_source(source_spec)
  applied_source = apply_motion_source_to_command(motion_cmd, source)
  if applied_source is None:
    return None
  if applied_source.motion_file is not None:
    print(f"[INFO]: Using motion file: {applied_source.motion_file}")
  elif applied_source.motion_path is not None:
    print(f"[INFO]: Using motion path: {applied_source.motion_path}")
  return applied_source


def _get_trained_policy(runner, device: str, stochastic: bool):
  if not stochastic:
    return runner.get_inference_policy(device=device)

  actor = runner.alg.get_policy()
  if hasattr(actor, "eval"):
    actor.eval()

  class StochasticPolicy:
    def __call__(self, obs) -> torch.Tensor:
      with torch.no_grad():
        return actor(obs, stochastic_output=True)

  return StochasticPolicy()


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = (
    deepcopy(cfg.env)
    if isinstance(cfg, PlayCliConfig)
    else load_env_cfg(task_id, play=True)
  )
  agent_cfg = (
    deepcopy(cfg.rl) if isinstance(cfg, PlayCliConfig) else load_rl_cfg(task_id)
  )

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  _configure_distillation_play_visualization(
    env_cfg, show_reference_motion=cfg.show_reference_motion
  )
  _apply_tracking_motion_source(env_cfg, cfg, dummy_mode=DUMMY_MODE)

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = _get_trained_policy(
      runner, device=device, stochastic=cfg.stochastic_policy
    )
    env = runner.env

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _get_trained_policy(
        _ckpt_runner,
        device=device,
        stochastic=cfg.stochastic_policy,
      )

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("tracking-bfm-play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import tracking_bfm.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  default_args = PlayCliConfig.from_task(chosen_task)
  args = tyro.cli(
    PlayCliConfig,
    args=remaining_args,
    default=default_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, default_args

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
