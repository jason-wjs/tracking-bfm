"""Batch motion filtering for multi-motion tracking datasets."""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from tracking_bfm.motion_source import (
  MotionSourceSpec,
  apply_motion_source_to_command,
  collect_motion_files,
  resolve_motion_source,
  shard_motion_files,
)

MotionType = Literal["isaaclab", "mujoco"]
ViewerMode = Literal["none", "auto", "native", "viser"]
GpuIds = list[int] | Literal["all"] | None


@dataclass(frozen=True)
class EvaluateConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  motion_path: str | None = None
  motion_type: MotionType = "isaaclab"
  history_steps: int | None = None
  future_steps: int | None = None
  num_envs: int = 1024
  device: str | None = None
  failure_threshold: float = 0.9
  output_file: str = "filtered_motions.json"
  viewer: ViewerMode = "none"
  torchrunx_log_dir: str | None = None
  gpu_ids: GpuIds = None


def _invert_reindex(reindex: list[int]) -> list[int]:
  if sorted(reindex) != list(range(len(reindex))):
    raise ValueError(f"Expected a permutation, got: {reindex}")
  inverse = [0] * len(reindex)
  for output_index, input_index in enumerate(reindex):
    inverse[input_index] = output_index
  return inverse


def _load_multi_motion_command_symbols() -> tuple[list[int], list[int], type[Any]]:
  try:
    from tracking_bfm.tasks.tracking.mdp.multi_motion_command import (
      _ISAACLAB_TO_MUJOCO_BODY_REINDEX,
      _ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    )
    from tracking_bfm.tasks.tracking.mdp.multi_motion_command import (
      MotionCommandCfg as MultiMotionCommandCfg,
    )
  except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("tracking_bfm.tasks.tracking"):
      raise RuntimeError(
        "Multi-motion command support is not available. Run the tracking task "
        "migration before executing data processing commands."
      ) from exc
    raise
  return (
    list(_ISAACLAB_TO_MUJOCO_JOINT_REINDEX),
    list(_ISAACLAB_TO_MUJOCO_BODY_REINDEX),
    MultiMotionCommandCfg,
  )


def _make_tensor_dict(data: Any, *, batch_size: list[int]) -> Any:
  from tensordict import TensorDict

  return TensorDict(data, batch_size=batch_size)


def motion_sequence_complete(env: Any, command_name: str) -> Any:
  """Terminate an evaluation episode exactly at the end of its assigned motion."""

  command = cast(Any, env.command_manager.get_term(command_name))
  return env.episode_length_buf >= command.motion_length


def _prepare_filtering_env_cfg(env_cfg: Any) -> Any:
  """Disable stochastic evaluation effects while preserving failure terminations."""

  from mjlab.managers.termination_manager import TerminationTermCfg

  from tracking_bfm.tasks.tracking.mdp.commands import MotionCommandCfg

  _, _, MultiMotionCommandCfg = _load_multi_motion_command_symbols()

  env_cfg = deepcopy(env_cfg)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg)):
    raise ValueError("The selected task is not a tracking task with a motion command.")
  if not isinstance(motion_cmd, MultiMotionCommandCfg):
    raise ValueError("Data filtering requires a multi-motion tracking task.")

  motion_cmd.sampling_mode = "start"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.if_log_metrics = False

  if "actor" in env_cfg.observations:
    env_cfg.observations["actor"].enable_corruption = False
  if "critic" in env_cfg.observations:
    env_cfg.observations["critic"].enable_corruption = False

  for event_name in (
    "push_robot",
    "base_com",
    "base_inertia",
    "body_inertia",
    "encoder_bias",
    "foot_friction",
  ):
    env_cfg.events.pop(event_name, None)

  env_cfg.episode_length_s = int(1e9)
  env_cfg.terminations.pop("time_out", None)
  env_cfg.terminations["motion_complete"] = TerminationTermCfg(
    func=motion_sequence_complete,
    time_out=True,
    params={"command_name": "motion"},
  )
  return env_cfg


def _configure_motion_command(
  motion_cmd: Any,
  *,
  motion_path: str,
  motion_type: MotionType,
  history_steps: int | None,
  future_steps: int | None,
) -> None:
  """Apply CLI motion overrides to the filtering motion command."""

  source = resolve_motion_source(
    MotionSourceSpec(motion_path=motion_path),
    required=True,
  )
  apply_motion_source_to_command(motion_cmd, source)
  motion_cmd.motion_type = motion_type
  if history_steps is not None:
    motion_cmd.history_steps = history_steps
  if future_steps is not None:
    motion_cmd.future_steps = future_steps


def _collect_motion_files(motion_root: str) -> list[Path]:
  return collect_motion_files(motion_root)


def _resolve_motion_root(cfg: EvaluateConfig | Any) -> str:
  if cfg.motion_path is not None:
    return cfg.motion_path
  if cfg.motion_path is None and cfg.wandb_run_path is None:
    raise ValueError(
      "Provide --motion-path explicitly, or provide --wandb-run-path so the motion "
      "artifact can be resolved."
    )

  source = resolve_motion_source(
    MotionSourceSpec(
      motion_path=cfg.motion_path,
      wandb_run_path=(
        str(cfg.wandb_run_path) if cfg.wandb_run_path is not None else None
      ),
    ),
    required=True,
  )
  assert source is not None
  return str(source.path)


def _resolve_checkpoint_path(task_id: str, cfg: EvaluateConfig | Any) -> tuple[Path, str]:
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.utils.os import get_wandb_checkpoint_path

  agent_cfg = load_rl_cfg(task_id)
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()

  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path, str(checkpoint_path)

  if cfg.wandb_run_path is None:
    raise ValueError(
      "Provide --checkpoint-file or --wandb-run-path to resolve the evaluation "
      "checkpoint."
    )

  checkpoint_path, _ = get_wandb_checkpoint_path(
    log_root_path,
    Path(cfg.wandb_run_path),
    cfg.wandb_checkpoint_name,
  )
  return checkpoint_path, str(checkpoint_path)


def _update_relative_body_poses(command: Any) -> None:
  """Mirror the command's relative-body update without advancing time steps."""

  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

  anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(
    1,
    len(command.cfg.body_names),
    1,
  )
  anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(
    1,
    len(command.cfg.body_names),
    1,
  )
  robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(
    1,
    len(command.cfg.body_names),
    1,
  )
  robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(
    1,
    len(command.cfg.body_names),
    1,
  )

  delta_pos_w = robot_anchor_pos_w_repeat.clone()
  delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
  delta_ori_w = yaw_quat(
    quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
  )

  command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
  command.body_pos_relative_w = delta_pos_w + quat_apply(
    delta_ori_w,
    command.body_pos_w - anchor_pos_w_repeat,
  )


def _recompute_observations(env: Any, command: Any) -> None:
  env.scene.write_data_to_sim()
  env.sim.forward()
  _update_relative_body_poses(command)
  env.sim.sense()
  env.observation_manager._obs_buffer = None
  env.obs_buf = env.observation_manager.compute(update_history=False)


def _assign_motion_indices(
  env: Any,
  command: Any,
  env_ids: Any,
  motion_indices: Any,
) -> None:
  import torch

  if env_ids.numel() == 0:
    return

  command.motion_idx[env_ids] = motion_indices
  command.motion_length[env_ids] = command.motion.file_lengths[motion_indices]
  command.time_steps[env_ids] = 0

  root_pos = command.body_pos_w[env_ids, 0].clone()
  root_ori = command.body_quat_w[env_ids, 0].clone()
  root_lin_vel = command.body_lin_vel_w[env_ids, 0].clone()
  root_ang_vel = command.body_ang_vel_w[env_ids, 0].clone()
  joint_pos = command.joint_pos[env_ids].clone()
  joint_vel = command.joint_vel[env_ids].clone()

  soft_joint_pos_limits = command.robot.data.soft_joint_pos_limits[env_ids]
  joint_pos = torch.clip(
    joint_pos,
    soft_joint_pos_limits[:, :, 0],
    soft_joint_pos_limits[:, :, 1],
  )

  command.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  command.robot.write_root_state_to_sim(
    torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
    env_ids=env_ids,
  )
  command.robot.clear_state(env_ids=env_ids)
  _recompute_observations(env, command)


def build_filter_report(
  *,
  task_id: str,
  motion_root: str,
  checkpoint: str,
  threshold: float,
  records: list[dict[str, Any]],
  rank: int,
  world_size: int,
) -> dict[str, Any]:
  sorted_records = sorted(records, key=lambda item: item["motion_index"])
  failed_records = [
    record for record in sorted_records if record["completion_ratio"] < threshold
  ]
  total_motion_count = len(sorted_records)
  failed_motion_count = len(failed_records)
  failed_motion_ratio = (
    failed_motion_count / total_motion_count if total_motion_count > 0 else 0.0
  )

  return {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": task_id,
    "motion_root": motion_root,
    "checkpoint": checkpoint,
    "failure_threshold": threshold,
    "rank": rank,
    "world_size": world_size,
    "total_motion_count": total_motion_count,
    "failed_motion_count": failed_motion_count,
    "failed_motion_ratio": failed_motion_ratio,
    "failed_motions": failed_records,
  }


def _load_policy(task_id: str, env: Any, device: str, checkpoint_path: Path) -> Any:
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def _shard_motion_files(
  motion_files: list[Path],
  world_size: int,
  rank: int,
) -> list[Path]:
  return shard_motion_files(motion_files, world_size=world_size, rank=rank)


def _runtime_rank_context(cfg: EvaluateConfig | Any) -> tuple[str, int, int]:
  import torch

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  rank = int(os.environ.get("RANK", "0"))

  if cuda_visible == "":
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  device = f"cuda:{local_rank}"
  return device, rank, world_size


def _rank_output_path(output_file: str, rank: int, world_size: int) -> Path:
  output_path = Path(output_file)
  if world_size <= 1:
    return output_path
  return output_path.with_name(
    f"{output_path.stem}.rank{rank:02d}-of-{world_size:02d}"
    f"{output_path.suffix or '.json'}"
  )


def _prepare_launch_cfg(cfg: EvaluateConfig | Any) -> EvaluateConfig | Any:
  if (
    isinstance(cfg, EvaluateConfig)
    and cfg.gpu_ids is not None
    and cfg.viewer != "none"
  ):
    print(
      "[INFO] gpu_ids provided; forcing viewer=none to avoid multi-process viewer "
      "conflicts."
    )
    return replace(cfg, viewer="none")
  return cfg


def merge_filter_reports(report_paths: list[Path], output_path: Path) -> dict[str, Any]:
  reports = []
  for report_path in sorted(report_paths):
    with report_path.open("r", encoding="utf-8") as file:
      reports.append(json.load(file))
  if not reports:
    raise ValueError("No partial reports found to merge.")

  merged_failed_motions: list[dict[str, Any]] = []
  total_motion_count = 0
  failed_motion_count = 0

  for report in reports:
    total_motion_count += int(report["total_motion_count"])
    failed_motion_count += int(report["failed_motion_count"])
    merged_failed_motions.extend(report.get("failed_motions", []))

  merged_failed_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_report = {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": reports[0]["task_id"],
    "motion_root": reports[0]["motion_root"],
    "checkpoint": reports[0]["checkpoint"],
    "failure_threshold": reports[0]["failure_threshold"],
    "world_size": max(int(report.get("world_size", 1)) for report in reports),
    "report_parts": len(reports),
    "total_motion_count": total_motion_count,
    "failed_motion_count": failed_motion_count,
    "failed_motion_ratio": (
      failed_motion_count / total_motion_count if total_motion_count > 0 else 0.0
    ),
    "failed_motions": merged_failed_motions,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(merged_report, file, indent=2)
  return merged_report


class FilteringEvalEnv:
  """VecEnv wrapper that records completed motions and refills idle env slots."""

  def __init__(
    self,
    env: Any,
    motion_files: list[Path],
    command: Any,
    rank: int,
  ) -> None:
    import torch

    self._env = env
    self._motion_files = motion_files
    self._command = command
    self._rank = rank
    self.records: list[dict[str, Any]] = []
    self.finished = False
    self._next_motion_index = 0
    self._active_mask = torch.zeros(
      self.unwrapped.num_envs,
      dtype=torch.bool,
      device=self.unwrapped.device,
    )
    self._assigned_motion_ids = torch.full(
      (self.unwrapped.num_envs,),
      -1,
      dtype=torch.long,
      device=self.unwrapped.device,
    )
    self._assigned_motion_lengths = torch.zeros(
      self.unwrapped.num_envs,
      dtype=torch.long,
      device=self.unwrapped.device,
    )
    env_ids = torch.arange(
      self.unwrapped.num_envs,
      dtype=torch.long,
      device=self.unwrapped.device,
    )
    self._assign_available(env_ids)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._env, name)

  @property
  def unwrapped(self) -> Any:
    return self._env.unwrapped

  @property
  def cfg(self) -> Any:
    return self._env.cfg

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  def reset(self) -> tuple[Any, dict[str, Any]]:
    import torch

    obs, extras = self._env.reset()
    self.records.clear()
    self.finished = False
    self._next_motion_index = 0
    self._active_mask.zero_()
    self._assigned_motion_ids.fill_(-1)
    self._assigned_motion_lengths.zero_()
    env_ids = torch.arange(
      self.unwrapped.num_envs,
      dtype=torch.long,
      device=self.unwrapped.device,
    )
    self._assign_available(env_ids)
    return _make_tensor_dict(self.unwrapped.obs_buf, batch_size=[self.num_envs]), extras

  def get_observations(self) -> Any:
    return _make_tensor_dict(self.unwrapped.obs_buf, batch_size=[self.num_envs])

  def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    pre_episode_lengths = self.unwrapped.episode_length_buf.clone()
    pre_motion_ids = self._assigned_motion_ids.clone()
    pre_motion_lengths = self._assigned_motion_lengths.clone()
    pre_active_mask = self._active_mask.clone()

    obs, rew, dones, extras = self._env.step(actions)

    done_env_ids = torch.where(dones.bool() & pre_active_mask)[0]
    if done_env_ids.numel() == 0:
      self.finished = bool(
        self._next_motion_index >= len(self._motion_files)
        and not self._active_mask.any()
      )
      return obs, rew, dones, extras

    terminated = self.unwrapped.reset_terminated[done_env_ids].detach().cpu()
    truncated = self.unwrapped.reset_time_outs[done_env_ids].detach().cpu()

    for idx, env_id in enumerate(done_env_ids.tolist()):
      motion_index = int(pre_motion_ids[env_id].item())
      motion_length = int(pre_motion_lengths[env_id].item())
      completed_steps = min(int(pre_episode_lengths[env_id].item()) + 1, motion_length)
      completion_ratio = completed_steps / float(max(motion_length, 1))
      self.records.append(
        {
          "motion_index": motion_index,
          "path": str(self._motion_files[motion_index].resolve()),
          "rank": self._rank,
          "completed_steps": completed_steps,
          "total_steps": motion_length,
          "completion_ratio": completion_ratio,
          "terminated": bool(terminated[idx].item()),
          "truncated": bool(truncated[idx].item()),
        }
      )

    self._assign_available(done_env_ids.to(device=self.unwrapped.device))
    self.finished = bool(
      self._next_motion_index >= len(self._motion_files) and not self._active_mask.any()
    )

    return obs, rew, dones, extras

  def close(self) -> None:
    self._env.close()

  def _assign_available(self, env_ids: Any) -> None:
    import torch

    if env_ids.numel() == 0:
      return

    remaining = len(self._motion_files) - self._next_motion_index
    if remaining <= 0:
      self._active_mask[env_ids] = False
      self._assigned_motion_ids[env_ids] = -1
      self._assigned_motion_lengths[env_ids] = 0
      return

    assign_count = min(int(env_ids.numel()), remaining)
    assign_env_ids = env_ids[:assign_count]
    motion_indices = torch.arange(
      self._next_motion_index,
      self._next_motion_index + assign_count,
      device=self.unwrapped.device,
      dtype=torch.long,
    )
    self._next_motion_index += assign_count

    _assign_motion_indices(
      self.unwrapped,
      self._command,
      assign_env_ids,
      motion_indices,
    )
    self._active_mask[assign_env_ids] = True
    self._assigned_motion_ids[assign_env_ids] = motion_indices
    self._assigned_motion_lengths[assign_env_ids] = self._command.motion.file_lengths[
      motion_indices
    ]

    if assign_count < int(env_ids.numel()):
      idle_env_ids = env_ids[assign_count:]
      self._active_mask[idle_env_ids] = False
      self._assigned_motion_ids[idle_env_ids] = -1
      self._assigned_motion_lengths[idle_env_ids] = 0


def _resolve_viewer_backend(viewer: ViewerMode) -> Literal["native", "viser"] | None:
  if viewer == "none":
    return None
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"
  return viewer


def _run_viewer_loop(viewer: Any) -> None:
  viewer._interrupted = False
  viewer.setup()
  now = time.perf_counter()
  viewer._stats_last_time = now
  viewer._last_tick_time = now
  try:
    while (
      viewer.is_running()
      and not cast(Any, viewer.env).finished
      and not viewer._interrupted
    ):
      if not viewer.tick():
        time.sleep(0.001)
      viewer._update_stats()
  finally:
    viewer.close()


def _run_viewer_evaluate(
  task_id: str,
  cfg: EvaluateConfig,
  motion_files: list[Path],
  checkpoint_path: Path,
  checkpoint_label: str,
  motion_root: str,
) -> dict[str, Any]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  device, rank, world_size = _runtime_rank_context(cfg)
  env_cfg = _prepare_filtering_env_cfg(load_env_cfg(task_id, play=False))
  motion_cmd = env_cfg.commands["motion"]
  _, _, MultiMotionCommandCfg = _load_multi_motion_command_symbols()
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  _configure_motion_command(
    motion_cmd,
    motion_path=motion_root,
    motion_type=cfg.motion_type,
    history_steps=cfg.history_steps,
    future_steps=cfg.future_steps,
  )
  env_cfg.scene.num_envs = min(cfg.num_envs, len(motion_files))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, device, checkpoint_path)
  command = cast(Any, env.command_manager.get_term("motion"))
  filtering_env = FilteringEvalEnv(vec_env, motion_files, command, rank=rank)

  viewer_backend = _resolve_viewer_backend(cfg.viewer)
  assert viewer_backend is not None
  viewer = (
    NativeMujocoViewer(filtering_env, policy)
    if viewer_backend == "native"
    else ViserPlayViewer(filtering_env, policy)
  )
  _run_viewer_loop(viewer)

  report = build_filter_report(
    task_id=task_id,
    motion_root=motion_root,
    checkpoint=checkpoint_label,
    threshold=cfg.failure_threshold,
    records=filtering_env.records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Failed: {report['failed_motion_count']} "
    f"({report['failed_motion_ratio']:.2%})."
  )
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, Any]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from tqdm import tqdm

  configure_torch_backends()
  device, rank, world_size = _runtime_rank_context(cfg)
  motion_root = _resolve_motion_root(cfg)
  motion_files = _shard_motion_files(
    _collect_motion_files(motion_root),
    world_size,
    rank,
  )
  checkpoint_path, checkpoint_label = _resolve_checkpoint_path(task_id, cfg)
  if cfg.viewer != "none":
    return _run_viewer_evaluate(
      task_id,
      cfg,
      motion_files,
      checkpoint_path,
      checkpoint_label,
      motion_root,
    )

  env_cfg = _prepare_filtering_env_cfg(load_env_cfg(task_id, play=False))
  motion_cmd = env_cfg.commands["motion"]
  _, _, MultiMotionCommandCfg = _load_multi_motion_command_symbols()
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  _configure_motion_command(
    motion_cmd,
    motion_path=motion_root,
    motion_type=cfg.motion_type,
    history_steps=cfg.history_steps,
    future_steps=cfg.future_steps,
  )
  env_cfg.scene.num_envs = min(cfg.num_envs, len(motion_files))

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  policy = _load_policy(task_id, vec_env, device, checkpoint_path)
  command = cast(Any, env.command_manager.get_term("motion"))
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  assigned_motion_ids = torch.full(
    (env.num_envs,),
    -1,
    dtype=torch.long,
    device=env.device,
  )
  assigned_motion_lengths = torch.zeros(
    env.num_envs,
    dtype=torch.long,
    device=env.device,
  )
  next_motion_index = 0

  def assign_available(target_env_ids: Any) -> None:
    nonlocal next_motion_index
    if target_env_ids.numel() == 0:
      return

    remaining = len(motion_files) - next_motion_index
    if remaining <= 0:
      active_mask[target_env_ids] = False
      assigned_motion_ids[target_env_ids] = -1
      assigned_motion_lengths[target_env_ids] = 0
      return

    assign_count = min(target_env_ids.numel(), remaining)
    assign_env_ids = target_env_ids[:assign_count]
    motion_indices = torch.arange(
      next_motion_index,
      next_motion_index + assign_count,
      device=env.device,
      dtype=torch.long,
    )
    next_motion_index += assign_count
    _assign_motion_indices(env, command, assign_env_ids, motion_indices)

    active_mask[assign_env_ids] = True
    assigned_motion_ids[assign_env_ids] = motion_indices
    assigned_motion_lengths[assign_env_ids] = command.motion.file_lengths[
      motion_indices
    ]

    if assign_count < target_env_ids.numel():
      idle_env_ids = target_env_ids[assign_count:]
      active_mask[idle_env_ids] = False
      assigned_motion_ids[idle_env_ids] = -1
      assigned_motion_lengths[idle_env_ids] = 0

  assign_available(env_ids)
  obs = _make_tensor_dict(env.obs_buf, batch_size=[env.num_envs])
  records: list[dict[str, Any]] = []
  progress = tqdm(total=len(motion_files), desc="Filtering motions", unit="motion")
  completed_motion_count = 0

  while completed_motion_count < len(motion_files):
    pre_episode_lengths = env.episode_length_buf.clone()
    pre_motion_ids = assigned_motion_ids.clone()
    pre_motion_lengths = assigned_motion_lengths.clone()
    pre_active_mask = active_mask.clone()

    with torch.no_grad():
      actions = policy(obs)

    obs, _, dones, _ = vec_env.step(actions)
    done_env_ids = torch.where(dones.bool() & pre_active_mask)[0]
    if done_env_ids.numel() == 0:
      continue

    terminated = env.reset_terminated[done_env_ids].detach().cpu()
    truncated = env.reset_time_outs[done_env_ids].detach().cpu()

    for idx, env_id in enumerate(done_env_ids.tolist()):
      motion_index = int(pre_motion_ids[env_id].item())
      total_steps = int(pre_motion_lengths[env_id].item())
      completed_steps = min(int(pre_episode_lengths[env_id].item()) + 1, total_steps)
      completion_ratio = completed_steps / float(max(total_steps, 1))
      records.append(
        {
          "motion_index": motion_index,
          "path": str(motion_files[motion_index].resolve()),
          "rank": rank,
          "completed_steps": completed_steps,
          "total_steps": total_steps,
          "completion_ratio": completion_ratio,
          "terminated": bool(terminated[idx].item()),
          "truncated": bool(truncated[idx].item()),
        }
      )

    completed_motion_count += int(done_env_ids.numel())
    progress.update(int(done_env_ids.numel()))
    assign_available(done_env_ids.to(device=env.device))
    obs = _make_tensor_dict(env.obs_buf, batch_size=[env.num_envs])

  progress.close()
  env.close()

  report = build_filter_report(
    task_id=task_id,
    motion_root=motion_root,
    checkpoint=checkpoint_label,
    threshold=cfg.failure_threshold,
    records=records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Failed: {report['failed_motion_count']} "
    f"({report['failed_motion_ratio']:.2%})."
  )
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def launch_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, Any]:
  from mjlab.utils.gpu import select_gpus

  cfg = cast(EvaluateConfig, _prepare_launch_cfg(cfg))
  if cfg.gpu_ids is None:
    return run_evaluate(task_id, cfg)

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    return run_evaluate(task_id, cfg)

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if cfg.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
    else:
      output_path = Path(cfg.output_file)
      os.environ["TORCHRUNX_LOG_DIR"] = str(
        output_path.parent / f"{output_path.stem}_torchrunx"
      )

  print(f"[INFO] Launching data filtering with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_evaluate, task_id, cfg)

  rank_report_paths = [
    _rank_output_path(cfg.output_file, rank=rank, world_size=num_gpus)
    for rank in range(num_gpus)
  ]
  merged_report = merge_filter_reports(rank_report_paths, Path(cfg.output_file))
  print(
    f"[INFO] Merged {len(rank_report_paths)} partial reports into "
    f"{Path(cfg.output_file).resolve()}"
  )
  return merged_report


__all__ = [
  "EvaluateConfig",
  "FilteringEvalEnv",
  "build_filter_report",
  "launch_evaluate",
  "merge_filter_reports",
  "motion_sequence_complete",
  "run_evaluate",
]
