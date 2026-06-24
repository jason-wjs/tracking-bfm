"""Generate motion datasets from successful teacher rollouts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from tracking_bfm.data_process.motion_filtering import (
  GpuIds,
  MotionType,
  _assign_motion_indices,
  _collect_motion_files,
  _configure_motion_command,
  _invert_reindex,
  _load_multi_motion_command_symbols,
  _load_policy,
  _make_tensor_dict,
  _prepare_filtering_env_cfg,
  _rank_output_path,
  _resolve_checkpoint_path,
  _resolve_motion_root,
  _runtime_rank_context,
  _shard_motion_files,
)


@dataclass(frozen=True)
class GenerateDatasetConfig:
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  motion_path: str | None = None
  motion_type: MotionType = "isaaclab"
  history_steps: int | None = None
  future_steps: int | None = None
  num_envs: int = 1024
  device: str | None = None
  completion_threshold: float = 0.95
  output_motion_path: str = "generated_motions"
  output_file: str = "generated_motions_report.json"
  torchrunx_log_dir: str | None = None
  gpu_ids: GpuIds = None


_MOTION_NPZ_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _mujoco_to_isaaclab_reindexes() -> tuple[list[int], list[int]]:
  isaaclab_to_mujoco_joint, isaaclab_to_mujoco_body, _ = (
    _load_multi_motion_command_symbols()
  )
  return (
    _invert_reindex(isaaclab_to_mujoco_joint),
    _invert_reindex(isaaclab_to_mujoco_body),
  )


def output_motion_path_for(
  source_motion_path: Path,
  motion_root: Path | str,
  output_motion_root: Path | str,
) -> Path:
  """Map a source motion path into the generated dataset preserving layout."""

  source_path = Path(source_motion_path).resolve()
  root_path = Path(motion_root).resolve()
  output_root_path = Path(output_motion_root)
  relative_path = source_path.relative_to(root_path)
  return output_root_path / relative_path


def save_rollout_motion(output_path: Path, rollout: dict[str, Any]) -> None:
  """Save one teacher rollout clip in the standard motion ``.npz`` format."""

  import numpy as np

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **rollout)


def build_generate_dataset_report(
  *,
  task_id: str,
  motion_root: str,
  output_motion_root: str,
  checkpoint: str,
  threshold: float,
  saved_records: list[dict[str, Any]],
  failed_records: list[dict[str, Any]],
  rank: int,
  world_size: int,
) -> dict[str, Any]:
  sorted_saved = sorted(saved_records, key=lambda item: item["motion_index"])
  sorted_failed = sorted(failed_records, key=lambda item: item["motion_index"])
  total_motion_count = len(sorted_saved) + len(sorted_failed)
  saved_motion_count = len(sorted_saved)
  failed_motion_count = len(sorted_failed)
  saved_motion_ratio = (
    saved_motion_count / total_motion_count if total_motion_count > 0 else 0.0
  )

  return {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": task_id,
    "motion_root": motion_root,
    "output_motion_root": output_motion_root,
    "checkpoint": checkpoint,
    "completion_threshold": threshold,
    "rank": rank,
    "world_size": world_size,
    "total_motion_count": total_motion_count,
    "saved_motion_count": saved_motion_count,
    "failed_motion_count": failed_motion_count,
    "saved_motion_ratio": saved_motion_ratio,
    "saved_motions": sorted_saved,
    "failed_motions": sorted_failed,
  }


def merge_generate_dataset_reports(
  report_paths: list[Path],
  output_path: Path,
) -> dict[str, Any]:
  reports = []
  for report_path in sorted(report_paths):
    with report_path.open("r", encoding="utf-8") as file:
      reports.append(json.load(file))
  if not reports:
    raise ValueError("No partial reports found to merge.")

  merged_saved_motions: list[dict[str, Any]] = []
  merged_failed_motions: list[dict[str, Any]] = []
  total_motion_count = 0
  saved_motion_count = 0
  failed_motion_count = 0

  for report in reports:
    total_motion_count += int(report["total_motion_count"])
    saved_motion_count += int(report["saved_motion_count"])
    failed_motion_count += int(report["failed_motion_count"])
    merged_saved_motions.extend(report.get("saved_motions", []))
    merged_failed_motions.extend(report.get("failed_motions", []))

  merged_saved_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_failed_motions.sort(
    key=lambda item: (item.get("rank", -1), item.get("motion_index", -1))
  )
  merged_report = {
    "created_at": datetime.now(tz=timezone.utc).isoformat(),
    "task_id": reports[0]["task_id"],
    "motion_root": reports[0]["motion_root"],
    "output_motion_root": reports[0]["output_motion_root"],
    "checkpoint": reports[0]["checkpoint"],
    "completion_threshold": reports[0]["completion_threshold"],
    "world_size": max(int(report.get("world_size", 1)) for report in reports),
    "report_parts": len(reports),
    "total_motion_count": total_motion_count,
    "saved_motion_count": saved_motion_count,
    "failed_motion_count": failed_motion_count,
    "saved_motion_ratio": (
      saved_motion_count / total_motion_count if total_motion_count > 0 else 0.0
    ),
    "saved_motions": merged_saved_motions,
    "failed_motions": merged_failed_motions,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(merged_report, file, indent=2)
  return merged_report


def _empty_rollout_buffer() -> dict[str, list[Any]]:
  return {field: [] for field in _MOTION_NPZ_FIELDS}


def _capture_rollout_batch(command: Any, env_ids: Any) -> dict[str, Any]:
  """Capture current robot state in the IsaacLab motion npz layout."""

  robot_data = command.robot.data
  joint_reindex, body_reindex = _mujoco_to_isaaclab_reindexes()
  joint_pos = robot_data.joint_pos[env_ids][:, joint_reindex]
  joint_vel = robot_data.joint_vel[env_ids][:, joint_reindex]
  body_pos_w = robot_data.body_link_pos_w[env_ids]
  body_pos_w = body_pos_w - command._env.scene.env_origins[env_ids, None, :]
  body_pos_w = body_pos_w[:, body_reindex, :]
  body_quat_w = robot_data.body_link_quat_w[env_ids][:, body_reindex, :]
  body_lin_vel_w = robot_data.body_link_lin_vel_w[env_ids][:, body_reindex, :]
  body_ang_vel_w = robot_data.body_link_ang_vel_w[env_ids][:, body_reindex, :]
  return {
    "joint_pos": joint_pos.detach().cpu().numpy().copy(),
    "joint_vel": joint_vel.detach().cpu().numpy().copy(),
    "body_pos_w": body_pos_w.detach().cpu().numpy().copy(),
    "body_quat_w": body_quat_w.detach().cpu().numpy().copy(),
    "body_lin_vel_w": body_lin_vel_w.detach().cpu().numpy().copy(),
    "body_ang_vel_w": body_ang_vel_w.detach().cpu().numpy().copy(),
  }


def _append_rollout_batch(
  rollout_buffers: dict[int, dict[str, list[Any]]],
  command: Any,
  env_ids: Any,
) -> None:
  if env_ids.numel() == 0:
    return

  batch = _capture_rollout_batch(command, env_ids)
  for batch_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
    buffer = rollout_buffers.setdefault(int(env_id), _empty_rollout_buffer())
    for field in _MOTION_NPZ_FIELDS:
      buffer[field].append(batch[field][batch_index])


def _stack_rollout_buffer(
  buffer: dict[str, list[Any]],
  *,
  fps: Any,
  frame_count: int,
) -> dict[str, Any]:
  import numpy as np

  rollout = {"fps": np.asarray(fps)}
  for field in _MOTION_NPZ_FIELDS:
    if len(buffer[field]) < frame_count:
      raise ValueError(
        f"Rollout field '{field}' has {len(buffer[field])} frames, "
        f"expected at least {frame_count}."
      )
    frames = buffer[field][:frame_count]
    if not frames:
      raise ValueError(f"Cannot save rollout with no frames for field '{field}'.")
    rollout[field] = np.stack(frames, axis=0)
  return rollout


def run_generate_dataset(
  task_id: str,
  cfg: GenerateDatasetConfig,
) -> dict[str, Any]:
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
  rollout_buffers: dict[int, dict[str, list[Any]]] = {}
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
      for env_id in target_env_ids.detach().cpu().tolist():
        rollout_buffers.pop(int(env_id), None)
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
    for env_id in assign_env_ids.detach().cpu().tolist():
      rollout_buffers[int(env_id)] = _empty_rollout_buffer()

    if assign_count < target_env_ids.numel():
      idle_env_ids = target_env_ids[assign_count:]
      active_mask[idle_env_ids] = False
      assigned_motion_ids[idle_env_ids] = -1
      assigned_motion_lengths[idle_env_ids] = 0
      for env_id in idle_env_ids.detach().cpu().tolist():
        rollout_buffers.pop(int(env_id), None)

  assign_available(env_ids)
  obs = _make_tensor_dict(env.obs_buf, batch_size=[env.num_envs])
  saved_records: list[dict[str, Any]] = []
  failed_records: list[dict[str, Any]] = []
  progress = tqdm(total=len(motion_files), desc="Generating motions", unit="motion")
  completed_motion_count = 0
  output_motion_root = Path(cfg.output_motion_path)

  while completed_motion_count < len(motion_files):
    active_env_ids = torch.where(active_mask)[0]
    _append_rollout_batch(rollout_buffers, command, active_env_ids)

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
      source_path = motion_files[motion_index]
      base_record = {
        "motion_index": motion_index,
        "path": str(source_path.resolve()),
        "rank": rank,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "completion_ratio": completion_ratio,
        "terminated": bool(terminated[idx].item()),
        "truncated": bool(truncated[idx].item()),
      }

      if completion_ratio >= cfg.completion_threshold:
        output_path = output_motion_path_for(
          source_path,
          motion_root,
          output_motion_root,
        )
        rollout = _stack_rollout_buffer(
          rollout_buffers[int(env_id)],
          fps=command.motion.fps_list[motion_index],
          frame_count=completed_steps,
        )
        save_rollout_motion(output_path, rollout)
        saved_records.append({**base_record, "output_path": str(output_path.resolve())})
      else:
        failed_records.append(base_record)

      rollout_buffers.pop(int(env_id), None)

    completed_motion_count += int(done_env_ids.numel())
    progress.update(int(done_env_ids.numel()))
    assign_available(done_env_ids.to(device=env.device))
    obs = _make_tensor_dict(env.obs_buf, batch_size=[env.num_envs])

  progress.close()
  env.close()

  report = build_generate_dataset_report(
    task_id=task_id,
    motion_root=motion_root,
    output_motion_root=str(output_motion_root),
    checkpoint=checkpoint_label,
    threshold=cfg.completion_threshold,
    saved_records=saved_records,
    failed_records=failed_records,
    rank=rank,
    world_size=world_size,
  )
  output_path = _rank_output_path(cfg.output_file, rank, world_size)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

  print(
    f"[INFO] Evaluated {report['total_motion_count']} motions. "
    f"Saved: {report['saved_motion_count']} "
    f"({report['saved_motion_ratio']:.2%})."
  )
  print(f"[INFO] Generated dataset root: {output_motion_root.resolve()}")
  print(f"[INFO] Report saved to {output_path.resolve()}")
  return report


def launch_generate_dataset(
  task_id: str,
  cfg: GenerateDatasetConfig,
) -> dict[str, Any]:
  from mjlab.utils.gpu import select_gpus

  if cfg.gpu_ids is None:
    return run_generate_dataset(task_id, cfg)

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    return run_generate_dataset(task_id, cfg)

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

  print(f"[INFO] Launching dataset generation with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_generate_dataset, task_id, cfg)

  rank_report_paths = [
    _rank_output_path(cfg.output_file, rank=rank, world_size=num_gpus)
    for rank in range(num_gpus)
  ]
  merged_report = merge_generate_dataset_reports(
    rank_report_paths,
    Path(cfg.output_file),
  )
  print(
    f"[INFO] Merged {len(rank_report_paths)} partial reports into "
    f"{Path(cfg.output_file).resolve()}"
  )
  return merged_report


__all__ = [
  "GenerateDatasetConfig",
  "build_generate_dataset_report",
  "launch_generate_dataset",
  "merge_generate_dataset_reports",
  "output_motion_path_for",
  "run_generate_dataset",
  "save_rollout_motion",
]
