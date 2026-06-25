"""ONNX export for tracking-bfm actor checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from tracking_bfm.export.checkpoint import (
  CheckpointFamily,
  CheckpointFamilyOption,
  detect_checkpoint_family,
  ensure_output_path_available,
  load_checkpoint,
  resolve_onnx_output_path,
)
from tracking_bfm.export.metadata import build_policy_metadata
from tracking_bfm.motion_source import (
  MotionSourceSpec,
  apply_motion_source_to_command,
  resolve_motion_source,
)


def resolve_policy_onnx_path(
  checkpoint_path: str | Path,
  output_name: str | Path | None = None,
) -> Path:
  """Resolve the policy ONNX path beside the source checkpoint."""
  return resolve_onnx_output_path(checkpoint_path, output_name=output_name)


def export_actor_model_to_onnx(
  *,
  actor,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamily,
  obs_group: str,
  output_name: str | Path | None = None,
  robot_name: str | None = None,
  overwrite: bool = False,
  verbose: bool = False,
) -> Path:
  """Export a rebuilt actor module to ONNX."""
  onnx_path = ensure_output_path_available(
    resolve_policy_onnx_path(checkpoint_path, output_name=output_name),
    overwrite=overwrite,
  )

  import torch
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx

  onnx_model = actor.as_onnx(verbose=verbose)
  onnx_model.to("cpu")
  onnx_model.eval()

  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),  # type: ignore[operator]
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=onnx_model.input_names,  # type: ignore[arg-type]
    output_names=onnx_model.output_names,  # type: ignore[arg-type]
    dynamic_axes={},
    dynamo=False,
  )

  attach_metadata_to_onnx(
    str(onnx_path),
    build_policy_metadata(
      task_id=task_id,
      obs_group=obs_group,
      checkpoint_family=checkpoint_family,
      robot_name=robot_name,
    ),
  )
  return onnx_path


def _apply_motion_source(
  env_cfg: Any,
  *,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
) -> None:
  """Apply an optional local motion source to tracking task env config."""
  source = resolve_motion_source(
    MotionSourceSpec(motion_file=motion_file, motion_path=motion_path)
  )
  motion_cfg = getattr(env_cfg, "commands", {}).get("motion")
  if motion_cfg is None or source is None:
    return

  apply_motion_source_to_command(motion_cfg, source)


def _apply_distillation_student_obs_overrides(
  env_cfg: Any,
  *,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
) -> None:
  """Apply student observation shape overrides used by distillation checkpoints."""
  if (
    student_history_steps is None
    and student_future_steps is None
    and student_robot_history_steps is None
  ):
    return

  motion_cfg = getattr(env_cfg, "commands", {}).get("motion")
  if motion_cfg is not None:
    if student_history_steps is not None and hasattr(motion_cfg, "history_steps"):
      motion_cfg.history_steps = int(student_history_steps)
    if student_future_steps is not None and hasattr(motion_cfg, "future_steps"):
      motion_cfg.future_steps = int(student_future_steps)

  observations = getattr(env_cfg, "observations", {})
  student_actor = observations.get("student_actor")
  if student_actor is None:
    return

  terms = getattr(student_actor, "terms", {})
  command_terms = ("ee_pose", "base_lin_vel_b", "base_ang_vel_b", "anchor_height_w")
  for term_name in command_terms:
    term = terms.get(term_name)
    if term is None:
      continue
    if student_history_steps is not None:
      term.params["history_steps"] = int(student_history_steps)
    if student_future_steps is not None:
      term.params["future_steps"] = int(student_future_steps)

  if student_robot_history_steps is None:
    return

  robot_state_terms = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  for term_name in robot_state_terms:
    term = terms.get(term_name)
    if term is not None:
      term.history_length = int(student_robot_history_steps)


def _apply_tracking_actor_obs_overrides(
  env_cfg: Any,
  *,
  obs_group: str = "actor",
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
) -> None:
  """Apply tracking actor observation overrides used by wbteleop checkpoints."""
  if (
    student_history_steps is None
    and student_future_steps is None
    and student_robot_history_steps is None
  ):
    return

  observations = getattr(env_cfg, "observations", {})
  actor = observations.get(obs_group)
  if actor is None:
    return

  terms = getattr(actor, "terms", {})
  ref_limb_ee_pose = terms.get("ref_limb_ee_pose_b")
  if ref_limb_ee_pose is not None:
    if student_history_steps is not None:
      ref_limb_ee_pose.params["history_steps"] = int(student_history_steps)
    if student_future_steps is not None:
      ref_limb_ee_pose.params["future_steps"] = int(student_future_steps)

  if student_robot_history_steps is None:
    return

  robot_state_terms = (
    "robot_limb_ee_pose_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  for term_name in robot_state_terms:
    term = terms.get(term_name)
    if term is not None:
      term.history_length = int(student_robot_history_steps)


def _build_actor_from_checkpoint(
  *,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamily,
  obs_group: str | None = None,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
  device: str = "cpu",
):
  """Rebuild and load the correct actor module for ONNX export."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import tracking_bfm.tasks  # noqa: F401

  env = None
  try:
    env_cfg = load_env_cfg(task_id, play=True)
    _apply_motion_source(
      env_cfg,
      motion_path=motion_path,
      motion_file=motion_file,
    )
    if checkpoint_family == "distillation":
      _apply_distillation_student_obs_overrides(
        env_cfg,
        student_history_steps=student_history_steps,
        student_future_steps=student_future_steps,
        student_robot_history_steps=student_robot_history_steps,
      )
    else:
      _apply_tracking_actor_obs_overrides(
        env_cfg,
        obs_group=obs_group or "actor",
        student_history_steps=student_history_steps,
        student_future_steps=student_future_steps,
        student_robot_history_steps=student_robot_history_steps,
      )
    runner_cfg = asdict(load_rl_cfg(task_id))
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env)
    runner = runner_cls(wrapped_env, runner_cfg, log_dir=None, device=device)
    runner.load(str(checkpoint_path), map_location=device)

    if checkpoint_family == "distillation":
      actor = runner.student_policy
      resolved_obs_group = obs_group or load_rl_cfg(task_id).student_obs_group
    else:
      actor = runner.alg.get_policy()
      resolved_obs_group = obs_group or "actor"

    actor.to("cpu")
    actor.eval()
    return actor, resolved_obs_group
  finally:
    if env is not None:
      env.close()


def export_checkpoint_to_onnx(
  *,
  checkpoint_path: str | Path,
  task_id: str,
  checkpoint_family: CheckpointFamilyOption = "auto",
  obs_group: str | None = None,
  motion_path: str | Path | None = None,
  motion_file: str | Path | None = None,
  student_history_steps: int | None = None,
  student_future_steps: int | None = None,
  student_robot_history_steps: int | None = None,
  output_name: str | Path | None = None,
  robot_name: str | None = None,
  overwrite: bool = False,
  device: str = "cpu",
  verbose: bool = False,
) -> Path:
  """Export a tracking-bfm checkpoint to ONNX."""
  checkpoint_path = Path(checkpoint_path)
  ensure_output_path_available(
    resolve_policy_onnx_path(checkpoint_path, output_name=output_name),
    overwrite=overwrite,
  )
  checkpoint = load_checkpoint(checkpoint_path)
  resolved_family = (
    detect_checkpoint_family(checkpoint)
    if checkpoint_family == "auto"
    else checkpoint_family
  )
  actor, resolved_obs_group = _build_actor_from_checkpoint(
    checkpoint_path=checkpoint_path,
    task_id=task_id,
    checkpoint_family=resolved_family,
    obs_group=obs_group,
    motion_path=motion_path,
    motion_file=motion_file,
    student_history_steps=student_history_steps,
    student_future_steps=student_future_steps,
    student_robot_history_steps=student_robot_history_steps,
    device=device,
  )
  return export_actor_model_to_onnx(
    actor=actor,
    checkpoint_path=checkpoint_path,
    task_id=task_id,
    checkpoint_family=resolved_family,
    obs_group=resolved_obs_group,
    output_name=output_name,
    robot_name=robot_name,
    overwrite=overwrite,
    verbose=verbose,
  )


__all__ = [
  "export_actor_model_to_onnx",
  "export_checkpoint_to_onnx",
  "resolve_policy_onnx_path",
]
