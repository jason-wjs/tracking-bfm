"""CLI for exporting tracking-bfm checkpoints to ONNX files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--checkpoint", required=True, help="Path to the source .pt file."
  )
  parser.add_argument(
    "--task-id", required=True, help="Registered tracking_bfm task id."
  )
  parser.add_argument(
    "--checkpoint-family",
    choices=["auto", "tracking", "distillation"],
    default="auto",
    help="Checkpoint family. Defaults to auto-detection.",
  )
  parser.add_argument(
    "--obs-group",
    default=None,
    help="Observation group override. Defaults to the family-specific group.",
  )
  parser.add_argument(
    "--motion-path",
    default=None,
    help="Directory containing .npz motion files for multi-motion tracking tasks.",
  )
  parser.add_argument(
    "--motion-file",
    default=None,
    help="Single .npz motion file for single-motion tracking tasks.",
  )
  parser.add_argument(
    "--student-history-steps",
    type=int,
    default=None,
    help=(
      "Sparse reference history_steps override. For wbteleop tracking, this "
      "applies only to ref_limb_ee_pose_b params, not to command history_length."
    ),
  )
  parser.add_argument(
    "--student-future-steps",
    type=int,
    default=None,
    help=(
      "Sparse reference future_steps override. For wbteleop tracking, this "
      "applies only to ref_limb_ee_pose_b params."
    ),
  )
  parser.add_argument(
    "--student-robot-history-steps",
    type=int,
    default=None,
    help=(
      "Robot-state observation history_length override. For wbteleop tracking, "
      "this applies to robot_limb_ee_pose_b and proprioceptive terms."
    ),
  )
  _add_common_export_args(parser)


def _add_latent_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--checkpoint",
    required=True,
    help="Path to the latent tracking actor .pt checkpoint.",
  )
  parser.add_argument(
    "--decoder-checkpoint",
    required=True,
    help="Path to the latent distillation decoder .pt checkpoint.",
  )
  parser.add_argument(
    "--task-id", required=True, help="Registered latent tracking task id."
  )
  parser.add_argument(
    "--obs-group",
    default="actor",
    help="Actor observation group metadata. Defaults to actor.",
  )
  parser.add_argument(
    "--proprio-obs-group",
    default="proprio_actor",
    help="Decoder proprio observation group metadata. Defaults to proprio_actor.",
  )
  parser.add_argument(
    "--motion-path",
    default=None,
    help="Directory containing .npz motion files for multi-motion tracking tasks.",
  )
  parser.add_argument(
    "--motion-file",
    default=None,
    help="Single .npz motion file for single-motion tracking tasks.",
  )
  parser.add_argument(
    "--latent-action-clip",
    type=float,
    default=None,
    help="Latent action clamp override. Defaults to the task runner config.",
  )
  _add_common_export_args(parser)


def _add_common_export_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--output-name",
    default=None,
    help="Optional ONNX filename. Defaults to deploy_<checkpoint_stem>.onnx.",
  )
  parser.add_argument(
    "--robot-name", default=None, help="Optional robot name metadata."
  )
  parser.add_argument(
    "--device", default="cpu", help="Device used to rebuild the actor."
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Allow replacing an existing ONNX file.",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose ONNX export logging.",
  )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export tracking-bfm checkpoints to ONNX files."
  )
  subparsers = parser.add_subparsers(dest="mode", required=True)

  policy_parser = subparsers.add_parser(
    "policy", help="Export a policy checkpoint to ONNX."
  )
  _add_policy_args(policy_parser)

  latent_parser = subparsers.add_parser(
    "latent",
    help=(
      "Export a latent tracking actor checkpoint plus frozen latent decoder "
      "checkpoint to one ONNX file."
    ),
  )
  _add_latent_args(latent_parser)

  return parser.parse_args(argv)


def _export_policy(args: argparse.Namespace) -> str:
  from tracking_bfm.export import export_checkpoint_to_onnx

  return str(
    export_checkpoint_to_onnx(
      checkpoint_path=args.checkpoint,
      task_id=args.task_id,
      checkpoint_family=args.checkpoint_family,
      obs_group=args.obs_group,
      motion_path=args.motion_path,
      motion_file=args.motion_file,
      student_history_steps=args.student_history_steps,
      student_future_steps=args.student_future_steps,
      student_robot_history_steps=args.student_robot_history_steps,
      output_name=args.output_name,
      robot_name=args.robot_name,
      overwrite=args.overwrite,
      device=args.device,
      verbose=args.verbose,
    )
  )


def _export_latent(args: argparse.Namespace) -> str:
  from tracking_bfm.export import export_latent_tracking_checkpoint_to_onnx

  return str(
    export_latent_tracking_checkpoint_to_onnx(
      checkpoint_path=args.checkpoint,
      decoder_checkpoint_path=args.decoder_checkpoint,
      task_id=args.task_id,
      obs_group=args.obs_group,
      proprio_obs_group=args.proprio_obs_group,
      motion_path=args.motion_path,
      motion_file=args.motion_file,
      latent_action_clip=args.latent_action_clip,
      output_name=args.output_name,
      robot_name=args.robot_name,
      overwrite=args.overwrite,
      device=args.device,
      verbose=args.verbose,
    )
  )


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)

  if args.mode == "policy":
    onnx_path = _export_policy(args)
  else:
    onnx_path = _export_latent(args)

  print(onnx_path)


if __name__ == "__main__":
  main()
