"""CLI for exporting latent tracking checkpoints to ONNX files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tracking_bfm.export import export_latent_tracking_checkpoint_to_onnx


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Export a latent tracking actor checkpoint plus frozen latent decoder "
      "checkpoint to one ONNX file."
    )
  )
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
  parser.add_argument("--task-id", required=True, help="Registered latent tracking task id.")
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
  parser.add_argument(
    "--output-name",
    default=None,
    help="Optional ONNX filename. Defaults to deploy_<checkpoint_stem>.onnx.",
  )
  parser.add_argument("--robot-name", default=None, help="Optional robot name metadata.")
  parser.add_argument("--device", default="cpu", help="Device used to rebuild models.")
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
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  onnx_path = export_latent_tracking_checkpoint_to_onnx(
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
  print(onnx_path)


if __name__ == "__main__":
  main()
