"""CLI wrapper for generated motion dataset rollouts."""

from __future__ import annotations

import argparse
from typing import Literal

from tracking_bfm.data_process.motion_dataset_generation import (
  GenerateDatasetConfig,
  launch_generate_dataset,
)


def _parse_gpu_ids(raw_gpu_ids: list[str] | None) -> list[int] | Literal["all"] | None:
  if raw_gpu_ids is None:
    return None
  if len(raw_gpu_ids) == 1 and raw_gpu_ids[0] == "all":
    return "all"
  return [int(gpu_id) for gpu_id in raw_gpu_ids]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="tracking-bfm-generate-motion-dataset",
    description="Generate replacement motion clips from successful teacher rollouts.",
  )
  parser.add_argument("task_id", help="Registered tracking task ID to evaluate.")
  parser.add_argument("--wandb-run-path")
  parser.add_argument("--wandb-checkpoint-name")
  parser.add_argument("--checkpoint-file")
  parser.add_argument("--motion-path")
  parser.add_argument(
    "--motion-type", choices=["isaaclab", "mujoco"], default="isaaclab"
  )
  parser.add_argument("--history-steps", type=int)
  parser.add_argument("--future-steps", type=int)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--device")
  parser.add_argument("--completion-threshold", type=float, default=0.95)
  parser.add_argument("--output-motion-path", default="generated_motions")
  parser.add_argument("--output-file", default="generated_motions_report.json")
  parser.add_argument("--torchrunx-log-dir")
  parser.add_argument(
    "--gpu-ids",
    nargs="+",
    help="GPU IDs to use, or 'all'. Omit to run in the current process.",
  )
  return parser


def config_from_args(args: argparse.Namespace) -> GenerateDatasetConfig:
  return GenerateDatasetConfig(
    wandb_run_path=args.wandb_run_path,
    wandb_checkpoint_name=args.wandb_checkpoint_name,
    checkpoint_file=args.checkpoint_file,
    motion_path=args.motion_path,
    motion_type=args.motion_type,
    history_steps=args.history_steps,
    future_steps=args.future_steps,
    num_envs=args.num_envs,
    device=args.device,
    completion_threshold=args.completion_threshold,
    output_motion_path=args.output_motion_path,
    output_file=args.output_file,
    torchrunx_log_dir=args.torchrunx_log_dir,
    gpu_ids=_parse_gpu_ids(args.gpu_ids),
  )


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  import tracking_bfm  # noqa: F401

  launch_generate_dataset(args.task_id, config_from_args(args))


if __name__ == "__main__":
  main()
