from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")

from tracking_bfm.tasks.tracking.mdp.motion_dataset import ReferenceMotionDataset


def _write_motion(path: Path, *, frames: int, offset: float) -> None:
  joint_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) + offset
  joint_vel = joint_pos + 1000.0
  body_pos = (
    np.arange(frames * 30 * 3, dtype=np.float32).reshape(frames, 30, 3) + offset
  )
  body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  body_lin_vel = body_pos + 2000.0
  body_ang_vel = body_pos + 3000.0
  np.savez(
    path,
    fps=np.array(50.0, dtype=np.float32),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
  )


def test_reference_motion_dataset_loads_single_mujoco_motion(tmp_path: Path) -> None:
  motion_file = tmp_path / "one.npz"
  _write_motion(motion_file, frames=3, offset=10.0)

  dataset = ReferenceMotionDataset(
    [motion_file],
    body_indexes=torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  assert dataset.num_files == 1
  assert dataset.time_step_total == 3
  assert dataset.file_lengths.tolist() == [3]
  assert dataset.length_starts.tolist() == [0]
  assert dataset.joint_pos.shape == (3, 29)
  assert dataset.body_pos_w.shape == (3, 2, 3)
  assert dataset.fps_list[0] == pytest.approx(50.0)


def test_reference_motion_dataset_concatenates_and_gathers_by_motion_id(
  tmp_path: Path,
) -> None:
  first = tmp_path / "first.npz"
  second = tmp_path / "second.npz"
  _write_motion(first, frames=2, offset=10.0)
  _write_motion(second, frames=4, offset=100.0)

  dataset = ReferenceMotionDataset(
    [first, second],
    body_indexes=torch.tensor([0, 1], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  motion_ids = torch.tensor([0, 1], dtype=torch.long)
  time_steps = torch.tensor([1, 3], dtype=torch.long)
  gathered = dataset.gather("joint_pos", motion_ids, time_steps)

  assert dataset.file_lengths.tolist() == [2, 4]
  assert dataset.length_starts.tolist() == [0, 2]
  torch.testing.assert_close(gathered[0], dataset.joint_pos[1])
  torch.testing.assert_close(gathered[1], dataset.joint_pos[5])


def test_reference_motion_dataset_clamps_windowed_time_steps(tmp_path: Path) -> None:
  motion_file = tmp_path / "one.npz"
  _write_motion(motion_file, frames=3, offset=10.0)
  dataset = ReferenceMotionDataset(
    [motion_file],
    body_indexes=torch.tensor([0], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  motion_ids = torch.tensor([0], dtype=torch.long)
  time_steps = torch.tensor([[-1, 1, 99]], dtype=torch.long)

  gathered = dataset.gather("joint_pos", motion_ids, time_steps)

  assert gathered.shape == (1, 3, 29)
  torch.testing.assert_close(gathered[0, 0], dataset.joint_pos[0])
  torch.testing.assert_close(gathered[0, 1], dataset.joint_pos[1])
  torch.testing.assert_close(gathered[0, 2], dataset.joint_pos[2])
