# Commands Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor tracking command internals so real-motion runtime behavior is covered, reference motion dataset loading is a focused module, and command consumers use an explicit reference gather interface.

**Architecture:** Keep `tracking_bfm.motion_source` as the workflow-to-command-config seam. Add `tracking_bfm.tasks.tracking.mdp.motion_dataset` for `.npz` loading, motion type reindexing, tensor concatenation, and reference field gathering. Keep `commands.py` and `multi_motion_command.py` as command runtime modules.

**Tech Stack:** Python 3.10, pytest, torch, numpy, mjlab `ManagerBasedRlEnv`, ruff, pyright.

---

### Task 1: Runtime Smoke Coverage

**Files:**
- Create: `tests/tasks/tracking/test_command_runtime_smoke.py`

- [ ] **Step 1: Write the smoke tests**

Create `tests/tasks/tracking/test_command_runtime_smoke.py` with:

```python
from __future__ import annotations

from pathlib import Path
import pytest
import torch

pytest.importorskip("mjlab")

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import tracking_bfm  # noqa: F401
from tracking_bfm.tasks.tracking.mdp.multi_motion_command import MotionCommandCfg

PRIMARY_TRACKING_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1"
PRIMARY_1STAGE_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage"
PRIMARY_WBTELEOP_ID = "Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop"
MOTION_ROOT = Path("/data_zcy/zcy/motion_data")
PREFERRED_MOTION_DIRS = (
  MOTION_ROOT / "noitom",
  MOTION_ROOT / "AMASS_LAFAN_Qingtong" / "lafan_qingtong",
)


def _sample_motion_files(limit: int = 2) -> list[Path]:
  if not MOTION_ROOT.exists():
    pytest.skip(f"Motion smoke data not found: {MOTION_ROOT}")

  motion_files: list[Path] = []
  search_roots = [path for path in PREFERRED_MOTION_DIRS if path.exists()]
  if not search_roots:
    search_roots = [MOTION_ROOT]

  for search_root in search_roots:
    for path in sorted(search_root.glob("*.npz")):
      motion_files.append(path)
      if len(motion_files) >= limit:
        return motion_files

  pytest.skip(f"No bounded smoke .npz files found under {MOTION_ROOT}")


def _build_env(task_id: str, tmp_motion_root: Path) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.scene.num_envs = 1
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_path = str(tmp_motion_root)
  motion_cmd.motion_file = ""
  motion_cmd.motion_type = "mujoco"
  motion_cmd.sampling_mode = "uniform"
  motion_cmd.if_log_metrics = False
  motion_cmd.history_steps = 1
  motion_cmd.future_steps = 2
  return ManagerBasedRlEnv(cfg=env_cfg, device="cpu")


@pytest.fixture
def smoke_motion_root(tmp_path: Path) -> Path:
  motion_root = tmp_path / "motions"
  motion_root.mkdir()
  for motion_file in _sample_motion_files():
    (motion_root / motion_file.name).symlink_to(motion_file)
  return motion_root


def test_primary_tracking_runtime_smoke_with_real_motion(
  smoke_motion_root: Path,
) -> None:
  env = _build_env(PRIMARY_TRACKING_ID, smoke_motion_root)
  try:
    obs, _ = env.reset()
    assert obs.shape[0] == env.num_envs

    command = env.command_manager.get_term("motion")
    assert command.motion.num_files >= 1
    gathered = command._gather_motion_field(
      "joint_pos",
      command.motion_idx,
      command.time_steps.unsqueeze(1),
    )
    assert gathered.shape[0] == env.num_envs

    actions = torch.zeros(env.num_envs, env.num_actions)
    next_obs, _, _, _ = env.step(actions)
    assert next_obs.shape[0] == env.num_envs
  finally:
    env.close()


@pytest.mark.parametrize("task_id", [PRIMARY_1STAGE_ID, PRIMARY_WBTELEOP_ID])
def test_sparse_and_wbteleop_runtime_smoke_with_real_motion(
  task_id: str,
  smoke_motion_root: Path,
) -> None:
  env = _build_env(task_id, smoke_motion_root)
  try:
    obs, _ = env.reset()
    assert obs.shape[0] == env.num_envs
    assert env.command_manager.get_term("motion").motion.num_files >= 1
  finally:
    env.close()
```

- [ ] **Step 2: Run the smoke tests**

Run:

```bash
uv run pytest tests/tasks/tracking/test_command_runtime_smoke.py -q
```

Expected: pass when `/data_zcy/zcy/motion_data/` has compatible motion files, or skip when the directory is unavailable.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/tasks/tracking/test_command_runtime_smoke.py
git commit -m "test: add tracking command runtime smoke"
```

### Task 2: Reference Motion Dataset Module

**Files:**
- Create: `tests/tasks/tracking/test_motion_dataset.py`
- Create: `src/tracking_bfm/tasks/tracking/mdp/motion_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/tasks/tracking/test_motion_dataset.py` with:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")

from tracking_bfm.tasks.tracking.mdp.motion_dataset import ReferenceMotionDataset


def _write_motion(path: Path, *, frames: int, offset: float) -> None:
  joint_pos = (
    np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) + offset
  )
  joint_vel = joint_pos + 1000.0
  body_pos = (
    np.arange(frames * 30 * 3, dtype=np.float32).reshape(frames, 30, 3)
    + offset
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
```

- [ ] **Step 2: Run the dataset tests and verify RED**

Run:

```bash
uv run pytest tests/tasks/tracking/test_motion_dataset.py -q
```

Expected: fail because `tracking_bfm.tasks.tracking.mdp.motion_dataset` does not exist.

- [ ] **Step 3: Implement the dataset module**

Create `src/tracking_bfm/tasks/tracking/mdp/motion_dataset.py` with the joint/body reindex constants copied from the current command modules, and this public class:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

MotionType = Literal["isaaclab", "mujoco"]


class ReferenceMotionDataset:
  def __init__(
    self,
    motion_files: Sequence[str | Path],
    body_indexes: torch.Tensor,
    motion_type: MotionType = "isaaclab",
    device: str | torch.device = "cpu",
  ) -> None:
    if not motion_files:
      raise ValueError("motion_files cannot be empty")
    self.device = device
    self._body_indexes = body_indexes.to(device=device, dtype=torch.long)
    joint_reindex, body_reindex = motion_reindex(motion_type)

    self.num_files = len(motion_files)
    self.fps_list: list[float] = []
    file_lengths: list[int] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_lin_vel_w_list: list[torch.Tensor] = []
    body_ang_vel_w_list: list[torch.Tensor] = []

    for motion_file in motion_files:
      motion_path = Path(motion_file)
      if not motion_path.is_file():
        raise FileNotFoundError(f"Invalid motion file: {motion_path}")
      data = np.load(motion_path)
      self.fps_list.append(float(data["fps"]))
      joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
      body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
      )
      body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
      )
      if joint_reindex is not None:
        joint_pos = joint_pos[:, joint_reindex]
        joint_vel = joint_vel[:, joint_reindex]
      if body_reindex is not None:
        body_pos_w = body_pos_w[:, body_reindex, :]
        body_quat_w = body_quat_w[:, body_reindex, :]
        body_lin_vel_w = body_lin_vel_w[:, body_reindex, :]
        body_ang_vel_w = body_ang_vel_w[:, body_reindex, :]

      body_pos_w = body_pos_w[:, self._body_indexes, :]
      body_quat_w = body_quat_w[:, self._body_indexes, :]
      body_lin_vel_w = body_lin_vel_w[:, self._body_indexes, :]
      body_ang_vel_w = body_ang_vel_w[:, self._body_indexes, :]

      joint_pos_list.append(joint_pos)
      joint_vel_list.append(joint_vel)
      body_pos_w_list.append(body_pos_w)
      body_quat_w_list.append(body_quat_w)
      body_lin_vel_w_list.append(body_lin_vel_w)
      body_ang_vel_w_list.append(body_ang_vel_w)
      file_lengths.append(joint_pos.shape[0])

    self.file_lengths = torch.tensor(file_lengths, dtype=torch.long, device=device)
    self.fps = self.fps_list[0]
    self.length_starts = torch.cat(
      [
        torch.zeros(1, dtype=torch.long, device=device),
        self.file_lengths[:-1].cumsum(dim=0),
      ]
    )
    self.joint_pos = torch.cat(joint_pos_list, dim=0)
    self.joint_vel = torch.cat(joint_vel_list, dim=0)
    self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
    self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
    self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
    self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)
    self.time_step_total = int(self.joint_pos.shape[0])

  def clamp_time_steps(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    max_time_steps = self.file_lengths[motion_ids] - 1
    if time_steps.ndim > 1:
      max_time_steps = max_time_steps.unsqueeze(-1)
    clamped_time_steps = torch.clamp_min(time_steps, 0)
    return torch.minimum(clamped_time_steps, max_time_steps)

  def frame_indices(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    clamped_time_steps = self.clamp_time_steps(motion_ids, time_steps)
    frame_starts = self.length_starts[motion_ids]
    if clamped_time_steps.ndim > 1:
      frame_starts = frame_starts.unsqueeze(-1)
    return frame_starts + clamped_time_steps

  def gather(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return getattr(self, field_name)[self.frame_indices(motion_ids, time_steps)]

  def get_motion_data_batch(
    self, motion_idx: int, time_steps_start: torch.Tensor, time_steps_end: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    time_steps_tensor = torch.arange(
      time_steps_start.item(),
      time_steps_end.item(),
      device=self.file_lengths.device,
      dtype=torch.long,
    )
    time_steps_tensor = torch.clamp(
      time_steps_tensor,
      torch.tensor(0, device=self.file_lengths.device),
      self.file_lengths[motion_idx] - 1,
    )
    frame_indices = self.length_starts[motion_idx] + time_steps_tensor
    return {
      "joint_pos": self.joint_pos[frame_indices],
      "joint_vel": self.joint_vel[frame_indices],
      "body_pos_w": self.body_pos_w[frame_indices],
      "body_quat_w": self.body_quat_w[frame_indices],
      "body_lin_vel_w": self.body_lin_vel_w[frame_indices],
      "body_ang_vel_w": self.body_ang_vel_w[frame_indices],
    }
```

Implement `motion_reindex(motion_type)` in the same module with the current `_ISAACLAB_*`, `_MUJOCO_*`, `_ISAACLAB_TO_MUJOCO_*` constants.

- [ ] **Step 4: Run dataset tests and verify GREEN**

Run:

```bash
uv run pytest tests/tasks/tracking/test_motion_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/tasks/tracking/test_motion_dataset.py src/tracking_bfm/tasks/tracking/mdp/motion_dataset.py
git commit -m "feat: add reference motion dataset"
```

### Task 3: Multi-Motion Command Uses Dataset

**Files:**
- Modify: `src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py`
- Modify: `tests/tasks/tracking/test_motion_dataset.py`

- [ ] **Step 1: Add a parity test for multi-motion field gathering**

Append to `tests/tasks/tracking/test_motion_dataset.py`:

```python
def test_reference_motion_dataset_replaces_multi_command_private_gather(
  tmp_path: Path,
) -> None:
  first = tmp_path / "first.npz"
  second = tmp_path / "second.npz"
  _write_motion(first, frames=2, offset=0.0)
  _write_motion(second, frames=2, offset=100.0)
  dataset = ReferenceMotionDataset(
    [first, second],
    body_indexes=torch.tensor([0, 1], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )

  motion_ids = torch.tensor([0, 1], dtype=torch.long)
  time_steps = torch.tensor([[0, 1], [1, 99]], dtype=torch.long)

  gathered = dataset.gather("body_pos_w", motion_ids, time_steps)

  assert gathered.shape == (2, 2, 2, 3)
  torch.testing.assert_close(gathered[0, 0], dataset.body_pos_w[0])
  torch.testing.assert_close(gathered[0, 1], dataset.body_pos_w[1])
  torch.testing.assert_close(gathered[1, 0], dataset.body_pos_w[3])
  torch.testing.assert_close(gathered[1, 1], dataset.body_pos_w[3])
```

- [ ] **Step 2: Run the parity test**

Run:

```bash
uv run pytest tests/tasks/tracking/test_motion_dataset.py::test_reference_motion_dataset_replaces_multi_command_private_gather -q
```

Expected: pass. This is a characterization test of the dataset behavior needed by the command refactor.

- [ ] **Step 3: Refactor `multi_motion_command.py`**

Edit `src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py`:

- import `ReferenceMotionDataset` from `tracking_bfm.tasks.tracking.mdp.motion_dataset`
- delete local `MotionLoader` and `MultiMotionLoader`
- delete duplicated joint/body reindex constants
- construct `self.motion = ReferenceMotionDataset(...)`
- replace `_clamp_motion_time_steps`, `_get_frame_indices`, and `_gather_motion_field` with delegating methods:

```python
  def _clamp_motion_time_steps(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.clamp_time_steps(motion_ids, time_steps)

  def _get_frame_indices(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.frame_indices(motion_ids, time_steps)

  def _gather_motion_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.gather(field_name, motion_ids, time_steps)
```

- [ ] **Step 4: Run tracking and data processing tests**

Run:

```bash
uv run pytest tests/tasks/tracking/test_tracking_tasks.py tests/tasks/tracking/test_motion_dataset.py tests/data_process/test_data_process.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py tests/tasks/tracking/test_motion_dataset.py
git commit -m "refactor: reuse reference dataset in multi motion command"
```

### Task 4: Public Reference Gather Interface

**Files:**
- Modify: `src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py`
- Modify: `src/tracking_bfm/tasks/distillation/mdp/commands.py`
- Modify: `src/tracking_bfm/tasks/tracking/wbteleop/observations.py`
- Modify: `tests/tasks/distillation/test_distillation_tasks.py`
- Modify: `tests/tasks/tracking/test_tracking_tasks.py`

- [ ] **Step 1: Write failing tests for public gather consumers**

Add to `tests/tasks/distillation/test_distillation_tasks.py`:

```python
def test_student_multistep_observations_use_public_reference_gather() -> None:
  from types import SimpleNamespace

  import torch

  from tracking_bfm.tasks.distillation.mdp import commands

  class FakeCommand:
    cfg = SimpleNamespace(body_names=("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"))
    time_steps = torch.tensor([5], dtype=torch.long)
    motion_idx = torch.tensor([0], dtype=torch.long)
    _env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))

    def gather_reference_field(self, field_name, motion_ids, time_steps):
      assert field_name in {"body_pos_w", "body_quat_w"}
      if field_name == "body_pos_w":
        return torch.zeros(1, 3, 3, 3)
      quat = torch.zeros(1, 3, 3, 4)
      quat[..., 0] = 1.0
      return quat

  env = SimpleNamespace(
    num_envs=1,
    command_manager=SimpleNamespace(get_term=lambda _: FakeCommand()),
  )

  obs = commands.student_ee_pose_b(
    env,
    "motion",
    ee_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
    history_steps=1,
    future_steps=2,
  )

  assert obs.shape == (1, 54)
```

Add to `tests/tasks/tracking/test_tracking_tasks.py`:

```python
def test_wbteleop_reference_observations_use_public_reference_gather() -> None:
  from types import SimpleNamespace

  import torch

  from tracking_bfm.tasks.tracking.wbteleop import observations

  class FakeCommand:
    cfg = SimpleNamespace(
      body_names=(
        "pelvis",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      )
    )
    time_steps = torch.tensor([5], dtype=torch.long)
    motion_idx = torch.tensor([0], dtype=torch.long)
    _env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))

    def gather_reference_field(self, field_name, motion_ids, time_steps):
      assert field_name in {"body_pos_w", "body_quat_w"}
      if field_name == "body_pos_w":
        return torch.zeros(1, 3, 5, 3)
      quat = torch.zeros(1, 3, 5, 4)
      quat[..., 0] = 1.0
      return quat

  env = SimpleNamespace(
    num_envs=1,
    command_manager=SimpleNamespace(get_term=lambda _: FakeCommand()),
  )

  obs = observations.ref_limb_ee_pose_b(
    env,
    "motion",
    history_steps=1,
    future_steps=2,
  )

  assert obs.shape == (1, 108)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/tasks/distillation/test_distillation_tasks.py::test_student_multistep_observations_use_public_reference_gather tests/tasks/tracking/test_tracking_tasks.py::test_wbteleop_reference_observations_use_public_reference_gather -q
```

Expected: fail because consumers still require `_gather_motion_field`.

- [ ] **Step 3: Add public method and migrate consumers**

In `src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py`, add:

```python
  def gather_reference_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.motion.gather(field_name, motion_ids, time_steps)
```

Keep `_gather_motion_field` as a compatibility wrapper:

```python
  def _gather_motion_field(
    self, field_name: str, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self.gather_reference_field(field_name, motion_ids, time_steps)
```

In `src/tracking_bfm/tasks/distillation/mdp/commands.py`, replace the private check and call with `gather_reference_field`:

```python
  gather_reference_field = getattr(command, "gather_reference_field", None)
  if not callable(gather_reference_field):
    raise NotImplementedError(
      "Student multi-step command observations require a motion command that "
      "supports reference field gathering."
    )
...
  values = gather_reference_field(field_name, command.motion_idx, reference_time_steps)
```

In `src/tracking_bfm/tasks/tracking/wbteleop/observations.py`, make the same change in `_reference_time_window`.

- [ ] **Step 4: Run selected tests**

Run:

```bash
uv run pytest tests/tasks/distillation/test_distillation_tasks.py tests/tasks/tracking/test_tracking_tasks.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py src/tracking_bfm/tasks/distillation/mdp/commands.py src/tracking_bfm/tasks/tracking/wbteleop/observations.py tests/tasks/distillation/test_distillation_tasks.py tests/tasks/tracking/test_tracking_tasks.py
git commit -m "refactor: expose reference gather interface"
```

### Task 5: Single-Motion Command Uses Dataset

**Files:**
- Modify: `src/tracking_bfm/tasks/tracking/mdp/commands.py`
- Modify: `tests/tasks/tracking/test_motion_dataset.py`

- [ ] **Step 1: Add single-command dataset compatibility test**

Append to `tests/tasks/tracking/test_motion_dataset.py`:

```python
def test_reference_motion_dataset_supports_single_command_time_indexing(
  tmp_path: Path,
) -> None:
  motion_file = tmp_path / "single.npz"
  _write_motion(motion_file, frames=3, offset=5.0)
  dataset = ReferenceMotionDataset(
    [motion_file],
    body_indexes=torch.tensor([0, 2], dtype=torch.long),
    motion_type="mujoco",
    device="cpu",
  )
  time_steps = torch.tensor([0, 2], dtype=torch.long)

  assert dataset.time_step_total == 3
  assert dataset.joint_pos[time_steps].shape == (2, 29)
  assert dataset.body_pos_w[time_steps].shape == (2, 2, 3)
```

- [ ] **Step 2: Run the compatibility test**

Run:

```bash
uv run pytest tests/tasks/tracking/test_motion_dataset.py::test_reference_motion_dataset_supports_single_command_time_indexing -q
```

Expected: pass. This verifies the dataset supports the indexing style used by the single-motion command.

- [ ] **Step 3: Refactor `commands.py` to use `ReferenceMotionDataset`**

Edit `src/tracking_bfm/tasks/tracking/mdp/commands.py`:

- import `ReferenceMotionDataset`
- delete local `MotionLoader` and duplicated reindex constants
- construct the motion dataset with a single file:

```python
    self.motion = ReferenceMotionDataset(
      [self.cfg.motion_file],
      self.body_indexes,
      motion_type=self.cfg.motion_type,
      device=self.device,
    )
```

Do not change `MotionCommandCfg`, command properties, reset behavior, sampling behavior, GUI behavior, or debug visualization.

- [ ] **Step 4: Run tracking tests**

Run:

```bash
uv run pytest tests/tasks/tracking/test_tracking_tasks.py tests/tasks/tracking/test_motion_dataset.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/tracking_bfm/tasks/tracking/mdp/commands.py tests/tasks/tracking/test_motion_dataset.py
git commit -m "refactor: reuse reference dataset in single motion command"
```

### Task 6: Documentation and Adaptive Sampling Review

**Files:**
- Modify: `docs/architecture/tracking-cleanup.md`
- Modify: `docs/superpowers/specs/2026-06-25-commands-refactor-design.md`

- [ ] **Step 1: Update architecture docs**

Update the `tracking/mdp/commands.py` and `tracking/mdp/multi_motion_command.py` rows in `docs/architecture/tracking-cleanup.md`:

```markdown
| `tracking/mdp/commands.py` | bfm-owned | Single-motion BFM command runtime; reference `.npz` loading and reindexing live in `tracking/mdp/motion_dataset.py`. | Keep as BFM delta over upstream command behavior; revisit upstream composition only after command runtime stays stable. |
| `tracking/mdp/multi_motion_command.py` | bfm-owned, needs-seam | Canonical BFM multi-motion command runtime; uses the shared reference motion dataset module but still owns adaptive sampling. | Keep; review adaptive sampling as a later internal module extraction. |
| `tracking/mdp/motion_dataset.py` | bfm-owned | Loads reference `.npz` motion tensors, applies motion-type reindexing, stores single/multi motion tensors, and gathers fields by motion id and time step. | Keep as the internal dataset seam shared by single and multi commands. |
```

Append a status note to the design spec:

```markdown
## Implementation Status

Runtime smoke coverage, reference motion dataset extraction, public reference
gathering, and single-command dataset reuse are implemented. Adaptive sampling
remains inside `multi_motion_command.py` and should be revisited only after the
dataset seam has remained stable through normal training/play usage.
```

- [ ] **Step 2: Run documentation diff review**

Run:

```bash
git diff -- docs/architecture/tracking-cleanup.md docs/superpowers/specs/2026-06-25-commands-refactor-design.md
```

Expected: docs only describe the implemented state and do not claim adaptive sampling was extracted.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/architecture/tracking-cleanup.md docs/superpowers/specs/2026-06-25-commands-refactor-design.md
git commit -m "docs: record commands dataset seam"
```

### Task 7: Final Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run full tests**

Run:

```bash
uv run pytest
```

Expected: all tests pass; runtime smoke tests pass or skip depending on local motion data availability.

- [ ] **Step 2: Run lint and formatting checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: both commands pass.

- [ ] **Step 3: Run type checking**

Run:

```bash
uv run pyright
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 4: Inspect git state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
```

Expected: working tree clean and the commands refactor commits visible on `main`.
