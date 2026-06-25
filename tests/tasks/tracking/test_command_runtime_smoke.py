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


def _assert_observation_batch(
  obs: dict[str, torch.Tensor], env: ManagerBasedRlEnv
) -> None:
  assert "actor" in obs
  assert obs["actor"].shape[0] == env.num_envs


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
    _assert_observation_batch(obs, env)

    command = env.command_manager.get_term("motion")
    assert command.motion.num_files >= 1
    gathered = command._gather_motion_field(
      "joint_pos",
      command.motion_idx,
      command.time_steps.unsqueeze(1),
    )
    assert gathered.shape[0] == env.num_envs

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    next_obs = env.step(actions)[0]
    _assert_observation_batch(next_obs, env)
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
    _assert_observation_batch(obs, env)
    assert env.command_manager.get_term("motion").motion.num_files >= 1
  finally:
    env.close()
