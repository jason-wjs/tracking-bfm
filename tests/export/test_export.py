from __future__ import annotations

from pathlib import Path

import pytest

from tracking_bfm.export.checkpoint import (
  detect_checkpoint_family,
  ensure_output_path_available,
  resolve_onnx_output_path,
)
from tracking_bfm.export.metadata import (
  build_latent_policy_metadata,
  build_policy_metadata,
)
from tracking_bfm.export.onnx_latent_policy import (
  export_actor_decoder_model_to_onnx,
  resolve_latent_policy_onnx_path,
)
from tracking_bfm.export.onnx_policy import (
  _apply_motion_source,
  export_actor_model_to_onnx,
  resolve_policy_onnx_path,
)
from tracking_bfm.scripts.export_latent_onnx import (
  parse_args as parse_latent_export_args,
)
from tracking_bfm.scripts.export_onnx import parse_args as parse_policy_export_args


def test_detect_checkpoint_family_tracking() -> None:
  assert detect_checkpoint_family({"actor_state_dict": {}}) == "tracking"


def test_detect_checkpoint_family_distillation() -> None:
  assert detect_checkpoint_family({"policy_state_dict": {}}) == "distillation"


def test_detect_checkpoint_family_rejects_unknown() -> None:
  with pytest.raises(ValueError, match="Unsupported checkpoint format"):
    detect_checkpoint_family({"optimizer_state_dict": {}})


def test_default_output_paths_stay_beside_checkpoint() -> None:
  checkpoint_path = Path("/tmp/run/model_5000.pt")

  assert resolve_onnx_output_path(checkpoint_path) == Path(
    "/tmp/run/deploy_model_5000.onnx"
  )
  assert resolve_policy_onnx_path(checkpoint_path) == Path(
    "/tmp/run/deploy_model_5000.onnx"
  )
  assert resolve_latent_policy_onnx_path(checkpoint_path, output_name="latent") == Path(
    "/tmp/run/latent.onnx"
  )


def test_output_path_guard_rejects_existing_file_without_overwrite(
  tmp_path: Path,
) -> None:
  output_path = tmp_path / "policy.onnx"
  output_path.write_text("existing")

  with pytest.raises(FileExistsError, match="already exists"):
    ensure_output_path_available(output_path, overwrite=False)

  assert ensure_output_path_available(output_path, overwrite=True) == output_path


def test_actor_export_checks_overwrite_before_touching_actor(tmp_path: Path) -> None:
  class Actor:
    called = False

    def as_onnx(self, verbose: bool = False):  # noqa: ANN202
      del verbose
      self.called = True
      raise AssertionError("export should reject before building the ONNX model")

  actor = Actor()
  checkpoint_path = tmp_path / "model_100.pt"
  checkpoint_path.write_text("dummy")
  (tmp_path / "deploy_model_100.onnx").write_text("existing")

  with pytest.raises(FileExistsError, match="already exists"):
    export_actor_model_to_onnx(
      actor=actor,
      checkpoint_path=checkpoint_path,
      task_id="Mjlab-TrackingBFM-Flat-Unitree-G1",
      checkpoint_family="tracking",
      obs_group="actor",
    )

  assert actor.called is False


def test_latent_export_checks_overwrite_before_touching_models(tmp_path: Path) -> None:
  class Actor:
    called = False

    def as_onnx(self, verbose: bool = False):  # noqa: ANN202
      del verbose
      self.called = True
      raise AssertionError("export should reject before building the ONNX model")

  class Decoder:
    called = False

    def as_onnx(self, verbose: bool = False):  # noqa: ANN202
      del verbose
      self.called = True
      raise AssertionError("export should reject before building the ONNX model")

  actor = Actor()
  decoder = Decoder()
  checkpoint_path = tmp_path / "model_100.pt"
  checkpoint_path.write_text("dummy")
  decoder_checkpoint_path = tmp_path / "decoder_100.pt"
  decoder_checkpoint_path.write_text("dummy")
  (tmp_path / "deploy_model_100.onnx").write_text("existing")

  with pytest.raises(FileExistsError, match="already exists"):
    export_actor_decoder_model_to_onnx(
      actor=actor,
      decoder=decoder,
      checkpoint_path=checkpoint_path,
      decoder_checkpoint_path=decoder_checkpoint_path,
      task_id="Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
      obs_group="actor",
      proprio_obs_group="proprio_actor",
      latent_action_clip=0.5,
    )

  assert actor.called is False
  assert decoder.called is False


def test_export_motion_source_wrapper_sets_multi_motion_path() -> None:
  class Command:
    motion_file = "old.npz"
    motion_path = ""

  class EnvCfg:
    commands = {"motion": Command()}

  cfg = EnvCfg()

  _apply_motion_source(cfg, motion_path="motions")

  assert cfg.commands["motion"].motion_path == "motions"
  assert cfg.commands["motion"].motion_file == ""


def test_export_motion_source_wrapper_rejects_motion_path_for_single_command() -> None:
  class Command:
    motion_file = ""

  class EnvCfg:
    commands = {"motion": Command()}

  with pytest.raises(ValueError, match="does not support `motion_path`"):
    _apply_motion_source(EnvCfg(), motion_path="motions")


def test_metadata_helpers_return_minimal_string_metadata() -> None:
  policy_metadata = build_policy_metadata(
    task_id="Mjlab-TrackingBFM-Flat-Unitree-G1",
    obs_group="actor",
    checkpoint_family="tracking",
    robot_name=None,
  )
  latent_metadata = build_latent_policy_metadata(
    task_id="Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
    decoder_checkpoint_path=Path("/tmp/decoder.pt"),
    obs_group="actor",
    proprio_obs_group="proprio_actor",
    robot_name="g1",
  )

  assert policy_metadata == {
    "task_id": "Mjlab-TrackingBFM-Flat-Unitree-G1",
    "obs_group": "actor",
    "checkpoint_family": "tracking",
  }
  assert latent_metadata == {
    "task_id": "Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
    "checkpoint_family": "latent_tracking",
    "decoder_checkpoint": "/tmp/decoder.pt",
    "obs_group": "actor",
    "proprio_obs_group": "proprio_actor",
    "robot_name": "g1",
  }
  assert all(isinstance(value, str) for value in policy_metadata.values())
  assert all(isinstance(value, str) for value in latent_metadata.values())


def test_export_onnx_cli_parses_overwrite_flag() -> None:
  args = parse_policy_export_args(
    [
      "--checkpoint",
      "/tmp/model.pt",
      "--task-id",
      "Mjlab-TrackingBFM-Flat-Unitree-G1",
      "--overwrite",
    ]
  )

  assert args.checkpoint == "/tmp/model.pt"
  assert args.overwrite is True


def test_export_latent_onnx_cli_parses_overwrite_flag() -> None:
  args = parse_latent_export_args(
    [
      "--checkpoint",
      "/tmp/model.pt",
      "--decoder-checkpoint",
      "/tmp/decoder.pt",
      "--task-id",
      "Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage",
      "--overwrite",
    ]
  )

  assert args.decoder_checkpoint == "/tmp/decoder.pt"
  assert args.overwrite is True
