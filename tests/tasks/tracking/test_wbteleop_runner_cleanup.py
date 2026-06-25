from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from tracking_bfm.tasks.tracking.wbteleop.runner import WbTeleopTrackingRunner


def test_wbteleop_runner_does_not_expose_legacy_checkpoint_migration() -> None:
  assert not hasattr(WbTeleopTrackingRunner, "_migrate_legacy_checkpoint")
  assert not hasattr(WbTeleopTrackingRunner, "_migrate_actor_distribution_keys")
