"""Compatibility exports for upstream tracking termination terms."""

from mjlab.tasks.tracking.mdp.terminations import (
  bad_anchor_ori,
  bad_anchor_pos,
  bad_anchor_pos_z_only,
  bad_motion_body_pos,
  bad_motion_body_pos_z_only,
)

__all__ = [
  "bad_anchor_ori",
  "bad_anchor_pos",
  "bad_anchor_pos_z_only",
  "bad_motion_body_pos",
  "bad_motion_body_pos_z_only",
]
