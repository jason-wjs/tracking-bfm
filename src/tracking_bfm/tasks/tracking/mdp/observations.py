"""Compatibility exports for upstream tracking observation terms."""

from mjlab.tasks.tracking.mdp.observations import (
  motion_anchor_ori_b,
  motion_anchor_pos_b,
  robot_body_ori_b,
  robot_body_pos_b,
)

__all__ = [
  "motion_anchor_ori_b",
  "motion_anchor_pos_b",
  "robot_body_ori_b",
  "robot_body_pos_b",
]
