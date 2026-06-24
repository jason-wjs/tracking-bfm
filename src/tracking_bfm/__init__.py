"""Standalone BFM tracking tasks built on mjlab."""

try:
  from tracking_bfm import tasks as tasks  # noqa: F401
except ModuleNotFoundError as exc:
  if exc.name != "mjlab":
    raise
