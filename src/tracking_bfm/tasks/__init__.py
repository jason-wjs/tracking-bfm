"""Task registrations for tracking-bfm."""

import tracking_bfm.tasks.distillation.config.g1 as _distillation_g1
import tracking_bfm.tasks.latent_tracking.config.g1 as _latent_tracking_g1
import tracking_bfm.tasks.latent_velocity.config.g1 as _latent_velocity_g1
import tracking_bfm.tasks.tracking.config.g1 as _tracking_g1
import tracking_bfm.tasks.tracking.wbteleop as _tracking_wbteleop

__all__: list[str] = []

del (
  _distillation_g1,
  _latent_tracking_g1,
  _latent_velocity_g1,
  _tracking_g1,
  _tracking_wbteleop,
)
