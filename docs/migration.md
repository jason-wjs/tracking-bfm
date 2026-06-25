# tracking-bfm Migration Notes

This document records task and feature parity decisions while moving BFM work
out of the original `mjlab` fork and into a standalone package.

## Dependency Policy

`tracking-bfm` depends on `mjlab` as an external package. It does not vendor
`src/mjlab` and does not import private `mjlab.scripts.*` modules. See
`docs/adr/0001-standalone-mjlab-dependency.md`.

## Task ID Migration

| Task family | Primary task ID | Legacy alias(es) | Registration path | Runner class | Status | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-motion tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1` | `Mjlab-Trackingbfm-Flat-Unitree-G1` | `tracking_bfm.tasks.tracking.config.g1` | `MjlabOnPolicyRunner` default | Registered | Canonical BFM tracking task. |
| 1-stage sparse tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.tracking.config.g1` | `MjlabOnPolicyRunner` default | Registered | Sparse-observation tracking variant. |
| WBTeleop tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop` | `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` | `tracking_bfm.tasks.tracking.wbteleop` | `tracking_bfm.tasks.tracking.wbteleop.runner.WbTeleopTrackingRunner` | Registered | WBTeleop-specific runner and algorithm. |
| Distillation | `Mjlab-DistillationBFM-Flat-Unitree-G1` | `Mjlab-Distillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Registered | Standard BFM distillation task. |
| Latent distillation | `Mjlab-LatentDistillationBFM-Flat-Unitree-G1` | `Mjlab-LatentDistillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Registered | Latent distillation runner config. |
| Latent tracking | `Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.latent_tracking.config.g1` | `tracking_bfm.tasks.latent_tracking.rl.LatentTrackingRunner` | Registered | Latent tracking 1-stage task. |
| Latent velocity | `Mjlab-LatentVelocityBFM-Flat-Unitree-G1` | `Mjlab-LatentRL-Flat-Unitree-G1` | `tracking_bfm.tasks.latent_velocity.config.g1` | `tracking_bfm.tasks.latent_velocity.rl.LatentVelocityRunner` | Registered | `LatentRL` remains a legacy alias. |

## Feature Parity Decisions

Rows marked `Needs decision` are pending; do not register or delete their task
IDs, config builders, or runner config builders without first updating this
table and the corresponding tests.

| Feature / variant | Legacy ID(s) | Current evidence | Current status | Decision | Compatibility action | Verification source | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-motion tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1` | Registered in `tracking/config/g1/__init__.py` | Registered | Keep | Preserve alias and primary ID. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| 1-stage sparse tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage` | Registered in `tracking/config/g1/__init__.py` | Registered | Keep | Preserve alias and primary ID. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| WBTeleop tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` | Registered in `tracking/wbteleop/__init__.py` | Registered | Keep | Preserve alias and primary ID. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| Distillation | `Mjlab-Distillation-Flat-Unitree-G1` | Registered in `distillation/config/g1/__init__.py` | Registered | Keep | Preserve alias and primary ID. | `tests/tasks/distillation/test_distillation_tasks.py` | None. |
| Latent distillation | `Mjlab-LatentDistillation-Flat-Unitree-G1` | Registered in `distillation/config/g1/__init__.py` | Registered | Keep | Preserve alias and primary ID. | `tests/tasks/distillation/test_distillation_tasks.py` | None. |
| Latent velocity / LatentRL | `Mjlab-LatentRL-Flat-Unitree-G1` | Registered in `latent_velocity/config/g1/__init__.py` | Registered | Keep alias | Preserve alias and primary ID. | `tests/tasks/latent/test_latent_tasks.py` | None. |
| ActionTrunk | `Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk` | `unitree_g1_flat_tracking_bfm_action_trunk_env_cfg` and action-trunk RL cfg exist. | Code exists, not registered | Needs decision | If still needed, register primary `Mjlab-TrackingBFM-Flat-Unitree-G1-ActionTrunk` plus legacy alias; otherwise mark removed and delete dead config. | Code search plus old fork parity check. | Confirm checkpoint/job dependency. |
| TestOptimal | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal` | Test-optimal env cfg exists in tracking config. | Code exists, not registered | Needs decision | If still needed, register primary `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal` plus legacy alias; otherwise mark removed. | Code search plus old fork parity check. | Confirm whether this was only an experiment probe. |
| NoRegNoDR | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | Test-optimal env cfg supports disabling regularization and domain randomization. | Code exists, not registered | Needs decision | If still needed, register primary `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR` plus legacy alias; otherwise mark removed. | Code search plus old fork parity check. | Confirm production use. |
| DistillationWbteleopObs | `Mjlab-DistillationWbteleopObs-Flat-Unitree-G1` | `unitree_g1_flat_distillation_wbteleop_obs_env_cfg` exists. | Code exists, not registered | Needs decision | If still needed, register primary `Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs` plus legacy alias; otherwise mark removed and delete dead config. | Code search plus old fork parity check. | Confirm naming before registering. |
| Rough latent velocity | `Mjlab-LatentRL-Rough-Unitree-G1` | Rough latent velocity cfg exists. | Code exists, not registered | Needs decision | If still needed, register primary `Mjlab-LatentVelocityBFM-Rough-Unitree-G1` plus legacy alias; otherwise mark removed. | Code search plus old fork parity check. | Confirm rough-terrain scope for standalone package. |
| Non-BFM tracking | `Mjlab-Tracking-Flat-Unitree-G1`, `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` | Generic tracking builders exist but are not registered in this package. | Not registered | Likely upstream scope | Do not register unless explicitly needed for BFM compatibility. | Old fork parity check. | Treat as `mjlab` scope until confirmed otherwise. |

## Workflow Migration

| Workflow | Console script | Root wrapper | Required inputs | Output | Status | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- |
| Train | `tracking-bfm-train` | `scripts/train.sh` | Task ID plus local Motion source or W&B registry name for tracking tasks. | RSL-RL logs/checkpoints. | Registered | W&B registry Motion source application uses `tracking_bfm.motion_source`; two-stage tyro CLI remains for config overrides. |
| Play | `tracking-bfm-play` | `scripts/play.sh` | Task ID and checkpoint or dummy-agent Motion source depending on mode. | Viewer session or video. | Registered | Motion source handling should be deepened later. |
| Evaluate | `tracking-bfm-evaluate` | `scripts/evaluate.sh` | Task ID and W&B run path. | Metrics JSON or printed metrics. | Registered | W&B run Motion source application uses `tracking_bfm.motion_source`. |
| ONNX export | `tracking-bfm-export-onnx`, `tracking-bfm-export-latent-onnx` | `scripts/export.sh` | Checkpoint and task ID. | ONNX policy artifact. | Registered | Local Motion source override uses `tracking_bfm.motion_source` through the existing export compatibility wrapper. |
| Data processing | `tracking-bfm-filter-motions`, `tracking-bfm-generate-motion-dataset`, `tracking-bfm-delete-failed-motions` | `scripts/data_process.sh` | Motion path/checkpoint/report depending on mode. | Filter report, generated motions, cleanup report. | Registered | Motion root resolution, file collection, and sharding use `tracking_bfm.motion_source` through data-process compatibility wrappers. |
| Diagnostics | `tracking-bfm-analyze-latent-space`, `tracking-bfm-inspect-checkpoint` | `scripts/diagnostics.sh` | Checkpoint/task or checkpoint path. | Plots, JSON, or console inspection. | Registered | Latent analysis uses Motion source overrides and should migrate after export/evaluate. |

## Follow-up Work

1. Decide whether `ActionTrunk`, `TestOptimal`, `NoRegNoDR`,
   `DistillationWbteleopObs`, and rough latent velocity should be registered,
   documented as removed, or deleted.
2. Confirm whether old W&B runs, checkpoints, or scripts still reference each
   unregistered legacy ID.
3. Extend `tracking_bfm.motion_source` only after the current seam is stable,
   starting with deferred `play.py` and runtime loader work described in
   `docs/architecture/motion-source.md`.
4. Keep README primary task IDs aligned with the registered task table above.
