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
| Multi-motion tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1` | `Mjlab-Trackingbfm-Flat-Unitree-G1` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Registered | Canonical BFM tracking task. |
| 1-stage sparse tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Registered | Sparse-observation tracking variant. |
| WBTeleop tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop` | `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` | `tracking_bfm.tasks.tracking.wbteleop` | `tracking_bfm.tasks.tracking.wbteleop.runner.WbTeleopTrackingRunner` | Registered | WBTeleop-specific runner and algorithm. |
| Test-optimal tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal` | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Registered | Optimality probe with full critic observations and global body pose rewards. |
| Test-optimal no-reg/no-DR tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Registered | Pure optimality probe with regularization rewards and domain randomization disabled. |
| Distillation | `Mjlab-DistillationBFM-Flat-Unitree-G1` | `Mjlab-Distillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Registered | Standard BFM distillation task. |
| Distillation WBTeleop observations | `Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs` | `Mjlab-DistillationWbteleopObs-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Registered | Standard distillation runner with WBTeleop-style student observations. |
| Latent distillation | `Mjlab-LatentDistillationBFM-Flat-Unitree-G1` | `Mjlab-LatentDistillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Registered | Latent distillation runner config. |
| Latent tracking | `Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.latent_tracking.config.g1` | `tracking_bfm.tasks.latent_tracking.rl.LatentTrackingOnPolicyRunner` | Registered | Latent tracking 1-stage task. |
| Latent velocity | `Mjlab-LatentVelocityBFM-Flat-Unitree-G1` | `Mjlab-LatentRL-Flat-Unitree-G1` | `tracking_bfm.tasks.latent_velocity.config.g1` | `tracking_bfm.tasks.latent_velocity.rl.LatentVelocityOnPolicyRunner` | Registered | `LatentRL` remains a legacy alias. |
| Rough latent velocity | `Mjlab-LatentVelocityBFM-Rough-Unitree-G1` | `Mjlab-LatentRL-Rough-Unitree-G1` | `tracking_bfm.tasks.latent_velocity.config.g1` | `tracking_bfm.tasks.latent_velocity.rl.LatentVelocityOnPolicyRunner` | Registered | Rough-terrain latent velocity task; `LatentRL-Rough` remains a legacy alias. |

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
| ActionTrunk | `Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk` | The original fork added action trunk fields to MJLab core action management. Official `mjlab==1.4.0` does not expose this Interface, and current tracking-bfm work no longer needs it. | Removed | Delete | Do not register the primary ID or legacy alias. Existing checkpoints that require this task must stay on the old fork. | User decision on 2026-06-25 plus code search. | Removal avoids carrying fork-only MJLab core assumptions in the standalone package. |
| TestOptimal | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal` | Test-optimal env cfg uses full critic observations for actor and global body pose rewards. | Registered | Keep | Register primary `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal` plus legacy alias. | `tests/tasks/tracking/test_tracking_tasks.py` | Optimality probe retained for compatibility and diagnostics. |
| NoRegNoDR | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | Test-optimal env cfg supports disabling regularization and domain randomization. | Registered | Keep | Register primary `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR` plus legacy alias. | `tests/tasks/tracking/test_tracking_tasks.py` | Pure optimality probe retained for compatibility and diagnostics. |
| DistillationWbteleopObs | `Mjlab-DistillationWbteleopObs-Flat-Unitree-G1` | `unitree_g1_flat_distillation_wbteleop_obs_env_cfg` builds WBTeleop-style student observations. | Registered | Keep | Register primary `Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs` plus legacy alias. | `tests/tasks/distillation/test_distillation_tasks.py` | Retained for compatibility and WBTeleop-observation student distillation. |
| Rough latent velocity | `Mjlab-LatentRL-Rough-Unitree-G1` | Rough latent velocity cfg adapts the upstream rough G1 velocity task for latent decoder control. | Registered | Keep | Register primary `Mjlab-LatentVelocityBFM-Rough-Unitree-G1` plus legacy alias. | `tests/tasks/latent/test_latent_tasks.py` | Retained for rough-terrain latent velocity experiments. |
| Non-BFM tracking | `Mjlab-Tracking-Flat-Unitree-G1`, `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` | Generic tracking builders exist but are not registered in this package. | Not registered | Likely upstream scope | Do not register unless explicitly needed for BFM compatibility. | Old fork parity check. | Treat as `mjlab` scope until confirmed otherwise. |

## Workflow Migration

| Workflow | Console script | Root wrapper | Required inputs | Output | Status | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- |
| Train | `tracking-bfm-train` | `scripts/train.sh` | Task ID plus local Motion source or W&B registry name for tracking tasks. | RSL-RL logs/checkpoints. | Registered | W&B registry Motion source application uses `tracking_bfm.motion_source`; two-stage tyro CLI remains for config overrides. |
| Play | `tracking-bfm-play` | `scripts/play.sh` | Task ID and checkpoint or dummy-agent Motion source depending on mode. | Viewer session or video. | Registered | Motion source handling uses `tracking_bfm.motion_source`; play supports explicit `motion_file`, explicit `motion_path`, W&B registry artifacts, and W&B run motion artifacts. |
| Evaluate | `tracking-bfm-evaluate` | `scripts/evaluate.sh` | Task ID and W&B run path. | Metrics JSON or printed metrics. | Registered | W&B run Motion source application uses `tracking_bfm.motion_source`. |
| ONNX export | `tracking-bfm-export-onnx`, `tracking-bfm-export-latent-onnx` | `scripts/export.sh` | Checkpoint and task ID. | ONNX policy artifact. | Registered | Local Motion source override uses `tracking_bfm.motion_source` through the existing export compatibility wrapper. |
| Data processing | `tracking-bfm-filter-motions`, `tracking-bfm-generate-motion-dataset`, `tracking-bfm-delete-failed-motions` | `scripts/data_process.sh` | Motion path/checkpoint/report depending on mode. | Filter report, generated motions, cleanup report. | Registered | Motion root resolution, file collection, and sharding use `tracking_bfm.motion_source` through data-process compatibility wrappers. |
| Diagnostics | `tracking-bfm-analyze-latent-space`, `tracking-bfm-inspect-checkpoint` | `scripts/diagnostics.sh` | Checkpoint/task or checkpoint path. | Plots, JSON, or console inspection. | Registered | Latent analysis uses Motion source overrides and should migrate after export/evaluate. |

## Follow-up Work

1. Confirm whether old W&B runs, checkpoints, or scripts still reference any
   remaining unregistered legacy IDs.
2. Extend `tracking_bfm.motion_source` only after the current seam is stable,
   starting with runtime loader work described in
   `docs/architecture/motion-source.md`.
3. Keep README primary task IDs aligned with the registered task table above.
