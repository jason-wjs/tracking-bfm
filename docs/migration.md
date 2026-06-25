# tracking-bfm Migration Notes

This document records task and feature parity decisions while moving BFM work
out of the original `mjlab` fork and into a standalone package.

## Dependency Policy

`tracking-bfm` depends on `mjlab` as an external package. It does not vendor
`src/mjlab` and does not import private `mjlab.scripts.*` modules. See
`docs/adr/0001-standalone-mjlab-dependency.md`.

## Task ID Migration

| Task family | Primary task ID | Removed legacy alias(es) | Registration path | Runner class | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Multi-motion tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1` | `Mjlab-Trackingbfm-Flat-Unitree-G1` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| 1-stage sparse tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| WBTeleop tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop` | `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` | `tracking_bfm.tasks.tracking.wbteleop` | `tracking_bfm.tasks.tracking.wbteleop.runner.WbTeleopTrackingRunner` | Primary registered | Legacy alias removed from current mainline. |
| Test-optimal tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal` | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| Test-optimal no-reg/no-DR tracking | `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | `tracking_bfm.tasks.tracking.config.g1` | `tracking_bfm.tasks.tracking.rl.MotionTrackingOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| Distillation | `Mjlab-DistillationBFM-Flat-Unitree-G1` | `Mjlab-Distillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Primary registered | Legacy alias removed from current mainline. |
| Distillation WBTeleop observations | `Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs` | `Mjlab-DistillationWbteleopObs-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Primary registered | Legacy alias removed from current mainline. |
| Latent distillation | `Mjlab-LatentDistillationBFM-Flat-Unitree-G1` | `Mjlab-LatentDistillation-Flat-Unitree-G1` | `tracking_bfm.tasks.distillation.config.g1` | `tracking_bfm.tasks.distillation.rl.DistillationRunner` | Primary registered | Legacy alias removed from current mainline. |
| Latent tracking | `Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage` | `Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage` | `tracking_bfm.tasks.latent_tracking.config.g1` | `tracking_bfm.tasks.latent_tracking.rl.LatentTrackingOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| Latent velocity | `Mjlab-LatentVelocityBFM-Flat-Unitree-G1` | `Mjlab-LatentRL-Flat-Unitree-G1` | `tracking_bfm.tasks.latent_velocity.config.g1` | `tracking_bfm.tasks.latent_velocity.rl.LatentVelocityOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |
| Rough latent velocity | `Mjlab-LatentVelocityBFM-Rough-Unitree-G1` | `Mjlab-LatentRL-Rough-Unitree-G1` | `tracking_bfm.tasks.latent_velocity.config.g1` | `tracking_bfm.tasks.latent_velocity.rl.LatentVelocityOnPolicyRunner` | Primary registered | Legacy alias removed from current mainline. |

## Feature Parity Decisions

Legacy IDs are retained here as migration history only. They are not registered
by the current mainline. Old checkpoints or scripts that require these IDs
belong on the archived fork.

| Feature / variant | Legacy ID(s) | Current evidence | Current status | Decision | Compatibility action | Verification source | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-motion tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-TrackingBFM-Flat-Unitree-G1`. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| 1-stage sparse tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage`. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| WBTeleop tracking | `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop`. | `tests/tasks/tracking/test_tracking_tasks.py` | None. |
| Distillation | `Mjlab-Distillation-Flat-Unitree-G1` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-DistillationBFM-Flat-Unitree-G1`. | `tests/tasks/distillation/test_distillation_tasks.py` | None. |
| Latent distillation | `Mjlab-LatentDistillation-Flat-Unitree-G1` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-LatentDistillationBFM-Flat-Unitree-G1`. | `tests/tasks/distillation/test_distillation_tasks.py` | None. |
| Latent velocity / LatentRL | `Mjlab-LatentRL-Flat-Unitree-G1` | Registered before compatibility cleanup. | Removed from current mainline | Delete alias | Register only `Mjlab-LatentVelocityBFM-Flat-Unitree-G1`. | `tests/tasks/latent/test_latent_tasks.py` | None. |
| ActionTrunk | `Mjlab-Trackingbfm-Flat-Unitree-G1-ActionTrunk` | The original fork added action trunk fields to MJLab core action management. Official `mjlab==1.4.0` does not expose this Interface, and current tracking-bfm work no longer needs it. | Removed | Delete | Do not register the primary ID or legacy alias. Existing checkpoints that require this task must stay on the old fork. | User decision on 2026-06-25 plus code search. | Removal avoids carrying fork-only MJLab core assumptions in the standalone package. |
| TestOptimal | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal` | Test-optimal env cfg uses full critic observations for actor and global body pose rewards. | Removed from current mainline | Delete alias | Register only `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal`. | `tests/tasks/tracking/test_tracking_tasks.py` | Optimality probe retained under canonical ID. |
| NoRegNoDR | `Mjlab-Trackingbfm-Flat-Unitree-G1-TestOptimal-NoRegNoDR` | Test-optimal env cfg supports disabling regularization and domain randomization. | Removed from current mainline | Delete alias | Register only `Mjlab-TrackingBFM-Flat-Unitree-G1-TestOptimal-NoRegNoDR`. | `tests/tasks/tracking/test_tracking_tasks.py` | Pure optimality probe retained under canonical ID. |
| DistillationWbteleopObs | `Mjlab-DistillationWbteleopObs-Flat-Unitree-G1` | `unitree_g1_flat_distillation_wbteleop_obs_env_cfg` builds WBTeleop-style student observations. | Removed from current mainline | Delete alias | Register only `Mjlab-DistillationBFM-Flat-Unitree-G1-WBTeleopObs`. | `tests/tasks/distillation/test_distillation_tasks.py` | Retained under canonical ID. |
| Rough latent velocity | `Mjlab-LatentRL-Rough-Unitree-G1` | Rough latent velocity cfg adapts the upstream rough G1 velocity task for latent decoder control. | Removed from current mainline | Delete alias | Register only `Mjlab-LatentVelocityBFM-Rough-Unitree-G1`. | `tests/tasks/latent/test_latent_tasks.py` | Retained under canonical ID. |
| Non-BFM tracking | `Mjlab-Tracking-Flat-Unitree-G1`, `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` | Generic tracking builders exist but are not registered in this package. | Not registered | Likely upstream scope | Do not register unless explicitly needed for BFM compatibility. | Old fork parity check. | Treat as `mjlab` scope until confirmed otherwise. |

## Workflow Migration

| Workflow | Console script | Root wrapper | Required inputs | Output | Status | Notes / TBD |
| --- | --- | --- | --- | --- | --- | --- |
| Train | `tracking-bfm-train` | `scripts/train.sh` | Task ID plus local Motion source or W&B registry name for tracking tasks. | RSL-RL logs/checkpoints. | Registered | W&B registry Motion source application uses `tracking_bfm.motion_source`; two-stage tyro CLI remains for config overrides. |
| Play | `tracking-bfm-play` | `scripts/play.sh` | Task ID and checkpoint or dummy-agent Motion source depending on mode. | Viewer session or video. | Registered | Motion source handling uses `tracking_bfm.motion_source`; play supports explicit `motion_file`, explicit `motion_path`, W&B registry artifacts, and W&B run motion artifacts. |
| Evaluate | `tracking-bfm-evaluate` | `scripts/evaluate.sh` | Task ID and W&B run path. | Metrics JSON or printed metrics. | Registered | W&B run Motion source application uses `tracking_bfm.motion_source`. |
| ONNX export | `tracking-bfm-export policy`, `tracking-bfm-export latent` | `scripts/export.sh` | Checkpoint and task ID. | ONNX policy artifact. | Registered | Local Motion source override uses `tracking_bfm.motion_source`; export startup is one canonical CLI. |
| Data processing | `tracking-bfm-filter-motions`, `tracking-bfm-generate-motion-dataset`, `tracking-bfm-delete-failed-motions` | `scripts/data_process.sh` | Motion path/checkpoint/report depending on mode. | Filter report, generated motions, cleanup report. | Registered | Motion root resolution, file collection, and sharding use `tracking_bfm.motion_source` through data-process compatibility wrappers. |
| Diagnostics | `tracking-bfm-analyze-latent-space`, `tracking-bfm-inspect-checkpoint` | `scripts/diagnostics.sh` | Checkpoint/task or checkpoint path. | Plots, JSON, or console inspection. | Registered | Latent analysis uses Motion source overrides and should migrate after export/evaluate. |

## Follow-up Work

1. Extend `tracking_bfm.motion_source` only after the current seam is stable,
   starting with runtime loader work described in
   `docs/architecture/motion-source.md`.
2. Keep README primary task IDs aligned with the registered task table above.
