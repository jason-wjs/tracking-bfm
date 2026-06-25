# Motion Source Module Plan

## Context

Motion source handling is currently spread across training, play, evaluation,
export, and data processing workflows. Each caller knows too much about command
cfg shape, `motion_file` versus `motion_path`, W&B artifact lookup, default
`motion.npz` layout, and local path validation.

This document defines the intended deep Module seam before implementation.

## Intended Seam

The Motion source Module should live between workflow adapters and MDP command
configs.

Workflow adapters:

- `tracking_bfm.scripts.train`
- `tracking_bfm.scripts.play`
- `tracking_bfm.scripts.evaluate`
- `tracking_bfm.export`
- `tracking_bfm.data_process`

MDP command configs:

- single-motion `MotionCommandCfg`
- multi-motion `MotionCommandCfg` from
  `tracking_bfm.tasks.tracking.mdp.multi_motion_command`

The Module should not own environment construction, policy loading, rollout,
viewer behavior, or low-level `.npz` tensor loading.

## Candidate Interface

Names are provisional until implementation:

- `MotionSourceSpec`
- `ResolvedMotionSource`
- `MotionCommandSourceShape`
- `MotionSourcePurpose`
- `MotionShard`
- `is_motion_command_cfg(...)`
- `resolve_motion_source(...)`
- `apply_motion_source_to_command(...)`
- `collect_motion_files(...)`

Potential internal adapters:

- `LocalMotionSourceAdapter`
- `WandbRegistryMotionSourceAdapter`
- `WandbRunMotionSourceAdapter`

## First Migration Batch

Start with low-risk call sites where behavior is already isolated:

1. Export local Motion source override in `tracking_bfm.export.onnx_policy`.
2. Data processing motion root resolution and `.npz` collection in
   `tracking_bfm.data_process.motion_filtering`.
3. W&B run motion artifact application in `tracking_bfm.scripts.evaluate`.
4. Training registry artifact application, while preserving the existing
   `registry_name` handoff to the runner.

Implementation status: this batch is implemented by
`tracking_bfm.motion_source`. Workflow adapters may keep temporary compatibility
wrappers, but source mutual exclusion, W&B artifact resolution, command source
shape handling, directory collection, and deterministic sharding should live in
that Module.

## Deferred Work

Defer these until the external seam is stable:

- `tracking_bfm.scripts.play`, because dummy/trained/checkpoint/W&B behavior is
  interleaved and currently biased toward a default single `motion.npz`.
- `multi_motion_command._resolve_motion_files`, because it is runtime command
  implementation and includes distributed sharding behavior.
- `MotionLoader` and `MultiMotionLoader`, because tensor loading and body/joint
  reindexing form a lower-level Motion dataset Module.

## Verification Strategy

The first implementation batch should add tests before moving production code:

- local `motion_file` and `motion_path` are mutually exclusive
- single-motion commands reject `motion_path`
- multi-motion commands accept `motion_path` and clear `motion_file`
- default W&B run artifact layout resolves `motion.npz` for single-motion uses
- directory collection is sorted and deterministic
- sharding is deterministic by rank and world size
