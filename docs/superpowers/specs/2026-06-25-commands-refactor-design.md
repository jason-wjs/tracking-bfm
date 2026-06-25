# Commands Refactor Design

## Context

`tracking_bfm.tasks.tracking.mdp.commands` and
`tracking_bfm.tasks.tracking.mdp.multi_motion_command` still carry a large part
of the historical forked tracking implementation.

The current cleanup direction is:

```text
tracking_bfm.tasks.tracking = BFM-specific tracking task surface + explicit deltas over mjlab tracking
```

The command layer should follow that direction without breaking registered task
IDs, legacy aliases, checkpoint compatibility, or the runtime behavior already
used by train, play, evaluate, export, distillation, WBTeleop, and data
processing workflows.

## Approved Direction

The commands refactor should proceed in this order:

1. Add runtime smoke coverage with real motion data.
2. Deepen the multi-motion command internals by extracting reference motion
   dataset loading and gathering.
3. Return to the single-motion command and reduce copied upstream logic where it
   can reuse the extracted dataset module.

This intentionally avoids starting with a direct single-command upstream reuse
attempt. The highest-risk and highest-leverage code is the multi-motion command,
but it should be protected by runtime smoke coverage before it is split.

## Target Architecture

### Motion Source Module

`tracking_bfm.motion_source` remains the external seam between workflow adapters
and command configs.

It owns:

- source mutual exclusion
- local file and directory source resolution
- W&B registry and W&B run artifact resolution
- applying a resolved source to `motion_file` or `motion_path`
- deterministic `.npz` collection and rank/world-size sharding helpers

It does not own tensor loading, environment construction, command runtime state,
sampling, rollout, or viewer behavior.

### Reference Motion Dataset Module

Introduce an internal module under the tracking MDP package:

```text
tracking_bfm.tasks.tracking.mdp.motion_dataset
```

This module should own reference motion data loading and indexing:

- `.npz` field loading
- `motion_type` validation
- IsaacLab-to-MuJoCo joint and body reindexing
- selected body indexing
- single-file and multi-file tensor storage
- per-file lengths, fps values, and concatenated frame offsets
- reference field gathering by motion id and time step

The initial interface should be deliberately small. Callers should be able to
construct a dataset from resolved motion files, then ask for tensor fields by
name and frame selection. The command runtime should not know how `.npz` files
are laid out or how IsaacLab and MuJoCo naming orders differ.

### Motion Command Runtime

`commands.py` and `multi_motion_command.py` remain the runtime command modules.

They own:

- command cfg dataclasses
- robot/body index lookup from the MJLab environment
- per-env runtime state including `time_steps`, `motion_idx`, and `motion_length`
- command properties consumed by rewards, observations, terminations, viewer,
  and export
- reset-to-reference and write-to-sim behavior
- debug visualization
- adaptive sampling until a later sampler extraction is justified

They should depend on the reference motion dataset module for motion tensors and
field gathering.

## Public Runtime Interface

Current downstream modules reach into multi-motion internals through
`_gather_motion_field`, `motion_idx`, and `time_steps`.

After the dataset extraction, add this explicit command method:

```text
gather_reference_field(field_name, motion_ids, time_steps)
```

This should be a public method rather than an underscore-prefixed private method.
Distillation and WBTeleop observations should migrate to this method once it
exists.

The command may continue exposing `motion_idx` and `time_steps` because those are
part of the current observation semantics. The refactor should not hide them
until all consumers have a better interface.

## Data Flow

1. Workflow adapter resolves a Motion source through `tracking_bfm.motion_source`.
2. Workflow adapter applies the source to the task env cfg command.
3. MJLab builds the command term from the cfg.
4. The command resolves its runtime file list from `motion_file` or
   `motion_path`.
5. The command constructs the reference motion dataset with body indexes,
   `motion_type`, and device.
6. The command runtime samples motion ids and time steps.
7. Observations, rewards, terminations, data processing, viewer, and export read
   reference tensors through command properties or the new reference gathering
   method.

## Error Handling

The refactor should preserve existing user-facing error semantics where
possible:

- `motion_file` and `motion_path` are mutually exclusive for multi-motion
  command configs.
- `motion_path` must point to a directory containing `.npz` files.
- `motion_file` must point to a file.
- an empty resolved motion list is an error.
- unsupported `motion_type` values raise `ValueError`.
- invalid distributed rank/world-size inputs raise deterministic errors when
  handled by shared helpers.

The new dataset module should prefer `ValueError` for invalid configuration and
`FileNotFoundError` for missing filesystem paths.

## Testing Strategy

### Runtime Smoke First

Before splitting the command implementation, add or document smoke coverage that
uses a small sample from:

```text
/data_zcy/zcy/motion_data/
```

The smoke should verify that at least one real `.npz` motion source can drive:

- primary multi-motion tracking env construction
- reset plus a short step loop
- multi-motion reference field gathering
- the sparse 1-stage tracking actor path
- WBTeleop reference observation terms

The tests should remain bounded and should not require long training.

### Dataset Unit Tests

The dataset extraction should add focused tests for:

- single-file loading
- multi-file concatenation and deterministic ordering
- body index selection
- IsaacLab-to-MuJoCo reindexing
- `motion_type="mujoco"` passthrough
- field gathering with scalar, vector, and windowed time steps
- clamping behavior at sequence boundaries

Small synthetic `.npz` fixtures are enough for these tests.

### Behavior Parity Tests

For each extraction step, keep existing tracking task tests passing and add
targeted parity assertions where the old command behavior was easy to observe:

- command cfg fields are unchanged
- task IDs still build the same command class
- `motion_file` and `motion_path` application still works
- distillation and WBTeleop observations can still gather history/future
  reference fields

## Phased Rollout

### Phase 1: Runtime Smoke

Add the minimal smoke coverage and helper selection logic for test motions.
This phase should not refactor command internals.

### Phase 2: Extract Reference Motion Dataset

Move `MotionLoader`, `MultiMotionLoader`, joint/body reindex constants, and
field gathering mechanics behind `motion_dataset`.

Keep command cfgs and command runtime properties stable.

### Phase 3: Public Reference Gather Method

Add the explicit command gather method and migrate distillation and WBTeleop
observations away from `_gather_motion_field`.

### Phase 4: Single Command Cleanup

Make the single-motion command reuse the same dataset module. Only after this
step should we revisit whether the remaining single-command runtime can compose
or subclass upstream MJLab command behavior.

### Phase 5: Adaptive Sampling Review

After the dataset module is stable, decide whether adaptive sampling should
remain inside `multi_motion_command.py` or move behind an internal
`AdaptiveMotionSampler` module.

## Non-goals

- Do not delete `multi_motion_command.py`.
- Do not change registered task IDs or legacy aliases.
- Do not move WBTeleop to a top-level task package.
- Do not rewrite adaptive sampling in the same phase as dataset extraction.
- Do not require a full training run as a refactor gate.
- Do not depend on private MJLab script modules.

## Success Criteria

The refactor is successful when:

- commands no longer own raw `.npz` loading and reindexing details;
- `multi_motion_command.py` has a smaller runtime-focused implementation;
- distillation and WBTeleop use an explicit reference gathering interface;
- single and multi command loaders share the same dataset implementation;
- all existing tests pass;
- bounded runtime smoke with real test motion data passes.

## Implementation Status

Runtime smoke coverage, reference motion dataset extraction, public reference
gathering, and single-command dataset reuse are implemented. Adaptive sampling
remains inside `multi_motion_command.py` and should be revisited only after the
dataset seam has remained stable through normal training/play usage.
