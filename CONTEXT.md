# tracking-bfm Context

## Domain Terms

**BFM task package**

`tracking-bfm` is a standalone Python package that contributes BFM task
registrations to `mjlab`. It owns BFM-specific task configs, training/play
entrypoints, export helpers, and motion data processing helpers. It does not
own the upstream `mjlab` runtime.

**mjlab dependency**

`mjlab` is consumed as an external package dependency. Public `mjlab` task
registry, environment, RL, viewer, and utility modules are valid dependency
points. Private `mjlab` script modules are not part of the intended interface
for this package.

**Task registration**

Task registration is the package interface exposed through the
`mjlab.tasks` entry point. Importing `tracking_bfm` imports
`tracking_bfm.tasks`, which populates the `mjlab` task registry.

**Primary task ID**

A primary task ID is the canonical identifier for new training, play, export,
and evaluation workflows. Primary IDs use `BFM` casing.

**Legacy task alias**

A Legacy task alias is an older task identifier kept so previous scripts,
checkpoints, and W&B configs can still resolve to the new package. New
references should prefer primary task IDs.

**Motion source**

A Motion source is the input that supplies reference motion data to a tracking
task. Current forms include a local `motion_file`, a local `motion_path`, a W&B
registry artifact, and a W&B run artifact.

**Single-motion command**

A single-motion command consumes one reference motion file. It is represented
by the `MotionCommandCfg` exported from `tracking_bfm.tasks.tracking.mdp`.

**Multi-motion command**

A multi-motion command can consume a directory of reference motion files or a
single file through compatibility fields. The canonical implementation lives in
`tracking_bfm.tasks.tracking.mdp.multi_motion_command`.

**CLI entrypoint**

A CLI entrypoint is a `pyproject.toml` console script under the
`tracking-bfm-*` command family. CLI entrypoints should stay thin and delegate
domain behavior to reusable modules.

**Quick shell script**

A Quick shell script is a root-level `scripts/*.sh` workflow wrapper around a
package CLI entrypoint. Quick shell scripts are not task-specific launch files.

**Tracking task**

A Tracking task trains or evaluates a policy against reference motion data.
Current registered tracking task surfaces are multi-motion BFM tracking,
1-stage sparse tracking, and WBTeleop tracking.

**Distillation task**

A Distillation task trains a student policy from teacher/reference behavior.
Current registered distillation surfaces include standard BFM distillation and
latent distillation.

**Latent velocity task**

A Latent velocity task uses latent representations for velocity-oriented
control. `LatentRL` is retained as a legacy task alias for this task family.

**ONNX export workflow**

An ONNX export workflow loads a policy checkpoint and task config, applies an
optional Motion source override, and writes an exported policy artifact.

**Data processing workflow**

A Data processing workflow evaluates, filters, generates, or deletes motion
data outside the training loop.

**Diagnostics workflow**

A Diagnostics workflow inspects checkpoints or latent spaces without changing
task registration.

## Architectural Direction

The package should deepen around task registration, Motion source handling,
export, and data processing. CLI files should remain adapters at the outer
seam. Shared domain behavior should move behind small interfaces that provide
Leverage across train, play, evaluate, export, and data processing workflows.
