# Tracking Module Cleanup Plan

## Goal

Clean up `tracking_bfm.tasks.tracking` from a migrated fork copy into a BFM
tracking delta package.

The target shape is:

```text
tracking_bfm.tasks.tracking = BFM-specific tracking task surface + explicit deltas over mjlab tracking
```

The target shape is not:

```text
tracking_bfm.tasks.tracking = copied mjlab tracking + scattered glue
```

## Upstream Relationship

The upstream `mjlab` package already provides a generic
`mjlab.tasks.tracking` task family. `tracking-bfm` should not maintain a copy
of that package by default.

`tracking-bfm` should own behavior only when it is BFM-specific, when upstream
does not expose a usable public seam, or when preserving a legacy BFM task
surface requires a local compatibility layer.

## Audit Labels

**reuse-upstream**

The Module appears generic and close enough to upstream `mjlab.tasks.tracking`
that it should be considered for direct reuse or deletion from
`tracking_bfm`.

**bfm-owned**

The Module contains BFM-specific behavior and should remain owned by
`tracking-bfm`.

**compat-shim**

The Module exists to preserve legacy imports or task IDs while forwarding to a
canonical implementation.

**legacy-candidate**

The Module or function supports an old fork task surface that is not currently
registered. It should be registered, documented as removed, or deleted after a
checkpoint/job compatibility decision.

**needs-seam**

The Module contains behavior that belongs behind a deeper interface before
further cleanup, such as Motion source resolution.

## File Audit Matrix

| Path | Label | Rationale | Next action |
| --- | --- | --- | --- |
| `tracking/__init__.py` | bfm-owned | Imports BFM tracking registrations and WBTeleop tracking variant. | Keep; do not register generic upstream tracking here. |
| `tracking/config/__init__.py` | bfm-owned | Package registration entry for BFM G1 configs. | Keep. |
| `tracking/config/g1/__init__.py` | bfm-owned | Registers BFM primary task IDs and legacy aliases. | Keep; update only through migration decisions. |
| `tracking/config/g1/env_cfgs.py` | bfm-owned | Owns G1 BFM env cfgs, including registered optimality probes. | Keep BFM primary cfgs aligned with `docs/migration.md`. |
| `tracking/config/g1/rl_cfg.py` | bfm-owned | Owns BFM runner cfgs for registered tracking variants. | Keep registered configs aligned with task parity decisions. |
| `tracking/env_cfgs.py` | bfm-owned | Public convenience exports for BFM tracking env cfg builders. | Keep; align with registered surface. |
| `tracking/rl_cfg.py` | bfm-owned | Public convenience exports for BFM tracking and WBTeleop runner configs. | Keep; align with registered surface. |
| `tracking/tracking_env_cfg.py` | bfm-owned, needs-seam | Forked from generic tracking but now includes BFM actor terms, body refs, command-class injection, and reward/termination deltas. | Keep for now; later isolate BFM deltas from reusable upstream pieces. |
| `tracking/mdp/__init__.py` | bfm-owned | Re-exports local tracking MDP terms and commands. | Keep while local MDP deltas exist. |
| `tracking/mdp/commands.py` | bfm-owned | Single-motion command differs from upstream and supports BFM metrics/behavior. | Keep; later compare for possible lower-level Motion dataset seam. |
| `tracking/mdp/multi_motion_command.py` | bfm-owned, needs-seam | Canonical BFM multi-motion command implementation. | Keep; do not split until Motion source seam is stable. |
| `tracking/mdp/multi_commands.py` | compat-shim | Historical import path that forwards to `multi_motion_command.py`. | Keep until legacy import path can be removed in a breaking change. |
| `tracking/mdp/observations.py` | reuse-upstream | Functions are identical to upstream `mjlab.tasks.tracking.mdp.observations`. | Keep as explicit upstream function re-export; do not wildcard-export upstream globals into the local MDP namespace. |
| `tracking/mdp/terminations.py` | reuse-upstream | Functions are identical to upstream `mjlab.tasks.tracking.mdp.terminations`. | Keep as explicit upstream function re-export; do not wildcard-export upstream globals into the local MDP namespace. |
| `tracking/mdp/metrics.py` | bfm-owned | Similar to old fork and tied to local command shape. | Keep until command interfaces are clarified. |
| `tracking/mdp/rewards.py` | bfm-owned | Contains BFM-specific reward additions such as joint action rate and body tracking deltas. | Keep; consider extracting BFM-only reward terms later. |
| `tracking/rl/__init__.py` | bfm-owned | Exposes local tracking runner. | Keep. |
| `tracking/rl/runner.py` | bfm-owned, needs-seam | Owns BFM tracking runner behavior including artifact/adaptive sampling coupling. | Keep; move Motion source logic out before deeper runner cleanup. |
| `tracking/wbteleop/` | bfm-owned | Registered WBTeleop tracking variant built on BFM tracking env cfg and runner semantics. | Keep under `tracking/`; do not promote to top-level task package unless it becomes independent of tracking. |

## Cleanup Order

1. Commit the current standalone boundary fixes and architecture docs.
2. Add behavior tests around `reuse-upstream` candidates before replacing local
   imports. Completed for upstream observation and termination terms.
3. Implement the Motion source Module described in
   `docs/architecture/motion-source.md`. First migration batch completed for
   export, data processing, evaluate, train, and play.
4. Resolve `legacy-candidate` entries through `docs/migration.md`.
5. Revisit large `bfm-owned` implementations after Motion source and legacy
   task surface are stable.

## Non-goals

- Do not move `wbteleop/` to a top-level package during this cleanup.
- Do not delete `multi_motion_command.py` or rewrite adaptive sampling as part
  of upstream reuse.
- Do not register old fork task IDs without confirming checkpoint/job
  compatibility requirements.
- Do not treat file size alone as a reason to split a Module.
