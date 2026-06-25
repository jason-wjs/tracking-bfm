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

`tracking-bfm` should own behavior only when it is BFM-specific or when
upstream does not expose a usable public seam. The current mainline has no
external users, so migration-era compatibility layers are deletion targets.

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
canonical implementation. During the current breaking cleanup, these Modules
should be deleted rather than carried forward.

**legacy-candidate**

The Module or function supports an old fork task surface that is not currently
registered. It should be registered under a canonical primary ID, documented as
old-fork-only, or deleted.

**removed-facade**

The Module was a shallow re-export layer whose callers can use the canonical
implementation Module directly. It should not be recreated unless a compatibility
contract explicitly requires it.

**needs-seam**

The Module contains behavior that belongs behind a deeper interface before
further cleanup, such as Motion source resolution.

## File Audit Matrix

| Path | Label | Rationale | Next action |
| --- | --- | --- | --- |
| `tracking/__init__.py` | bfm-owned | Imports BFM tracking registrations and WBTeleop tracking variant. | Keep; do not register generic upstream tracking here. |
| `tracking/config/__init__.py` | bfm-owned | Package registration entry for BFM G1 configs. | Keep. |
| `tracking/config/g1/__init__.py` | bfm-owned | Registers BFM primary task IDs. | Keep; do not register legacy aliases. |
| `tracking/config/g1/env_cfgs.py` | bfm-owned | Owns G1 BFM env cfgs, including registered optimality probes. | Keep BFM primary cfgs aligned with `docs/migration.md`. |
| `tracking/config/g1/rl_cfg.py` | bfm-owned | Owns BFM runner cfgs for registered tracking variants. | Keep registered configs aligned with task parity decisions. |
| `tracking/env_cfgs.py` | removed-facade | Former public convenience exports duplicated ownership between `tracking/config/g1/` and `tracking/wbteleop/`. | Removed; import env cfg builders from their canonical implementation Modules. |
| `tracking/rl_cfg.py` | removed-facade | Former public convenience exports duplicated ownership between `tracking/config/g1/` and `tracking/wbteleop/`. | Removed; import RL cfg builders from their canonical implementation Modules. |
| `tracking/tracking_env_cfg.py` | bfm-owned | Composes upstream generic tracking env cfg and applies BFM actor terms, body refs, command-class injection, event, reward, and termination deltas locally. | Keep as the BFM delta layer; avoid reintroducing copied upstream sim/action/scene defaults. |
| `tracking/mdp/__init__.py` | bfm-owned | Exposes local tracking MDP terms and commands. | Keep local-only exports while local MDP deltas exist. |
| `tracking/mdp/commands.py` | bfm-owned | Single-motion BFM command runtime; reference `.npz` loading and reindexing live in `tracking/mdp/motion_dataset.py`. | Keep as BFM delta over upstream command behavior; revisit upstream composition only after command runtime stays stable. |
| `tracking/mdp/multi_motion_command.py` | bfm-owned, needs-seam | Canonical BFM multi-motion command runtime; uses the shared reference motion dataset module and delegates adaptive sampling state after cleanup. | Keep; avoid changing public command cfg fields during extraction. |
| `tracking/mdp/adaptive_sampling.py` | bfm-owned | Owns adaptive sampling bin geometry, failure-rate windows, and sampling probabilities for multi-motion tracking. | Keep as the internal adaptive sampling state Module. |
| `tracking/mdp/multi_commands.py` | compat-shim | Historical import path that forwarded to `multi_motion_command.py`. | Deleted during current breaking cleanup. |
| `tracking/mdp/motion_dataset.py` | bfm-owned | Loads reference `.npz` motion tensors, applies motion-type reindexing, stores single/multi motion tensors, and gathers fields by motion id and time step. | Keep as the internal dataset seam shared by single and multi commands. |
| `tracking/mdp/observations.py` | reuse-upstream | Functions are identical to upstream `mjlab.tasks.tracking.mdp.observations`. | Deleted; configs import upstream terms directly. |
| `tracking/mdp/terminations.py` | reuse-upstream | Functions are identical to upstream `mjlab.tasks.tracking.mdp.terminations`. | Deleted; configs import upstream terms directly. |
| `tracking/mdp/metrics.py` | reuse-upstream | Metrics are re-exported from upstream `mjlab.tasks.tracking.mdp.metrics`. | Deleted; evaluate imports upstream metrics directly. |
| `tracking/mdp/rewards.py` | bfm-owned | BFM-specific reward additions remain local. | Keep local additions only; import common upstream rewards directly from `mjlab`. |
| `tracking/rl/__init__.py` | bfm-owned | Exposes local tracking runner. | Keep. |
| `tracking/rl/policy_artifact.py` | bfm-owned | Owns policy artifact paths and motion metadata construction for tracking runner exports. | Keep as the runner artifact helper Module. |
| `tracking/rl/runner.py` | bfm-owned, needs-seam | Owns BFM tracking runner behavior and delegates artifact path/metadata helpers. | Keep learning-loop behavior here; keep artifact details behind `policy_artifact.py`. |
| `tracking/wbteleop/` | bfm-owned | Registered WBTeleop tracking variant built on BFM tracking env cfg and runner semantics. | Keep under `tracking/`; do not promote to top-level task package unless it becomes independent of tracking. |

## Cleanup Order

1. Verify the current standalone baseline.
2. Delete migration-era compatibility shims: legacy task aliases, historical
   MDP import paths, old export console scripts, and old checkpoint migration
   helpers.
3. Add behavior tests around `reuse-upstream` candidates before replacing local
   imports.
4. Implement the Motion source Module described in
   `docs/architecture/motion-source.md`. First migration batch completed for
   export, data processing, evaluate, train, and play.
5. Revisit large `bfm-owned` implementations after Motion source and canonical
   task surface are stable.

## Non-goals

- Do not move `wbteleop/` to a top-level package during this cleanup.
- Do not delete `multi_motion_command.py` or rewrite adaptive sampling as part
  of upstream reuse.
- Do not register old fork task IDs in the current mainline.
- Do not treat file size alone as a reason to split a Module.
