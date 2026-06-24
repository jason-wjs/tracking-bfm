# Tracking BFM Standalone Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new standalone `tracking-bfm` repository that depends on `mjlab==1.4.0` and migrates all seven required BFM task IDs without modifying or vendoring `mjlab`.

**Architecture:** The new package is `tracking_bfm`. Task modules register with `mjlab` through the `mjlab.tasks` entry point, export code lives under `tracking_bfm.export`, reusable motion/data processing under `tracking_bfm.data_process`, and executable Python CLIs under `tracking_bfm.scripts`. The old fork at `/data_team/junsong/tracking/tracking_bfm` is a read-only source.

**Tech Stack:** Python 3.10+, `uv`, `mjlab==1.4.0`, `torch`, `tyro`, `pytest`, `ruff`.

---

## Fixed Scope

Required primary task IDs:

- `Mjlab-TrackingBFM-Flat-Unitree-G1`
- `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage`
- `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop`
- `Mjlab-DistillationBFM-Flat-Unitree-G1`
- `Mjlab-LatentDistillationBFM-Flat-Unitree-G1`
- `Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage`
- `Mjlab-LatentVelocityBFM-Flat-Unitree-G1`

Required legacy aliases:

- `Mjlab-Trackingbfm-Flat-Unitree-G1`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop`
- `Mjlab-Distillation-Flat-Unitree-G1`
- `Mjlab-LatentDistillation-Flat-Unitree-G1`
- `Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage`
- `Mjlab-LatentRL-Flat-Unitree-G1`

Deferred:

- `ActionTrunk`, because it depends on the fork-only `action_trunk_len` core patch.
- `TestOptimal` / `NoRegNoDR`, because they are experiment probes rather than first-release tasks.
- Any changes under `src/mjlab` in the old fork.

## Repository Layout

Create or maintain:

```text
src/tracking_bfm/
  __init__.py
  tasks/
    __init__.py
    registry.py
    tracking/
    distillation/
    latent_tracking/
    latent_velocity/
  export/
  data_process/
  scripts/
    train.py
    play.py
    evaluate.py
    export_onnx.py
    export_latent_onnx.py
    data_process/
    diagnostics/
tests/
```

No `src/mjlab` directory is allowed in this repository.

## Task 1: Repository Scaffold And Registration Infrastructure

**Files:**
- Create/modify: `pyproject.toml`
- Create/modify: `README.md`
- Create: `src/tracking_bfm/__init__.py`
- Create: `src/tracking_bfm/tasks/__init__.py`
- Create: `src/tracking_bfm/tasks/registry.py`
- Test: `tests/test_project_structure.py`

- [ ] **Step 1: Add project metadata**

`pyproject.toml` must define package `tracking-bfm`, dependency `mjlab[cu128]==1.4.0`, entry point group `mjlab.tasks`, and CLI entry points listed in this plan.

- [ ] **Step 2: Add task registration helper**

`tracking_bfm.tasks.registry.register_task_with_aliases()` must call `mjlab.tasks.registry.register_mjlab_task()` once for the primary ID and once for each alias using the same config objects.

- [ ] **Step 3: Add structure tests**

`tests/test_project_structure.py` must assert there is no `src/mjlab`, `pyproject.toml` has the `mjlab.tasks` entry point, and `tracking_bfm` imports.

## Task 2: Tracking Task Migration

**Files:**
- Create: `src/tracking_bfm/tasks/tracking/mdp/multi_motion_command.py`
- Create: `src/tracking_bfm/tasks/tracking/env_cfgs.py`
- Create: `src/tracking_bfm/tasks/tracking/rl_cfg.py`
- Create: `src/tracking_bfm/tasks/tracking/wbteleop/*`
- Create: `src/tracking_bfm/tasks/tracking/__init__.py`
- Test: `tests/tasks/tracking/test_tracking_tasks.py`

- [ ] **Step 1: Migrate multi-motion command**

Port old `src/mjlab/tasks/tracking/mdp/multi_commands.py` to `multi_motion_command.py`, updating imports from `mjlab.tasks.tracking.mdp.multi_commands` to `tracking_bfm.tasks.tracking.mdp.multi_motion_command`.

- [ ] **Step 2: Build BFM tracking env configs without modifying mjlab**

Use public `mjlab` config classes and functions to construct BFM variants. Do not require `make_tracking_env_cfg(motion_command_cfg_cls=...)` because that is a fork-only change. If needed, copy the small BFM-specific builder into `tracking_bfm.tasks.tracking.env_cfgs`.

- [ ] **Step 3: Register primary IDs and aliases**

Register `TrackingBFM`, `TrackingBFM-1Stage`, and `TrackingBFM-WBTeleop` with aliases.

- [ ] **Step 4: Add tests**

Tests must verify full-reference actor exists for `TrackingBFM`, sparse actor terms exist for `1Stage`, and WBTeleop exposes `teacher_actor` while student actor uses limb-reference terms.

## Task 3: Distillation And Latent Distillation Migration

**Files:**
- Create: `src/tracking_bfm/tasks/distillation/**`
- Test: `tests/tasks/distillation/test_distillation_tasks.py`

- [ ] **Step 1: Port distillation MDP, models, algorithm, runner, and teacher adapter**

Update imports to the new package path. Preserve checkpoint formats used by old tests.

- [ ] **Step 2: Register `DistillationBFM` and `LatentDistillationBFM`**

Register primary IDs and legacy aliases. Teacher task references must use primary ID by default where possible.

- [ ] **Step 3: Add tests**

Tests must construct env/rl configs, validate observation groups, and validate latent config fields without requiring GPU training.

## Task 4: Latent Tracking And Latent Velocity Migration

**Files:**
- Create: `src/tracking_bfm/tasks/latent_tracking/**`
- Create: `src/tracking_bfm/tasks/latent_velocity/**`
- Test: `tests/tasks/latent/test_latent_tasks.py`

- [ ] **Step 1: Port latent decoder wrapper and runners**

Update imports to consume `tracking_bfm.tasks.distillation` and `tracking_bfm.tasks.tracking`.

- [ ] **Step 2: Register required task IDs**

Register `LatentTrackingBFM-1Stage` and `LatentVelocityBFM`.

- [ ] **Step 3: Add tests**

Tests must validate task registration, required runner classes, and latent decoder checkpoint config fields.

## Task 5: Export Module Migration

**Files:**
- Create: `src/tracking_bfm/export/__init__.py`
- Create: `src/tracking_bfm/export/onnx_policy.py`
- Create: `src/tracking_bfm/export/onnx_latent_policy.py`
- Create: `src/tracking_bfm/export/checkpoint.py`
- Create: `src/tracking_bfm/export/metadata.py`
- Create: `src/tracking_bfm/scripts/export_onnx.py`
- Create: `src/tracking_bfm/scripts/export_latent_onnx.py`
- Test: `tests/export/test_export.py`

- [ ] **Step 1: Rename old deployment logic to export**

Port old `src/mjlab/deployment/*` into `tracking_bfm.export` with no `deployment` naming in paths or public APIs.

- [ ] **Step 2: Add safe output behavior**

Exports must not overwrite existing files unless an explicit overwrite flag is passed.

- [ ] **Step 3: Add tests**

Tests must cover checkpoint family detection, default output paths, metadata, and overwrite rejection.

## Task 6: Data Process Migration

**Files:**
- Create: `src/tracking_bfm/data_process/motion_filtering.py`
- Create: `src/tracking_bfm/data_process/motion_dataset_generation.py`
- Create: `src/tracking_bfm/data_process/failed_motion_cleanup.py`
- Create: `src/tracking_bfm/scripts/data_process/filter_motions.py`
- Create: `src/tracking_bfm/scripts/data_process/generate_motion_dataset.py`
- Create: `src/tracking_bfm/scripts/data_process/delete_failed_motions.py`
- Test: `tests/data_process/test_data_process.py`

- [ ] **Step 1: Split old data_filtering.py**

Port evaluate/filter behavior to `motion_filtering.py`, generated dataset behavior to `motion_dataset_generation.py`, and delete behavior to `failed_motion_cleanup.py`.

- [ ] **Step 2: Make deletion safe**

Delete command must dry-run by default and require an explicit execute flag to remove files.

- [ ] **Step 3: Add tests**

Tests must cover report parsing, dry-run deletion, and CLI parse.

## Task 7: Script Wrappers And Diagnostics

**Files:**
- Create: `src/tracking_bfm/scripts/train.py`
- Create: `src/tracking_bfm/scripts/play.py`
- Create: `src/tracking_bfm/scripts/evaluate.py`
- Create: `src/tracking_bfm/scripts/diagnostics/analyze_latent_space.py`
- Create: `src/tracking_bfm/scripts/diagnostics/inspect_checkpoint.py`
- Test: `tests/scripts/test_scripts.py`

- [ ] **Step 1: Add thin train/play/evaluate wrappers**

Wrappers must import `tracking_bfm` to trigger task registration and then delegate to `mjlab.scripts.train`, `mjlab.scripts.play`, or tracking evaluation utilities without reimplementing training loops.

- [ ] **Step 2: Port latent-space diagnostic**

Port old `analyze_latent_space.py` into `scripts/diagnostics`, updating imports and task IDs.

- [ ] **Step 3: Add checkpoint diagnostic**

Implement a small checkpoint inspector that prints JSON with checkpoint family and top-level keys.

- [ ] **Step 4: Add tests**

Tests must validate each CLI entry point exists and help/parse paths do not import `src/mjlab`.

## Task 8: Integration Verification

**Files:**
- Modify any files needed to fix integration.

- [ ] **Step 1: Run structure checks**

Run `find src -path '*/mjlab*' -print` and expect no results.

- [ ] **Step 2: Run unit tests**

Run `uv run pytest`.

- [ ] **Step 3: Run CLI smoke tests**

Run `uv run tracking-bfm-inspect-checkpoint --help`, `uv run tracking-bfm-export-onnx --help`, and `uv run list-envs`.

- [ ] **Step 4: Fix failures**

Fix integration issues while preserving module boundaries.
