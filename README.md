# tracking-bfm

Standalone BFM tracking tasks built on `mjlab==1.4.0`.

This repository intentionally does not fork or vendor `mjlab`. It registers BFM tasks
through the `mjlab.tasks` entry point mechanism under the `tracking_bfm` package.

## Install

```bash
uv sync
uv run pytest
```

## Primary Task IDs

Use these IDs for new training, play, export, and evaluation runs:

- `Mjlab-TrackingBFM-Flat-Unitree-G1`
- `Mjlab-TrackingBFM-Flat-Unitree-G1-1Stage`
- `Mjlab-TrackingBFM-Flat-Unitree-G1-WBTeleop`
- `Mjlab-DistillationBFM-Flat-Unitree-G1`
- `Mjlab-LatentDistillationBFM-Flat-Unitree-G1`
- `Mjlab-LatentTrackingBFM-Flat-Unitree-G1-1Stage`
- `Mjlab-LatentVelocityBFM-Flat-Unitree-G1`

## Legacy Aliases

The package also registers legacy aliases for compatibility with older jobs,
checkpoints, and scripts. Prefer the primary IDs above in new references.

- `Mjlab-Trackingbfm-Flat-Unitree-G1`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-1Stage`
- `Mjlab-Trackingbfm-Flat-Unitree-G1-wbteleop`
- `Mjlab-Distillation-Flat-Unitree-G1`
- `Mjlab-LatentDistillation-Flat-Unitree-G1`
- `Mjlab-LatentTrackingbfm-Flat-Unitree-G1-1Stage`
- `Mjlab-LatentRL-Flat-Unitree-G1`

## Module Boundaries

- `tracking_bfm.tasks` owns BFM task config construction and registration.
- `tracking_bfm.export` owns ONNX export helpers and checkpoint metadata.
- `tracking_bfm.data_process` owns reusable motion filtering, dataset generation,
  and failed-motion cleanup helpers.
- `tracking_bfm.scripts` owns thin CLI wrappers that import `tracking_bfm` to
  register tasks and then delegate to reusable modules.

No `src/mjlab` package belongs in this repository. Upstream `mjlab` is consumed as
a dependency only.

## Use

```bash
uv run tracking-bfm-train Mjlab-TrackingBFM-Flat-Unitree-G1
uv run tracking-bfm-play Mjlab-TrackingBFM-Flat-Unitree-G1
uv run tracking-bfm-evaluate Mjlab-TrackingBFM-Flat-Unitree-G1
uv run tracking-bfm-export-onnx --help
uv run tracking-bfm-export-latent-onnx --help
uv run tracking-bfm-filter-motions --help
uv run tracking-bfm-generate-motion-dataset --help
uv run tracking-bfm-delete-failed-motions --help
uv run tracking-bfm-analyze-latent-space --help
uv run tracking-bfm-inspect-checkpoint --help
```
