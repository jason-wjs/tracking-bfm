# Quick shell scripts

These scripts are workflow-level shortcuts for this repository's package console
scripts. They are intentionally thin wrappers around `uv run tracking-bfm-*`.

Use environment variables for common overrides and append any advanced CLI flags
after the script name.

```bash
TASK=Mjlab-TrackingBFM-Flat-Unitree-G1 MOTION_PATH=/path/to/motions \
  ./scripts/train.sh --agent.max-iterations 1000

TASK=Mjlab-TrackingBFM-Flat-Unitree-G1 CHECKPOINT_FILE=/path/to/model.pt \
  MOTION_FILE=/path/to/motion.npz ./scripts/play.sh

WANDB_RUN_PATH=entity/project/run_id ./scripts/evaluate.sh

CHECKPOINT=/path/to/model.pt TASK=Mjlab-DistillationBFM-Flat-Unitree-G1 \
  ./scripts/export.sh

MODE=latent CHECKPOINT=/path/to/latent_actor.pt \
  DECODER_CHECKPOINT=/path/to/decoder.pt ./scripts/export.sh

MODE=filter CHECKPOINT_FILE=/path/to/model.pt MOTION_PATH=/path/to/motions \
  ./scripts/data-process.sh

MODE=checkpoint CHECKPOINT=/path/to/model.pt JSON=1 ./scripts/diagnostics.sh
```
