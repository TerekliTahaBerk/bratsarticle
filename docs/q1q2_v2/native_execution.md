# Native M1 development execution contract

The selected architecture-attribution loss is frozen from the complete
15-run development-only loss screen before either reportable native queue is
generated.

The convergence queue contains nine native 2D/2.5D models, five patient-level
folds, and the common five seeds (225 runs). Every run is trained for at least
10,000 optimizer steps before early stopping can activate. Exact short- and
medium-budget checkpoints and validation-patient rows are saved at 2,000 and
10,000 steps. The full result uses the best checkpoint under the frozen
patient-level mean-regional-Dice rule; a separate terminal checkpoint is
mandatory.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/generate_q1q2_native_main_queue.py

BRATS_CACHE_ROOT=cache/brats-mmap-v2 PYTHONPATH=src \
.venv/bin/python scripts/run_q1q2_m1_main_queue.py \
  --allow-reportable-development-training \
  --dataset-root data/brats2020/BraTS2020_TrainingData
```

The compute-matched queue is separate. It contains the eight component-core
models under the same folds, seeds, preprocessing, selected loss, and
validation frequency (200 runs). Each run stops at the first optimizer-step
boundary that reaches four total accelerator-hours. The 30,000-step ceiling
is a safety bound; reaching it before four hours makes the run invalid rather
than silently changing the estimand.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/generate_q1q2_native_compute_matched_queue.py

BRATS_CACHE_ROOT=cache/brats-mmap-v2 PYTHONPATH=src \
.venv/bin/python scripts/run_q1q2_m1_compute_matched_queue.py \
  --allow-reportable-development-training \
  --dataset-root data/brats2020/BraTS2020_TrainingData
```

Both queues are sequential and restart-safe. They refuse concurrent loss,
native, Swin, or nnU-Net MPS work. A failed seed is never replaced. Best,
milestone, recovery, and terminal checkpoints have distinct paths and hashes.
External data and the legacy internal 74-patient subset are not accessible.

The loss-interaction sensitivity uses the primary selected-loss runs plus one
deterministic alternative-loss run for four component-attribution finalists
(100 additional runs). The alternative is the first loss unequal to the
selected loss in the predeclared priority list. nnU-Net is excluded from this
interaction estimand before main outcomes because its official BraTS setup
uses overlapping region targets, which are not mathematically compatible with
the repository's mutually exclusive four-class loss catalog. This exclusion
does not remove nnU-Net from the main strong-baseline matrix.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/generate_q1q2_loss_interaction_queue.py

BRATS_CACHE_ROOT=cache/brats-mmap-v2 PYTHONPATH=src \
.venv/bin/python scripts/run_q1q2_m1_loss_interaction_queue.py \
  --allow-reportable-development-training \
  --dataset-root data/brats2020/BraTS2020_TrainingData
```
