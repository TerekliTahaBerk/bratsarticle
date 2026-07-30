# Swin UNETR M1 execution contract

Swin UNETR is a required experimental comparator, not a literature-score
substitute. Its reportable development matrix contains the same five
patient-level folds and the same five training seeds as every other main
model: 25 runs in total.

The frozen M1 configuration is
`configs/q1q2_v2/swin_m1_runner.yaml`. It uses the MONAI Apache-2.0
implementation, four input modalities, four mutually exclusive output
classes, 96×96×96 patches, microbatch one, and two-step gradient
accumulation. Validation uses untouched complete patient volumes, sliding
window inference, and the common central evaluator. The loss is read only
from the completed development-CV loss-selection freeze.

Every run is trained for at least 10,000 optimizer steps before early
stopping can activate. Exact short- and medium-budget checkpoints and
validation-patient rows are saved at 2,000 and 10,000 steps. The best and
terminal checkpoints remain separate.

MPS exposes a known nondeterministic backward kernel for this architecture.
This is not silently ignored. Before any reportable Swin run, execute the
predeclared real-data repeat-tolerance audit:

```bash
BRATS_CACHE_ROOT=cache/brats-mmap-v2 \
.venv/bin/python scripts/run_q1q2_swin_mps_repeat_tolerance.py \
  --allow-training-diagnostics \
  --dataset-root data/brats2020/BraTS2020_TrainingData
```

The audit compares the loss, logits, and all parameters after one identical
real-data optimization step. A failed audit blocks Swin from the experimental
comparator set and therefore blocks Gate F; it does not permit a seed change
or a literature value.

After a passing audit and frozen loss selection, generate and run the queue:

```bash
.venv/bin/python scripts/generate_q1q2_swin_main_queue.py

BRATS_CACHE_ROOT=cache/brats-mmap-v2 \
.venv/bin/python scripts/run_q1q2_m1_swin_queue.py \
  --allow-reportable-development-training \
  --dataset-root data/brats2020/BraTS2020_TrainingData
```

The queue is restart-safe and refuses to overlap the loss-screen, native,
nnU-Net, or nnU-Net preflight MPS locks. Best, recovery, and terminal
checkpoints are kept separately. External and legacy internal-test access are
prohibited.

After training stops, the exact best checkpoint is reloaded and a second
full-volume sliding-window pass produces every common regional, surface, and
lesion metric. This reportable table is hash-linked to the checkpoint and to
the exact fold, model, and seed. Gate G rejects a selection-only Dice table as
insufficient evidence.

Resource timing is defined at the effective optimizer-step level, so each
Swin measurement includes the frozen two-microbatch gradient accumulation.
After 20 warm-up optimizer steps, 100 synchronized step durations are stored
with total accelerator-hours, parameter count, checkpoint size, and MPS
allocated unified-memory peaks. Gate G requires the complete timing vector.
