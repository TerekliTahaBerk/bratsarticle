# Fair Training and Experiment Registry Protocol

## Scope and freeze point

This protocol governs development-set pilots and finalist training. It does not
authorize internal held-out test access. Values are frozen before Gate 8; any
hardware or budget revision requires a new config version before results are
observed.

The target accelerator is one `NVIDIA A100-SXM4-80GB`. This is a study design
choice that creates one reproducible hardware identity and leaves capacity for
later three-dimensional comparators. Every compared run within a regime must
use that exact model. The present Apple M1 Max host is suitable only for
bounded implementation diagnostics and is ineligible for reportable pilots.

## Compute-matched regime

Each run stops when either 30,000 optimizer steps or 8.0 GPU-hours is reached,
whichever occurs first. Each model/loss family receives no more than four
tuning trials and 16.0 aggregate tuning GPU-hours. Only training and validation
data may be used.

The common training envelope is:

- raw input shape `[4, 240, 240]`;
- batch size 16 and gradient accumulation 1, for effective batch size 16;
- float16 automatic mixed precision;
- one integrated linear-warm-up plus cosine-decay schedule;
- 1,000 warm-up optimizer steps and minimum learning-rate fraction 0.01;
- no pretraining and no fine-tuning cost;
- external pretraining cost marked included, so a future pretrained comparator
  cannot silently omit that cost.

A budget overrun marks a run invalid. Equal epoch count is not a fairness
criterion because architecture-dependent throughput makes an epoch-count
comparison potentially misleading.

## Convergence-matched regime

Each model may run for at most 50,000 optimizer steps. Validation occurs every
500 optimizer steps. The monitored endpoint is validation patient-wise mean
regional Dice, maximized with minimum improvement 0.001. Training stops after
12 validation checks without that improvement.

The only scheduler is the integrated 1,000-step linear warm-up followed by
cosine decay through step 50,000. The selected checkpoint maximizes validation
patient-wise mean regional Dice; lower validation loss breaks a tie. The report
must retain both best and terminal steps and the full 12-check convergence
window.

## Experiment registry

Every reportable run is created under:

```text
artifacts/runs/<run_id>/
  config.yaml
  metadata.json
  metrics_per_epoch.jsonl
  validation_per_case.csv
  checkpoints/
  resource_profile.json
  logs/
```

The registry refuses unsafe or duplicate run identifiers. It writes the fully
resolved config, not merely the source path. Metadata records run ID, Git
commit and dirty state, config/data/split hashes, seed, model, loss, optimizer,
scheduler, hardware and software versions, timestamps, GPU-hours, peak
allocated/reserved VRAM, parameter count, MAC/FLOP input specification, best
checkpoint, completion status, error trace, and test-access state.

Run status begins as `running` and closes as `completed`, `failed`, or
`invalid`. A failed run must include its error trace. Patient-level validation
rows and epoch/checkpoint metrics are machine-readable; result tables must be
generated from these artifacts rather than manually transcribed.

## Guardrails

- The internal held-out test subset is forbidden in both training regimes.
- The ordinary registry default records test access as unauthorized and false.
- Artifact roots are checked against raw-data roots.
- CPU smoke timing is not converted into a surrogate GPU-hour estimate.
- A missing A100, absent resource fields, duplicate run ID, or exhausted budget
  prevents a run from being treated as a valid fair comparison.
