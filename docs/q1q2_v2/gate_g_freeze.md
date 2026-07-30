# Gate G checkpoint and analysis freeze

Gate G is the irreversible boundary between development and the single
external evaluation. Its frozen contract is
`configs/q1q2_v2/gate_g_freeze.yaml`.

The gate requires all 600 prespecified runs:

- 225 native main-convergence runs;
- 25 Swin UNETR main-convergence runs;
- 50 official nnU-Net main-convergence runs;
- 200 four-hour native compute-matched runs; and
- 100 native architecture-by-loss interaction runs.

Native and Swin convergence runs pass only when the prespecified patience rule
stops training after at least 10,000 steps. Reaching the 50,000-step ceiling
without satisfying patience is an extended-training blocker, not evidence of
convergence. Compute-matched runs pass only when the measured four-hour budget
is the stop reason. Official nnU-Net runs require the complete 1,000-epoch
schedule and a final learning rate no greater than 1% of its initial value.

Every run must have a clean start commit, exact model/fold/seed identity,
non-replaced seed, best and terminal/final checkpoints, a resource profile,
and an exact-fold common-metric table hash-linked to its best checkpoint.
Native, Swin, and nnU-Net main runs also require the frozen 2,000- and
10,000-step budget-sensitivity checkpoints. No external prediction or legacy
74-patient inference may precede this audit.

The read-only audit is:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_q1q2_gate_g.py
```

It writes `artifacts/q1q2_v2/gate_g_audit.json` and leaves external inference
prohibited when anything is missing. Once and only once the audit passes,
freeze the checkpoints and all statistical/evaluation inputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_q1q2_gate_g.py \
  --allow-analysis-freeze
```

The command creates an immutable 600-run checkpoint manifest and an analysis
freeze containing hashes for the statistical plan, evaluator, sensitivity
grids, subgroup/robustness protocols, model matrix, seeds, splits, loss
selection, nnU-Net plan selection, and external cohort contract. Gate H must
read this freeze before it can open the external predictions. External
retuning remains prohibited after the gate passes.
