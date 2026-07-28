# Gate 8 Status — Pilot Execution Readiness

**Gate decision:** BLOCKED before reportable training  
**Reason:** Frozen hardware and data-root preflight is not satisfied  
**Internal held-out test access:** Not performed

## Completed preparation

The single-seed development pilot is fully enumerated without scheduling the
six-by-seven factorial grid:

- six architecture-screen arms at fixed CE + soft Dice;
- six additional U-Net loss-screen arms;
- the shared U-Net/CE+Dice arm is reused rather than duplicated;
- 12 unique model/loss runs in total;
- seed `20260729`;
- at most 2,000 optimizer steps or 0.5 GPU-hours per arm;
- validation every 500 steps, with at least three validation checks required;
- patient-wise validation mean regional Dice as the endpoint;
- no internal-test access.

Elimination requires both a paired mean difference worse than the 0.02
practical margin and a paired 95% bootstrap upper bound below zero. If this
does not yield a usable shortlist, the fallback retains the top three
architectures and top two losses. The next stage is predeclared as three seeds;
only finalists may advance to five seeds.

The runner:

- requires `--allow-pilot-training`;
- checks the exact frozen GPU before importing or opening the dataset;
- runs only one named arm per invocation;
- uses the single integrated warm-up/cosine scheduler;
- stores scheduler state in checkpoints;
- stops at the first step or GPU-hour limit;
- reconstructs complete validation volumes and calls the central evaluator;
- writes best-case rows and all run/resource provenance through the registry;
- marks a run invalid if the minimum validation count is not achieved.

## Current preflight result

| Check | Result |
|---|---|
| Canonical manifest exists | PASS |
| Train manifest exists | PASS |
| Validation manifest exists | PASS |
| Test manifest absent from plan | PASS |
| Pilot budget within Gate 7 limits | PASS |
| CUDA available | FAIL |
| Exactly one visible CUDA device | FAIL |
| GPU is `NVIDIA A100-SXM4-80GB` | FAIL |
| `BRATS2020_ROOT` set | FAIL |
| Dataset root exists | FAIL |

No CPU timing, synthetic metric, or Gate 5 memorization result is substituted
for the missing pilot evidence. Consequently Gate 9 shortlist selection cannot
start without violating the declared order.

## Handoff command on an eligible host

After cloning the same commit and setting authorized data/cache roots:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/prepare_gate8_pilots.py \
  --require-eligible-host

PYTHONPATH=src ./.venv/bin/python scripts/run_gate8_pilot.py \
  --arm architecture_unet \
  --allow-pilot-training
```

Each remaining ID is listed in `reports/gate8_pilot_plan.json` and must be
started separately. The machine-readable blocker is
`reports/gate8_preflight.json`.
