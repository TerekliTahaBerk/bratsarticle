# Gate 7 Completion — Fairness Protocol and Experiment Registry

**Decision:** PASS for protocol and registry freeze  
**Reportable pilot eligibility on current host:** FAIL

| Acceptance condition | Result |
|---|---|
| Separate compute-matched protocol | PASS |
| Exact GPU identity and GPU-hour/step limits | PASS |
| Explicit tuning, AMP, input, batch, accumulation, and pretraining fields | PASS |
| Separate convergence-matched protocol | PASS |
| Validation frequency, patience, minimum delta, and checkpoint rule | PASS |
| One integrated warm-up plus cosine scheduler | PASS |
| Epoch count rejected as fairness criterion | PASS |
| Required machine-readable run layout | PASS |
| Commit/dirty/config/data/split/model/resource/test-access metadata | PASS |
| Unsafe/duplicate run IDs rejected | PASS |
| Failed runs require error traces | PASS |

## Frozen values

Both regimes target exactly one `NVIDIA A100-SXM4-80GB`.

- Compute-matched: first of 30,000 optimizer steps or 8.0 GPU-hours.
- Tuning: at most four trials and 16.0 GPU-hours per model/loss family.
- Convergence-matched: at most 50,000 steps, validation every 500 steps,
  patience 12 checks, and minimum improvement 0.001.
- Input/batch: `[4, 240, 240]`, batch 16, accumulation 1.
- Scheduler: one 1,000-step linear warm-up plus cosine decay.
- Pretraining: none; internal held-out test use: prohibited.

The config validation artifact was generated from commit
`fa59bde43ccfca507f5ef2a564e964d1f6916c4f`.

## Verification

- Ruff: PASS
- Mypy strict: PASS
- Pytest: PASS (77 passed)
- CUDA-only evaluator equality: SKIP (CUDA unavailable)
- Fairness config validation: PASS
- Registry layout/lifecycle integration tests: PASS
- Current host eligibility: FAIL (no visible CUDA device)

Gate 7 is complete because the design and enforcement mechanisms are frozen.
The host failure is a Gate 8 execution blocker, not permission to substitute
CPU timing or silently change hardware.

## Artifacts

- `configs/protocols/compute_matched.yaml`
- `configs/protocols/convergence_matched.yaml`
- `configs/experiments/registry.yaml`
- `reports/fairness_and_registry_protocol.md`
- `reports/gate7_protocol_validation.json`
