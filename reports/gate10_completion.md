# Gate 10 Completion

**Decision:** PASS — internal-test evaluation is now protocol-eligible.

## Frozen split

- Train patients: 258
- Validation patients: 37
- Internal held-out test patients: 74
- Membership copied byte-for-byte from the passing provisional split.
- Internal-test outcomes were not evaluated during the freeze.

## Frozen candidates

| Candidate | Seeds | Checkpoints |
|---|---|---:|
| unet_reference | 20260729, 20260730, 20260731 | 3 |
| bunet | 20260729, 20260730, 20260731, 20260732, 20260733 | 5 |
| unet_res | 20260729, 20260730, 20260731, 20260732, 20260733 | 5 |

Primary candidate: `bunet`. Every checkpoint is evaluated separately; no seed ensemble is permitted.

## Frozen inference and statistics

- Statistical unit: patient.
- Primary endpoint: patient mean of WT, TC, and ET Dice.
- Candidate value: per-patient arithmetic mean across frozen seeds.
- Confidence intervals: 10,000 paired patient bootstrap resamples.
- Hypothesis tests: 100,000 two-sided paired sign-flip permutations.
- Multiplicity: Holm correction over three primary-endpoint comparisons.
- Secondary endpoints and subgroups: estimation only.
- Post-processing, threshold tuning, checkpoint replacement, and model selection after test access are prohibited.

## Development-derived subgroup thresholds

- Small WT burden: ≤ 64267.000000 mm³
- Medium WT burden: > 64267.000000 and ≤ 123163.333333 mm³
- Large WT burden: > 123163.333333 mm³

These tertiles were derived from the 258 training patients only.

## Guard

The append-only internal-test audit contained no prior test-manifest access event. Gate 11 must use the exact split, checkpoint, evaluator, preprocessing, and statistical-plan hashes frozen here.
