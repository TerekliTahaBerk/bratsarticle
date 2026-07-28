# Gate 2 Completion — Provisional Patient-Level Split

**Decision:** PASS

## Scope completed

- Generated a deterministic patient-level split from the 369-subject BraTS
  2020 canonical cohort.
- Assigned exactly 258 subjects to training, 37 to validation, and 74 to the
  internal held-out test subset.
- Used grade, enhancing-tumor presence, and whole-tumor volume quartile as the
  primary joint stratum; TC and ET quartiles and log-transformed WT/TC/ET
  volumes were included in candidate balance scoring.
- Searched 256 deterministic candidates from the declared base seed
  `20260729`; candidate 160 minimized the predeclared balance objective.
- Kept all outputs provisional. Nothing was copied to `splits/frozen/`.

## Integrity checks

| Check | Result |
|---|---|
| Canonical subjects covered exactly once | PASS (369/369) |
| Train/validation/test ID overlap | PASS (0) |
| Same-role file-hash duplication across subjects | PASS (0) |
| Four-modality image-signature duplication | PASS (0) |
| Deterministic regeneration | PASS |
| Maximum categorical prevalence deviation | PASS (0.0439 ≤ 0.0800) |
| Maximum absolute volume SMD | PASS (0.0484 ≤ 0.3500) |

The three manifests were regenerated twice from the same configuration. Their
SHA-256 values were unchanged:

| Manifest | SHA-256 |
|---|---|
| Train | `49a8d6836aba5f69bb39d5eb513e29c3746adaae72f02ecdc0df094cd86d7425` |
| Validation | `95df721b5b013475b2100224279247e53afcc19c6ab79debf54283d27dc3d5bf` |
| Internal test | `455b3b661be73a84fc99458798ee9a5cbbf9c70deac0b425397220fbbab7a525` |

## Test-access control

The normal development loader accepts only `train` and `validation`.
Internal-test loading requires an explicit `allow_test_evaluation=True`
authorization, a non-empty purpose, and an append-only audit event in
`artifacts/test_access_log.jsonl`. The guard and audit behavior are covered by
unit tests using temporary synthetic manifests. No real internal-test manifest
was opened through the evaluator during Gate 2.

## Automated verification

- Ruff formatting and lint: PASS
- Mypy strict type checking: PASS
- Pytest: PASS (12 tests after the Gate 2 additions)
- Balance figures visually inspected: PASS

## Outputs

- `splits/provisional/train.csv`
- `splits/provisional/validation.csv`
- `splits/provisional/test.csv`
- `splits/provisional/split_metadata.json`
- `splits/provisional/categorical_balance.csv`
- `splits/provisional/continuous_balance.csv`
- `reports/split_balance_report.md`
- `figures/split_balance_grade.png`
- `figures/split_balance_et_presence.png`
- `figures/split_balance_total_tumor_quartiles.png`
- `figures/split_balance_volume_smd.png`

