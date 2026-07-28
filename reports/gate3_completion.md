# Gate 3 Completion — Central Evaluator

**Decision:** PASS

## Completed

- Implemented one central patient-volume evaluator under `src/evaluation/`.
- Added label, four-class softmax, and three-channel nested-sigmoid decoding.
- Made nested consistency optional, named, counted, and ablation-ready.
- Implemented all primary and secondary voxel-wise and lesion-wise metrics
  listed in the protocol.
- Declared connectivity, lesion filtering, matching, empty-mask, HD95, Surface
  Dice, and post-processing behavior in a versioned YAML configuration.
- Added separate raw/filtered evaluation stages; raw is the only default.
- Added deterministic patient-level summaries that expose NaN and infinity
  counts.
- Used the licensed Google DeepMind surface-distance implementation as a
  dependency; no external source code was copied into the repository.

## Verification

| Check | Result |
|---|---|
| Ruff formatting/lint | PASS |
| Mypy strict | PASS |
| Full Pytest suite | PASS (27 passed) |
| CUDA-only consistency | SKIP (CUDA unavailable on Apple M1 Max) |
| Internal held-out test access | Not performed |

The complete metric contract and edge-case rules are recorded in
`reports/evaluator_specification.md`.

