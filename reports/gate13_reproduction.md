# Gate 13 Clean-Clone Reproducibility Audit

**Decision:** PASS

## Audited snapshot

- Commit: `a7f974e811ba80798b0a520149abfa986cc2ef1b`
- Tree: `31b629ee217d586655b136a7b13605fd4c9ca4ff`
- Tracked manifest entries: 230
- Raw-data environment variables removed: BRATS2019_ROOT, BRATS2020_ROOT, BRATS_CACHE_ROOT
- Started (UTC): 2026-07-29T18:31:10.667475+00:00
- Finished (UTC): 2026-07-29T18:31:59.934885+00:00

## Checks

| Check | Result | Seconds |
|---|---|---:|
| clean local clone | PASS | 0.623 |
| tracked artifact hashes | PASS | 0.486 |
| ruff | PASS | 0.031 |
| mypy | PASS | 12.450 |
| pytest | PASS | 22.249 |
| Gate 12 generation pass 1 | PASS | 6.314 |
| Gate 12 byte identity pass 1 | PASS | 0.074 |
| Gate 12 generation pass 2 | PASS | 6.306 |
| Gate 12 byte identity pass 2 | PASS | 0.073 |
| final tracked artifact hashes | PASS | 0.513 |

## Test result

- Pytest summary: `108 passed, 2 skipped in 21.35s`
- Generated reporting outputs reproduced byte-for-byte: true
- Final clean-clone worktree clean: true

## Scope boundary

This audit reproduces all tracked analyses, figures, tables, hashes, and software tests without raw BraTS roots, caches, or local model checkpoints. Full retraining and a new internal-test inference pass are deliberately outside this clean-clone check: they require the authorized dataset and frozen checkpoint bundle, and a new test pass would require a separately logged access event.
