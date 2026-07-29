# Reproducibility Contract

Every reportable experiment must be recoverable from:

1. Git commit and clean/dirty state
2. locked Python environment
3. data-manifest SHA-256
4. split-manifest SHA-256
5. fully resolved configuration and configuration SHA-256
6. random seed applied to Python, NumPy, PyTorch, and DataLoader workers
7. hardware and software metadata
8. machine-readable per-epoch and per-case results
9. checkpoint selection rule
10. resource profile and completion/failure status

Tables and figures must be regenerated from run artifacts. Manual result entry
is prohibited.

The internal held-out test subset is inaccessible through the ordinary
training loader. Test evaluation requires `--allow-test-evaluation` and creates
an append-only access record in `artifacts/test_access_log.jsonl`.

Baseline checkpoints atomically store model, optimizer, AMP scaler, epoch,
global step, and Python/NumPy/PyTorch CPU/CUDA RNG states. Resume rejects a
configuration-hash mismatch. The CPU regression test requires bit-identical
loss and parameters between uninterrupted and interrupted/resumed execution.

The same checkpoint format is exercised for every Gate 6 architecture. The
machine-readable Gate 6 inventory stores config hashes, feature flags,
parameter counts, closest-width matching outcomes, and tensor traces; the
bounded smoke artifact records the producing Git commit and hardware.

Gate 7 adds a strict run registry with a resolved `config.yaml`,
`metadata.json`, append-only `metrics_per_epoch.jsonl`, patient-level
`validation_per_case.csv`, checkpoint/log directories, and
`resource_profile.json`. Duplicate or unsafe run identifiers are rejected;
failed runs cannot close without an error trace.

Gate 8 checkpoints additionally store the integrated scheduler state. The
pilot plan is hash-linked to its config and clean Git commit, runs one arm per
explicit invocation, and rejects hardware/data preflight failures before
opening development data.

The Gate 8 analyzer accepts only all 12 clean, tagged, in-budget runs with the
frozen hashes, A100 identity, checkpoint, and exact validation-patient set.
Missing or invalid artifacts produce an audit but no shortlist.
