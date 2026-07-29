# Environment

The canonical local development environment uses Python 3.11. The legacy lock
is retained as `requirements-lock.txt`. The q1q2 v2 Apple environment is frozen
separately in `q1q2_v2-requirements-lock.txt`, with machine-readable provenance
and hashes in `q1q2_v2-environment.json`.

Regenerate the v2 snapshot only after an intentional dependency change:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_q1q2_environment_lock.py
```

The local lock includes exact package versions for audit, tests and MPS
operator smokes. Definitive training requires a separate, cluster-specific
immutable lock and container digest whose GPU model, driver, CUDA, cuDNN,
PyTorch, MONAI and operating-system metadata are recorded in every run
artifact. That environment cannot be frozen until the compute allocation
blocker is resolved.
