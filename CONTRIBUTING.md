# Contributing

Create changes from the current development branch and keep raw data outside
Git. Every scientific change must include its configuration, artifact
provenance, tests, and the claim/report entries it affects.

Before proposing a change:

1. Read `AGENTS.md`, `DATA_USAGE.md`, and the frozen v2 protocols.
2. Run Ruff, strict mypy, Pytest, and the relevant leakage/gate tests.
3. Do not open the legacy internal test or external confirmatory results unless
   the corresponding guarded command and freeze manifest authorize it.
4. Record third-party implementation provenance and license before adding a
   dependency.
5. Never hand-edit reported figure or table values.

Negative results and failed seeds must remain visible. RES and WC are published
BU-Net components and must not be described as novel contributions here.
