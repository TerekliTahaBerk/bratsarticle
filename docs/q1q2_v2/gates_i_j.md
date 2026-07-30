# Q1/Q2 v2 Gates I and J

Gate I is the final reproducibility audit for the new study. It is separate
from the immutable legacy Gate 13 package and cannot use the legacy
74-patient internal subset as new evidence.

After Gates G and H and all downstream analyses complete:

```bash
.venv/bin/python scripts/audit_q1q2_gate_i.py --clean-clone
.venv/bin/python scripts/audit_q1q2_gate_i.py --build
.venv/bin/python scripts/audit_q1q2_gate_i.py --verify
```

The build fails unless all 600 development runs share one recorded training
commit, all 300 main fold-seed checkpoints completed Gate H, all 12
model-level prediction manifests contain 146 patients, and every declared
analysis/figure/qualitative hash still matches. Verification is artifact-only:
it opens neither raw BraTS data nor a model inference session.

The numerical statistics, subgroup tables, resource tables, and standard
result figures are regenerated and must remain byte-identical. Existing
qualitative panels are hash-verified; rerendering those panels additionally
requires the ignored, derived external-image cache and is therefore reported
as a separate scope boundary rather than falsely called data-free.

Gate J prevents manual transcription of result values:

```bash
.venv/bin/python scripts/build_q1q2_claim_package.py \
  --template manuscript/q1q2_v2_manuscript.template.md
.venv/bin/python scripts/build_q1q2_claim_package.py --audit-only
```

The registry contains scalar cells from the confirmatory contrasts, model
metric summaries, measured resource/Pareto table, exploratory subgroup table,
and deterministic qualitative selections. Each value retains its source
file hash, row selector, column, and inferential role.

Within:

```text
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
...
<!-- END_ARTIFACT_BOUND_RESULTS -->
```

standalone numeric literals are rejected. Values must use a token such as:

```text
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEAN_DIFFERENCE|3f}}
```

This contract does not judge writing origin and must not be described as an
AI detector. It ensures that reported numbers are traceable to measured
artifacts and that unsupported affirmative claims remain auditable.
