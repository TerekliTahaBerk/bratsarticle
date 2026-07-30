# Frozen statistical execution

`configs/q1q2_v2/statistical_execution.yaml` turns the prespecified analysis
plan into an artifact-only executable stage. It cannot run until Gate G has
frozen all 600 development runs and Gate H has passed for all 300 main
checkpoints.

The analysis never opens MRI volumes or the legacy 74-patient internal subset.
It verifies the Gate G/H hashes, reconstructs development cross-validation
metrics from each best checkpoint, and keeps the 95-patient confirmatory glioma
cohort separate from the 51-patient supportive other-neoplasm cohort.

The best 2D model used in the fourth secondary contrast is selected only from
development cross-validation. The frozen tie breakers are lower mean
accelerator-hours and then lexicographic model ID. External outcomes do not
participate in this selection.

For each of the five confirmatory contrasts, the stage produces:

- paired patient mean and median differences;
- a 10,000-resample paired patient bootstrap interval;
- a 100,000-resample paired sign-flip p value;
- Holm correction across the complete five-contrast family;
- standardized paired effect and probability of superiority;
- a seed-then-patient hierarchical bootstrap;
- a fold-resampling sensitivity interval; and
- a crossed mixed-effects sensitivity with model and fold fixed effects plus
  patient and training-seed random intercepts.

All regional, boundary, volume, and lesion metrics receive finite/non-finite
descriptive summaries. Infinity-valued HD95 observations remain infinity and
are counted explicitly; they are never silently imputed.

After Gate H passes, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_q1q2_statistics.py
```

The command requires `statsmodels` for the frozen mixed-effects sensitivity.
Any missing replicate, altered source hash, non-finite primary Dice value, or
non-converged mixed model blocks completion rather than producing a partial
confirmatory claim.
