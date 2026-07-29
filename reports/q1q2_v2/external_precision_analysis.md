# External confirmatory precision and power plan

This calculation was completed before model inference on the external cohort.
It is a planning analysis, not a result.

- Eligible primary external glioma patients: 95
- Historical planning SD of paired patient differences: 0.058880
- Expected two-sided 95% CI half-width under that SD: 0.0120 Dice
- Prespecified practical-effect threshold: 0.020 mean-regional Dice.
- Rationale: a two-percentage-point average across WT, TC and ET is large enough to require a distributed regional gain rather than a rounding-level change. It is an interpretation threshold, not a clinical MCID or an equivalence/non-inferiority margin.

| True paired mean difference | Approximate two-sided power |
|---:|---:|
| 0.010 | 0.374 |
| 0.015 | 0.691 |
| 0.020 | 0.906 |
| 0.025 | 0.984 |
| 0.030 | 0.998 |

The planning SD is artifact-derived from the legacy internal patient-level U-Net+RES versus U-Net contrast after seed averaging. Because the new parameter-matched comparator and African cohort may have a different variance, final inference will emphasize the observed paired confidence interval.
