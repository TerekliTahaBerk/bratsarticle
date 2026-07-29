# Gate 9 Completion

**Decision:** PASS

## Artifact audit

- Valid reportable arms: 16/16
- Validation patients per arm: 37
- Invalid runs: 0
- Duplicate arms: 0
- Internal-test access used: false
- GPU-hour range: 0.328278-0.409094

Diagnostic, Gate 8, and superseded-protocol runs listed as foreign by the audit were excluded from selection.

## Three-seed confirmation

| Candidate | Role | Mean regional Dice | Seed SD | Paired mean difference | 95% bootstrap CI | Eliminated |
|---|---|---:|---:|---:|---:|:---:|
| bunet | finalist_eligible | 0.744574 | 0.007829 | 0.000000 | [0.000000, 0.000000] | no |
| unet_reference | reference | 0.739250 | 0.005927 | -0.005325 | [-0.016131, 0.005195] | no |
| unet_res | finalist_eligible | 0.739415 | 0.007898 | -0.005159 | [-0.021199, 0.005915] | no |
| unet_wc | finalist_eligible | 0.730196 | 0.002279 | -0.014378 | [-0.027917, -0.001091] | yes |

Predeclared finalists: bunet, unet_res

## Five-seed finalist analysis

| Candidate | Seeds | Mean regional Dice | Seed SD | Paired mean difference | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|
| bunet | 5 | 0.742628 | 0.007725 | 0.000000 | [0.000000, 0.000000] |
| unet_res | 5 | 0.739916 | 0.007932 | -0.002712 | [-0.011521, 0.004092] |

Primary finalist by the frozen ranking rule: bunet

Candidates frozen for internal-test evaluation: unet_reference, bunet, unet_res

## Scope

Gate 9 used development-validation data only. The paired confidence interval for the two five-seed finalists includes zero, so the ranking does not establish superiority. Internal-test performance, generalization, clinical applicability, thresholds, and post-processing remain unobserved and unfrozen at this gate.
