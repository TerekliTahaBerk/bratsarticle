# Gate 11 Completion

**Decision:** PASS

## Artifact and access audit

- Valid frozen checkpoints: 13/13
- Patients per checkpoint: 74
- Guarded internal-test manifest openings: 1
- Invalid runs: 0

## Internal held-out test estimates

| Candidate | Mean regional Dice | WT Dice | TC Dice | ET Dice |
|---|---:|---:|---:|---:|
| unet_reference | 0.735529 | 0.809625 | 0.692731 | 0.704233 |
| bunet | 0.752255 | 0.823213 | 0.724234 | 0.709319 |
| unet_res | 0.755986 | 0.827370 | 0.725657 | 0.714932 |

## Frozen primary-endpoint comparisons

| Comparison | Mean difference | 95% paired bootstrap CI | Raw p | Holm p | Reject |
|---|---:|---:|---:|---:|:---:|
| bunet_vs_unet_reference | 0.016726 | [0.005107, 0.029525] | 0.007130 | 0.014260 | yes |
| unet_res_vs_unet_reference | 0.020457 | [0.008093, 0.034364] | 0.001500 | 0.004500 | yes |
| bunet_vs_unet_res | -0.003731 | [-0.007581, -0.000198] | 0.049530 | 0.049530 | yes |

## Scope

All frozen candidates and seeds are reported. Secondary endpoints and subgroups are estimation-only. These results are from one internal held-out subset on a single dataset and do not establish external generalization or clinical applicability.

Predeclared qualitative cases: success=BraTS20_Training_166, hard=BraTS20_Training_137, failure=BraTS20_Training_323
