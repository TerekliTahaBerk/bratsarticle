# Q1/Q2 v2 blocker and disposition report

Status date: 2026-07-30
Decision: **NOT READY — COMPUTE BLOCKER REMAINS**

## Resolved initial blockers

### External cohort

Resolved. Official processed BraTS-Africa TCIA v1 data were acquired. All 146
patients passed four-modality/label integrity. The primary external glioma
cohort contains 95 patients; 51 other-neoplasm cases are supportive only.
Zero overlap with 369 BraTS 2020 development patients was established across
53,874 comparisons. Gate C passes.

### Accelerator visibility

Resolved as a detection issue. Outside the application sandbox, PyTorch
2.13.0 reports MPS built and available on the Apple M1 Max. Matrix and Swin
UNETR 64³ forward/backward smoke tests passed. Memory is reported as
MPS framework-reported allocated unified memory, not VRAM.

### Repository license

Resolved under the rights holder's explicit authorization. Original repository
code is Apache-2.0; package metadata, LICENSE, NOTICE and third-party notices
are present. MONAI, nnU-Net v2 and `surface-distance` are Apache-2.0
dependencies and are imported rather than copied.

## Active hard blocker — full experimental allocation

The frozen minimum contains:

| Work package | Runs |
|---|---:|
| 12 models × 5 folds × 5 seeds, convergence matched | 300 |
| 8 component-core models × 5 folds × 5 seeds, compute matched | 200 |
| Three-loss development CV selection | 15 |
| Additional finalist architecture×loss sensitivity | 100 |
| **Total before reproduction reruns** | **615** |

Legacy measured MPS timing gives a median `0.671` seconds per native 2D
optimizer step and `9.33` hours at the 50,000-step ceiling. A real MONAI Swin
UNETR 64³ MPS forward/backward smoke passed in `0.848` seconds; extrapolation
at that smaller-than-frozen patch is `11.78` hours per 50,000 steps.

The known portion totals about `4,032` accelerator-hours or `168` serial days
and still excludes 25 nnU-Net v2 2D and 50 nnU-Net v2 3D/interaction runs.
Only one accelerator is available. This is not a credible bounded execution
for the mandatory design.

Scientific impact: Gates E and F cannot be completed, Gate G cannot freeze
checkpoints/loss, and Gate H external inference is prohibited. No primary
external result or manuscript conclusion exists. A Q1/Q2-ready decision is prohibited.

Required user action: provide a declared CUDA scheduler/cluster allocation
with sufficient parallel GPU-hours and storage. The exact device class,
available GPU count, per-job wall-time, total allocation and launch command are
required. Alternatively, explicitly authorize a protocol revision before any
result is observed; the five folds and common five seeds cannot be silently
reduced.

Affected experiments: all main/convergence runs, core compute-matched runs,
loss interaction, external inference, perturbation analyses, resource profiling,
hierarchical statistics, full-training reproduction, final figures/tables,
manuscript and reviewer response.

## Stop decision

Following the governing prompt, full training stops at this hard blocker.
Unit, synthetic, integrity, model-shape, matching, loss-parity, 2.5D context
and hardware smoke tests are permitted and recorded. The legacy internal test
was not opened, and no external prediction metric was computed.
