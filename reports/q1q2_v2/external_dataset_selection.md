# External dataset selection for Q1/Q2 v2

Selection date: 2026-07-30
Decision: **BraTS-Africa TCIA v1 selected**

## Primary cohort

The processed BraTS-Africa release (`10.7937/V8H6-8X67`, CC BY 4.0) is the
primary external cohort because it provides a genuine population/acquisition
shift, the required T1/T1ce/T2/FLAIR modalities, expert tumor-subregion labels,
and zero content overlap with the 369 BraTS 2020 development patients.

The release contains 146 complete patients. The confirmatory manifest is
restricted to the 95 cases in the official `95 Glioma` metadata sheet. The
51 `OtherNeoplasms` cases are not exchangeable with the target glioma
population and are retained only for separately labelled supportive analysis.

Because the primary sample is below 100, a prespecified precision calculation
was completed before external inference. Using the artifact-derived legacy
paired-difference SD (`0.05888`), n=95 gives an expected 95% CI half-width of
about `0.0120` Dice and approximately 90.6% two-sided power for a true 0.020
paired difference. This is planning evidence, not an external result.

## Eligibility checks

| Criterion | Result |
|---|---|
| Official version/license | PASS |
| Four MRI modalities | 146/146 |
| Compatible label semantics | PASS, fixed 3→4 ET mapping |
| Independent institution/population shift | PASS |
| Identifier/content overlap with BraTS 2020 | 0 across 53,874 pairs |
| Primary diagnosis group | 95 glioma |
| External result access before freeze | None |

## Alternatives considered

- **UCSF-PDGM v5** is a large compatible backup, but part of the collection
  entered BraTS 2021 and would need the same content audit.
- **UPenn-GBM v2** is unsuitable wholesale because its descriptor reports 173
  cases previously included in BraTS/FeTS.
- **FeTS/public BraTS 2021** inherits earlier BraTS cases and lacks the open
  protected-test artifacts required for per-patient inference.
- A newer BraTS edition is not independent merely by edition number.

## Frozen use rules

External normalization is the same label-free per-volume nonzero-voxel
normalization used in development. No external thresholding, post-processing,
loss choice, model selection or retraining is permitted. The audit manifest is
locked by SHA-256, and confirmatory inference remains disabled until Gate G.
