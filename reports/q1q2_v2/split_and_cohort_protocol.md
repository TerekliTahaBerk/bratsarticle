# Split and cohort protocol

Status: **frozen before external model inference**

## Development cohort

All 369 unique BraTS 2020 training patients form the development cohort. A deterministic candidate search selected five patient-level stratified folds using grade, ET presence, and WT volume quartile.

| Fold | Training patients | Validation patients |
|---:|---:|---:|
| 1 | 295 | 74 |
| 2 | 295 | 74 |
| 3 | 295 | 74 |
| 4 | 295 | 74 |
| 5 | 296 | 73 |

Every patient is a validation case in exactly one fold. No slices cross patient or fold boundaries. The legacy 258/37/74 partition is not used for v2 development, and its 74-patient internal subset is prohibited for all new-model inference.

## External confirmatory cohort

The primary external manifest contains 95 eligible glioma patients from the processed BraTS-Africa TCIA v1 release. The 51 other-neoplasm cases are excluded from confirmatory inference and may only be used in a separately labelled supportive analysis.

No external model prediction, metric, threshold selection, post-processing choice, or adaptation was performed during cohort design. External inference is allowed once only after the complete model/checkpoint and statistical freeze passes Gate G.

## Preprocessing isolation

The current normalization is per volume over nonzero voxels and does not estimate cohort-wide parameters. Any future learned normalization, sampler, augmentation, threshold, calibration, or post-processing value is fitted using training rows of the current fold only. External labels are never used for adaptation.

## Machine-readable anchors

- Canonical development manifest SHA-256: `b9ab5bdf598521df5b3e26348ef4561f6ace631187385e180841e1712938e78b`
- External inventory SHA-256: `82710419cf7277e0ae1e78ef1b7931e5bc4eef01bf6a2e9b09cea1dd455517ec`
- External test manifest SHA-256: `b714811c97bb28328298598012fc4ba18894d478149148b5f636295c881f1382`
- Selected assignment seed: `20261414`
- Candidate index: `684` of `1000`
