# bratsarticle

Reproducible research pipeline for leakage-safe, patient-level evaluation of
U-Net-family models on multimodal BraTS glioma segmentation data.

## Current status

Gate 0 (repository and environment audit) is complete. No training results,
model comparisons, or internal held-out test results are currently claimed by
this repository.

The initial audit is available at
[`reports/phase0_repository_audit.md`](reports/phase0_repository_audit.md).

## Scientific scope

The core study will:

- treat BraTS 2020 training subjects as the canonical labeled cohort;
- use BraTS 2019 only for identity and duplicate auditing;
- enforce patient-level train, validation, and internal held-out test splits;
- evaluate RES and WC as previously published BU-Net components, not as novel
  modules;
- generate metrics, tables, and figures from machine-readable experiment
  artifacts;
- prevent test-set use during training, model selection, threshold selection,
  and post-processing development.

## Data safety

Raw MRI and segmentation files are not versioned and must never be modified,
renamed, or written in place. Dataset roots will be supplied through
environment variables:

```text
BRATS2020_ROOT
BRATS2019_ROOT
BRATS_CACHE_ROOT
```

The `data/` directory and medical-image files are excluded from Git.

## Development

The implementation stack and reproducible setup commands will be added after
the Gate 1 environment and data-integrity design is approved. Full training
and internal held-out test evaluation require separate explicit approval.
