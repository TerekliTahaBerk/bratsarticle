# Data Usage and Immutability

## Canonical cohort

The canonical labeled cohort is the BraTS 2020 training dataset. BraTS 2019 is
used only to audit subject identity and duplication with BraTS 2020.

## Raw-data rules

- Raw NIfTI and metadata files are inputs only.
- No command may rename, move, rewrite, compress, decompress, normalize in
  place, or add sidecar files below a raw-data root.
- Paths are supplied through `BRATS2020_ROOT`, `BRATS2019_ROOT`, or explicit
  config overrides.
- Generated manifests store dataset-relative paths rather than workstation
  absolute paths.
- Every generated output and cache path is rejected if it equals or is located
  within a raw-data root.
- The repository ignores `data/`, `*.nii`, and `*.nii.gz`.
- Normalization is computed within each patient and modality. No test-cohort
  statistic is fitted.
- Optional normalized-volume caches require `BRATS_CACHE_ROOT` or another
  explicit external path and are atomically written outside raw data.

## Approved generated locations

- `manifests/`: inventories and canonical metadata
- `reports/`: human-readable audit and analysis reports
- `splits/`: provisional and frozen subject manifests
- `artifacts/`: local run logs, checkpoints, and test-access audit events
- `figures/`: script-generated publication figures

## Data sharing

This repository must not redistribute BraTS images or labels. Public releases
may contain code, dataset-relative manifests, subject identifiers when allowed
by the applicable data agreement, checksums, aggregate statistics, and
instructions for authorized users to reproduce the manifests.
