# Gate 1 Completion — Data Integrity and Duplicate Audit

**Status:** PASS

## Verified cohort

- BraTS 2020 training: 369 complete and eligible subjects
- BraTS 2019: 335 complete and eligible subjects
- BraTS 2020 grades: 293 HGG and 76 LGG
- BraTS 2019 grades: 259 HGG and 76 LGG
- Audited NIfTI files: 3,520
- File read/integrity errors: 0
- Unexpected segmentation labels: 0

## Cross-year duplicate result

- The official mapping identifies 335 BraTS 2019 subjects in BraTS 2020.
- BraTS 2020 contains 34 subjects without a BraTS 2019 identity.
- All four MRI modalities are content-equivalent for all 335 mapped pairs.
- 334 mapped pairs have all five files byte-identical.
- One mapped pair contains a two-voxel segmentation annotation revision:
  - BraTS 2019: `BraTS19_CBICA_BLJ_1`
  - BraTS 2020: `BraTS20_Training_128`
  - revision: two voxels changed from background label 0 to ET label 4
- The canonical cohort therefore remains the 369 BraTS 2020 training subjects;
  BraTS 2019 is not an independent added cohort.

## Naming exception

`BraTS20_Training_355` stores its segmentation as
`W39_1998.09.19_Segm.nii`. The audit discovered it through a generic,
reported segmentation fallback. No raw file was renamed or modified.

## Generated artifacts

- `manifests/raw/brats2020_inventory.csv`
- `manifests/raw/brats2019_inventory.csv`
- `manifests/canonical/brats2020_canonical_manifest.csv`
- `manifests/audit/duplicate_mapping.csv`
- `manifests/audit/file_integrity_report.csv`
- `reports/data_audit_summary.md`
- `reports/data_audit_summary.json`

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| BraTS 2020 raw inventory | `b9ab5bdf598521df5b3e26348ef4561f6ace631187385e180841e1712938e78b` |
| BraTS 2019 raw inventory | `9f0aaeb1734253b11d46523cf955127453cdfb342c76e66cf459c229843ccd81` |
| BraTS 2020 canonical manifest | `b9ab5bdf598521df5b3e26348ef4561f6ace631187385e180841e1712938e78b` |
| Duplicate mapping | `00ecd9e197f2cc6df075c1df7041e322b8767b012042f91f8d05f92213df8025` |
| File-integrity report | `0ac14b298dec28a5a98f8006f912331bd20a47c008fb9209837004f7eb550de3` |

## Validation

- Ruff formatting and lint: PASS
- mypy strict type checking: PASS
- pytest: 6 passed
- synthetic audit raw-file mtime preservation: PASS
- real-data two-subject smoke audit: PASS
- full read-only audit: PASS
