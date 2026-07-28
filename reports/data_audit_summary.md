# BraTS Data Audit Summary

**Run scope:** `full`

## Cohort inventory

| Dataset | Subjects | Complete | Eligible | Grade counts |
|---|---:|---:|---:|---|
| brats2020 | 369 | 369 | 369 | `{"HGG":293,"LGG":76}` |
| brats2019 | 335 | 335 | 335 | `{"HGG":259,"LGG":76}` |

## Cross-year mapping

- Mapping rows: 369
- Mapped overlaps: 335
- New BraTS 2020 subjects: 34
- Mapped pairs with all five exact file hashes equal: 334
- Mapped pairs with all five voxel contents equivalent: 334
- Mapped pairs with all four MRI modalities equivalent: 335
- Mapped pairs with a segmentation annotation revision: 1

## File integrity

- Audited NIfTI files: 3520
- File errors: 0
- Naming exceptions: 1
- Subjects with invalid label sets: 0

## Gate checks

| Check | Result |
|---|---|
| `brats2020_subject_count_is_369` | PASS |
| `brats2019_subject_count_is_335` | PASS |
| `mapped_overlap_count_is_335` | PASS |
| `new_brats2020_subject_count_is_34` | PASS |
| `all_brats2020_subjects_complete` | PASS |
| `all_brats2019_subjects_complete` | PASS |
| `no_file_integrity_errors` | PASS |
| `all_segmentation_label_sets_valid` | PASS |
| `all_mapped_image_modalities_content_equivalent` | PASS |

**Gate 1 integrity status:** PASS
