# Immutable Scientific Rules

These rules apply to every human or automated contributor in this repository.

1. Never modify, rename, move, or overwrite raw MRI or segmentation files.
2. Resolve raw-data roots only through configuration or the environment
   variables `BRATS2020_ROOT` and `BRATS2019_ROOT`.
3. Never write caches, manifests, predictions, or logs below a raw-data root.
4. Treat BraTS 2020 training as the canonical labeled cohort. Use BraTS 2019
   only for identity and duplicate auditing unless the protocol is explicitly
   revised and documented.
5. Create train, validation, and internal held-out test partitions only at the
   patient level.
6. Never use the internal held-out test subset for training, model selection,
   threshold selection, calibration, or post-processing development.
7. Require `--allow-test-evaluation` for every internal-test evaluation and
   append the access event to `artifacts/test_access_log.jsonl`.
8. Call non-official evaluation data the “internal held-out test subset.”
9. Treat RES and WC as previously published BU-Net components. Never describe
   them as novel contributions of this project.
10. Generate reported metrics, tables, and figures from machine-readable
    experiment artifacts. Do not hand-edit scientific results.
11. Do not import legacy manuscript values or figures without reproducible
    provenance and validation.
12. Do not copy external code before verifying its license and recording its
    provenance.
13. Do not start full training or internal-test evaluation unless the relevant
    protocol, manifests, evaluator tests, and compute environment have passed
    their gates.
14. Use patient-level statistical units; never treat slices or random seeds as
    independent patients.
15. Record data-manifest hashes, split hashes, configuration hashes, code
    commit, environment, seed, hardware, resources, and completion status for
    every reportable run.
