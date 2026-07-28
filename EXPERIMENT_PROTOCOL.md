# Experimental Protocol

## Study identity

Patient-level, leakage-safe, duplicate-aware, and compute-controlled evaluation
of U-Net-family models for multimodal glioma segmentation.

## Core models

1. Standard 2D U-Net
2. BU-Net reimplementation
3. Standard Res U-Net
4. Res U-Net + WC
5. nnU-Net 2D
6. nnU-Net 3D when hardware permits
7. At most one transformer or hybrid baseline

RES and WC are attributed to BU-Net and are evaluated as baseline/ablation
components.

The Standard 2D U-Net is implemented first and isolated from all BU-Net,
residual, WC, and transformer components. Its versioned architecture and
optimization settings are `configs/training/unet2d_baseline.yaml`. Full
training requires an explicit flag and a CUDA host; bounded diagnostics do not
authorize test access.

## Partitions

All partitions are patient-level. The provisional split contains 258 training,
37 validation, and 74 internal held-out test subjects. It was deterministically
selected from 256 stratified candidates using seed `20260729`. Exact membership
and SHA-256 hashes are recorded in `splits/provisional/split_metadata.json`.
The split remains provisional until the statistical plan and finalist
definitions are frozen.

## Evaluation discipline

- Development uses training and validation only.
- The development loader cannot open the internal-test manifest.
- The internal held-out test subset is opened only after statistical-analysis
  configuration and finalist definitions are frozen.
- Every authorized internal-test access requires an explicit flag, purpose, and
  append-only event in `artifacts/test_access_log.jsonl`.
- Primary statistical unit: patient.
- Primary endpoint: patient-wise arithmetic mean of WT, TC, and ET Dice.
- Pixel accuracy is not a primary metric.

The versioned metric contract is `configs/evaluation/default.yaml`, explained
in `reports/evaluator_specification.md`. HD95 and Surface Dice use physical
spacing. Empty-mask, connected-component, matching, and post-processing rules
are explicit. Raw and filtered evaluation stages are stored as separate rows.
Thresholds, lesion-size rules, and Surface Dice tolerance must be frozen before
the internal held-out test is opened.

## Fairness regimes

- **Compute-matched:** common GPU-hour, optimizer-step, and tuning budgets.
- **Convergence-matched:** model-appropriate training to a predeclared
  convergence/early-stopping rule.

Equal epoch counts are not assumed to be fair.

## Preprocessing

The versioned preprocessing contract is
`configs/data/preprocessing.yaml`, explained in
`reports/preprocessing_specification.md`. T1, T1ce, T2, and FLAIR are
patient/modality-normalized on nonzero voxels. Validation and internal-test
datasets preserve all slices and use no random augmentation. Any intensity
clipping rule must be fixed from development data before test access. Caches
must remain outside raw-data roots.
