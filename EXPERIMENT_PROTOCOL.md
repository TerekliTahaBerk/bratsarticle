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

The Gate 6 feature matrix is versioned in `configs/models/`. Residual blocks,
BU-Net RES skips, and BU-Net WC are independent flags. Equal-width models form
the primary component-cost matrix. A closest-integer-width sensitivity search
uses the controlled U-Net parameter count as its target and a 5% tolerance;
unattainable matches remain reported as failures. Candidate optimization
objectives and all mathematical settings are versioned in
`configs/losses/catalog.yaml`. No loss or architecture is selected from
internal-test performance.

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

The frozen Gate 7 configs target one `NVIDIA A100-SXM4-80GB`. Compute-matched
runs stop at the first of 30,000 optimizer steps or 8.0 GPU-hours.
Convergence-matched runs stop at 50,000 steps or after 12 validation checks
without at least 0.001 improvement, with validation every 500 steps. Both use
one integrated linear-warm-up plus cosine-decay scheduler. If this hardware is
unavailable, the protocol must be revised and versioned before observing pilot
results; measurements from different GPU models may not be pooled silently.

## Experiment registry

Every reportable run uses `artifacts/runs/<run_id>/` with resolved config,
metadata, epoch metrics, patient-level validation CSV, checkpoints, resource
profile, and logs. The registry records Git dirty state and commit, config/data/
split hashes, seed, model/loss/optimizer/scheduler, software and hardware,
timestamps, GPU-hours and peak VRAM, parameters and complexity input,
checkpoint selection, completion/failure, error trace, and test-access state.

## Pilot elimination

Gate 8 uses 12 unique single-seed development runs rather than the complete
six-architecture by seven-loss grid. Six architecture arms share CE + soft
Dice; six additional losses are screened on U-Net, with the shared arm reused.
Each arm stops at 2,000 optimizer steps or 0.5 GPU-hours and validates every
500 steps. Elimination uses paired patient results: both a mean decrement
larger than 0.02 and a paired-bootstrap upper 95% bound below zero are required.
The fallback shortlist is three architectures and two losses. The current host
fails the frozen A100 preflight, so no pilot result or shortlist exists.

## Preprocessing

The versioned preprocessing contract is
`configs/data/preprocessing.yaml`, explained in
`reports/preprocessing_specification.md`. T1, T1ce, T2, and FLAIR are
patient/modality-normalized on nonzero voxels. Validation and internal-test
datasets preserve all slices and use no random augmentation. Any intensity
clipping rule must be fixed from development data before test access. Caches
must remain outside raw-data roots.
