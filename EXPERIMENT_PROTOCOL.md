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
Gate 10 copied all three manifests byte-for-byte into `splits/frozen/` after
the statistical plan and finalist definitions were fixed. The frozen metadata
retains the original hashes and records the clean protocol commit used for the
freeze.

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

The original frozen Gate 7 configs target one `NVIDIA A100-SXM4-80GB`.
Gate 8 was subsequently revised, versioned, and re-preflighted for one
`Apple M1 Max` MPS accelerator before reportable pilot results were observed.
That pilot protocol uses full precision, batch size 16, 2,000 optimizer steps,
and a 0.5 GPU-hour-equivalent wall-time cap. Measurements from different
accelerator protocols are never pooled silently.

## Experiment registry

Every reportable run uses `artifacts/runs/<run_id>/` with resolved config,
metadata, epoch metrics, patient-level validation CSV, checkpoints, resource
profile, and logs. The registry records Git dirty state and commit, config/data/
split hashes, seed, model/loss/optimizer/scheduler, software and hardware,
timestamps, GPU-hours and peak VRAM, parameters and complexity input,
checkpoint selection, completion/failure, error trace, and test-access state.

## Pilot elimination

Gate 8 used 12 unique single-seed development runs rather than the complete
six-architecture by seven-loss grid. Six architecture arms shared CE + soft
Dice; six additional losses were screened on U-Net, with the shared arm reused.
Each arm stopped at 2,000 optimizer steps or 0.5 GPU-hours and performed one
full patient-level validation at step 2,000. Elimination used paired patient
results: both a mean decrement larger than 0.02 and a paired-bootstrap upper
95% bound below zero were required. The complete artifact audit and selected
shortlists are machine-readable in `reports/gate8_artifact_audit.json` and
`reports/gate8_pilot_analysis.json`.

## Multi-seed confirmation

Gate 9 fixed BCE + Focal Tversky and compared the Standard 2D U-Net reference,
BU-Net, residual U-Net, and wide-channel U-Net over three predeclared seeds.
Selection used the arithmetic mean of patient-wise WT, TC, and ET Dice after
averaging each patient's result across seeds. A candidate was eliminated only
when its paired mean decrement exceeded 0.01 and its 10,000-resample paired
bootstrap upper 95% bound was below zero. The reference U-Net remained a
mandatory internal-test comparator and was ineligible for finalist selection.

BU-Net and residual U-Net were selected by the frozen top-two/minimum-two rule
and received two additional predeclared seeds, yielding five seeds per
finalist. BU-Net ranked first at Gate 9, but the paired bootstrap interval
between the finalists included zero; this ranking is not a superiority claim.
The frozen internal-test candidate set is Standard 2D U-Net, BU-Net, and
residual U-Net. Gate 9 used no internal-test data.

## Statistical freeze

The Gate 10 plan is `configs/statistics/gate10.yaml`. It pins Standard 2D U-Net
as the mandatory three-seed reference and pins all five development seeds for
BU-Net and residual U-Net. Each of the 13 best-validation checkpoints is
identified by path, model-config hash, and checkpoint SHA-256 in
`reports/gate10_checkpoint_manifest.json`. Every seed is evaluated separately;
there is no probability, logit, or label ensemble. Candidate metrics first
average the frozen seed values within each patient, preserving the patient as
the inferential unit.

The primary endpoint is the patient's arithmetic mean WT/TC/ET Dice. Three
predeclared paired comparisons share one family: BU-Net versus Standard U-Net,
residual U-Net versus Standard U-Net, and BU-Net versus residual U-Net.
Two-sided paired sign-flip permutation tests use 100,000 resamples and Holm
correction at alpha 0.05. Paired percentile bootstrap intervals use 10,000
patient resamples. Secondary endpoints and grade, ET-presence, and tumor-burden
subgroups are estimation-only. Small/medium/large whole-tumor thresholds were
fixed from training patients alone. Missing and infinite metric values are not
imputed.

Raw evaluation, four-class argmax decoding, 1 mm Surface Dice, HD95, and no
post-processing are frozen. After internal-test access there is no checkpoint
replacement, model selection, threshold adjustment, post-processing tuning, or
unreported candidate/seed exclusion.

## Preprocessing

The versioned preprocessing contract is `configs/data/preprocessing.yaml`,
explained in
`reports/preprocessing_specification.md`. T1, T1ce, T2, and FLAIR are
patient/modality-normalized on nonzero voxels. Validation and internal-test
datasets preserve all slices and use no random augmentation. Any intensity
clipping rule must be fixed from development data before test access. Caches
must remain outside raw-data roots. Gate 8 used an atomically written,
memory-mapped NPY cache for the 258 training and 37 validation subjects; the
conversion report explicitly records that the test manifest was not accessed.
