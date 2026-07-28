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

## Partitions

All partitions are patient-level. The planned provisional counts are 258
training, 37 validation, and 74 internal held-out test subjects. Exact
membership is generated only after Gate 1 data integrity passes.

## Evaluation discipline

- Development uses training and validation only.
- The internal held-out test subset is opened only after statistical-analysis
  configuration and finalist definitions are frozen.
- Primary statistical unit: patient.
- Primary endpoint: patient-wise arithmetic mean of WT, TC, and ET Dice.
- Pixel accuracy is not a primary metric.

## Fairness regimes

- **Compute-matched:** common GPU-hour, optimizer-step, and tuning budgets.
- **Convergence-matched:** model-appropriate training to a predeclared
  convergence/early-stopping rule.

Equal epoch counts are not assumed to be fair.
