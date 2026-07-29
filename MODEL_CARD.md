# Model card

## Intended scope

This repository studies multimodal preoperative glioma segmentation for
methodological comparison. It is research software, not a medical device, and
must not be used for diagnosis, treatment planning, or clinical decisions.

## Version 1

The immutable `v1-bounded-2d-component-study` tag contains bounded internal 2D
component evidence. Its 74-patient internal result is legacy evidence only and
cannot validate new models.

## Version 2

The v2 design covers eight controlled U-Net-family models, nnU-Net v2 2D/3D, a
five-slice 2.5D U-Net, and Swin UNETR. All main models use five patient-level
folds and the same five seeds. Parameter- and compute-matched RES controls are
frozen before training.

No v2 model weights or performance claims are released yet. Full training is
blocked by the recorded compute allocation requirement, so the external
confirmatory set has not been opened for inference.

## Inputs and outputs

Inputs are co-registered T1, T1ce, T2, and FLAIR volumes. The primary output is
a mutually exclusive four-class softmax segmentation mapped to BraTS labels
0/1/2/4. External BraTS-Africa source label 3 maps to internal enhancing-tumor
label 4.

## Limitations

Performance, subgroup behavior, calibration, domain robustness, efficiency,
and clinical utility are unknown for v2 until the complete frozen experiments
finish. A successful smoke test is not evidence of segmentation accuracy.
