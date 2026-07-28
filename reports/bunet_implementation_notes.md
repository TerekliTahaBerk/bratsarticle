# BU-Net Source-Fidelity and Implementation Notes

## Attribution

The residual extended skip (RES) and wide context (WC) components are
reimplementations of previously published BU-Net components. They are not
contributions invented by this project. The primary source is Rehman et al.,
“BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture,”
*Electronics* 9(12), 2203 (2020),
[doi:10.3390/electronics9122203](https://doi.org/10.3390/electronics9122203).
No external implementation code was copied.

## Source-aligned decisions

- The network is two-dimensional and uses same-padded convolutions.
- Convolutions in the feature-extraction blocks are followed by batch
  normalization and ReLU, except for the final classifier.
- The controlled reportable configuration uses dropout probability 0.3.
- Each RES transform has four parallel separable branches. Their kernel lengths
  are 9, 11, 13, and 15; each branch applies `N×1` followed by `1×N`.
- The untransformed skip is added to the four RES branch outputs. The sum is
  refined by `3×3`, `3×3`, and `1×1` convolutional stages before decoder
  concatenation.
- WC is placed at the encoder–decoder transition. It has two separable
  branches, one ordered `N×1 → 1×N` and the other `1×N → N×1`, whose outputs
  are summed. The controlled kernel length is 15.

## Source ambiguity resolved explicitly

The WC prose describes crossed branch ordering, while the published schematic
appears to label two same-orientation operations on each branch. The
implementation follows the prose because it unambiguously states the intended
opposite ordering. This choice is a documented interpretation, not a claim
that the schematic is error-free.

## Deliberate study-specific deviations

- The source diagram uses six sigmoid output filters for its original task.
  This study emits four raw logits for mutually exclusive BraTS 2020 classes:
  background and labels 1, 2, and 4. Cross-entropy candidates use softmax;
  channel-wise BCE candidates use sigmoid against one-hot targets. Inference
  remains an argmax class decision before conversion back to BraTS labels.
- The source-scale channel progression is represented by powers of two, but
  the controlled ablation configurations begin at 16 channels to make pilot
  experiments feasible. Width is configurable and every parameter count is
  reported.
- The implementation applies dropout after encoder blocks before pooling and
  after decoder concatenation before decoding. The source reports dropout 0.3
  but does not specify every insertion point sufficiently for a unique
  reconstruction.
- The Res U-Net comparators use conventional two-convolution residual blocks
  with a learned `1×1` projection when the channel count changes. These blocks
  are controls; they are distinct from the BU-Net RES skip transform.
- Bilinear alignment is used only when an odd input dimension creates a
  one-pixel transposed-convolution mismatch. Reportable 240×240/256×256 inputs
  at depth four do not require this fallback.

## Ablation identity

The feature flags are independent:

| Configuration | Residual blocks | RES skips | WC |
|---|:---:|:---:|:---:|
| U-Net | no | no | no |
| U-Net + RES | no | yes | no |
| U-Net + WC | no | no | yes |
| BU-Net | no | yes | yes |
| Res U-Net | yes | no | no |
| Res U-Net + WC | yes | no | yes |

The equal-width matrix is the primary component-cost comparison. A separate
integer-width search records the closest parameter-matched sensitivity model
and whether it actually meets the predeclared 5% tolerance.
