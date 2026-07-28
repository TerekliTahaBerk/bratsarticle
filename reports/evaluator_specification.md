# Central Evaluator Specification

**Gate:** 3  
**Status:** PASS  
**Configuration:** `configs/evaluation/default.yaml`

## Evaluation unit and regions

The statistical unit is one patient volume. Slice-wise values are never
treated as independent observations. Predictions are converted to the three
standard BraTS regions:

- **WT:** labels 1, 2, and 4
- **TC:** labels 1 and 4
- **ET:** label 4

The primary endpoint is the unweighted arithmetic mean of the patient's WT,
TC, and ET Dice values. Pixel accuracy is intentionally absent from the
primary metrics.

The original BraTS benchmark used Dice and a robust 95th-percentile Hausdorff
measure to combine overlap and boundary assessment
([Menze et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4833122/)).

## Supported prediction encodings

1. Integer BraTS labels `{0, 1, 2, 4}`.
2. Four-channel softmax/logit output in channel order
   `[background, label 1, label 2, label 4]`; decoding uses `argmax`.
3. Three-channel sigmoid/logit output in channel order `[WT, TC, ET]`, with
   independently configured thresholds.

For nested sigmoid output, `ET ⊆ TC ⊆ WT` consistency is **off by default**.
When enabled, the sole supported rule is `outward_union`: ET positives are
retained and added to TC; the resulting TC is added to WT. The number of
pre-correction violation voxels is emitted per patient. Consistency correction
is therefore a named, configurable ablation rather than hidden
post-processing.

## Voxel-wise metrics

Every region receives:

- Dice and IoU;
- sensitivity, precision, and specificity;
- physical-space HD95 in millimetres;
- area-weighted Surface Dice at the configured tolerance;
- signed, absolute, and relative volume error;
- predicted and reference voxel counts.

No smoothing constant is used for reported Dice or IoU. For non-empty masks,
`Dice = 2TP / (2TP + FP + FN)` and `IoU = TP / (TP + FP + FN)`.

HD95 is the maximum of the two directed, surfel-area-weighted 95th-percentile
surface distances. Surface Dice is also surfel-area weighted and uses physical
voxel spacing. Both are calculated with Google DeepMind's Apache-2.0
`surface-distance` 0.1 reference implementation
([source and metric description](https://github.com/google-deepmind/surface-distance)).
The Surface Dice tolerance is a declared secondary-analysis parameter; its
provisional default is 1.0 mm and it must be frozen before internal-test use.
Surface Dice was introduced to measure tolerance-based surface agreement
([Nikolov et al.](https://arxiv.org/abs/1809.04430)).

## Empty-mask policy

| Situation | Dice / IoU | Surface Dice | HD95 | Sensitivity / precision when denominator is zero |
|---|---:|---:|---:|---:|
| Both masks empty | 1 | 1 | 0 mm | NaN |
| Exactly one mask empty | 0 | 0 | Infinity | Formula if defined, otherwise NaN |

Relative volume error is zero when both masks are empty and positive infinity
when the reference is empty but the prediction is not. Summary generation
reports finite, NaN, and infinite counts separately; it never silently drops
infinite values.

## Lesion-wise evaluation

The default provisional lesion policy is:

- 26-connected 3D connected components;
- minimum lesion size of one voxel and 0 mm³, meaning no size removal;
- one-to-one maximum-total-IoU assignment;
- a match requires positive overlap and IoU at least the configured threshold
  (default 0.0);
- unmatched reference lesions are false negatives;
- unmatched prediction lesions are false positives.

The assignment is solved with SciPy's rectangular linear assignment
implementation in maximum-weight mode
([SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)).
This definition makes split and merge errors observable: a one-to-many split
can produce at most one match and leaves extra predicted components as false
positives; a many-to-one merge can match at most one reference component and
leaves the others as false negatives.

Lesion recall is undefined (NaN) when the reference contains no eligible
lesion. Lesion precision is undefined when the prediction contains no eligible
lesion. False-positive lesion count remains defined in all cases. Lesion-wise
Dice and HD95 are reference-lesion anchored: unmatched reference lesions
receive Dice 0 and HD95 infinity. When no reference lesion exists, both
lesion-wise overlap and distance are NaN.

BraTS lesion-wise evaluation has used 26-connectivity and reports lesion-wise
Dice and HD95, supporting the inclusion of explicit component-aware endpoints
([BraTS-METS challenge description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312806/)).

## Post-processing visibility

The evaluator can emit separate `raw` and `filtered` rows for the same patient.
Filtering removes prediction components below declared voxel and physical
volume thresholds. The default configuration emits only `raw`; adding
`filtered` requires an explicitly versioned ablation configuration. Raw and
filtered values are never merged or overwritten.

## Synthetic verification

The test suite covers:

- perfect prediction;
- completely empty prediction;
- empty ET reference;
- false-positive ET lesion;
- one-to-many lesion split;
- many-to-one lesion merge;
- nested-region violation with correction off and on;
- Dice–IoU mathematical consistency;
- axis/order shape error;
- batch-size invariance;
- NumPy/PyTorch CPU consistency;
- CPU/GPU consistency when CUDA is available;
- deterministic repeated output;
- explicit raw-versus-filtered post-processing;
- four-class channel mapping and evaluator-config loading.

On the current Apple M1 Max host, 27 repository tests pass and the CUDA-only
consistency test is skipped because CUDA hardware is unavailable. The test
remains active and will run automatically in a CUDA environment.

## Scope boundary

This gate validates metric definitions and behavior. It does not produce model
performance claims, select thresholds, access the real internal-test loader,
or justify clinical applicability.

