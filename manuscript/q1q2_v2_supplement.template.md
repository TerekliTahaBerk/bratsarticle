# Supplementary Material

## Capacity- and Compute-Matched Evaluation of Published BU-Net Components

This supplement is generated with the same Gate J claim registry as the main
manuscript and reviewer response. Tables named below are populated from the
declared machine-readable artifacts after Gates G and H. A missing or
hash-mismatched source blocks the final package.

## Supplementary Methods

### S1. Data inventory and identity audit

The data inventory records modality and label paths, byte size, SHA-256,
geometry, orientation, spacing, affine, data type, finite-value status, label
set, and WT, TC, and ET burden for every development patient. BraTS 2019 is
used only for identity auditing. Cross-edition and external comparisons use
identifier, raw-file hash, exact normalized content, sampled content, and
normalized-volume signatures.

**Supplementary Table S1.** Development and external cohort construction,
inclusion, exclusion, modality completeness, label semantics, and missing
metadata.

**Supplementary Figure S1.** Patient identity, edition overlap, and external
cohort flow.

### S2. Patient-level folds

Development folds are deterministic and patient-level. Stratification uses
grade when available, ET presence, and tumor burden. Preprocessing quantities
that require estimation are learned from the training portion of each fold.
The legacy internal subset is unavailable to v2 loaders and evaluators.

**Supplementary Table S2.** Fold-level patient counts and stratification
characteristics.

**Supplementary Figure S2.** Fold balance for grade, ET presence, and tumor
burden.

### S3. Executable model definitions

The component matrix distinguishes BU-Net RES pathways, WC, and conventional
residual blocks. Plain controls are selected independently for parameter and
declared-input compute matching. Official nnU-Net behavior is retained where
required, the controlled 2.5D model predicts the center of a five-slice stack,
and Swin UNETR uses the maintained MONAI implementation.

**Supplementary Table S3.** Layer definitions, tensor shapes, width, depth,
parameters, computation, receptive-field proxy, initialization, and
implementation source for every model.

### S4. Loss equations and output semantics

The loss catalog records logits-to-probability mappings, channel inclusion,
reduction axes, class aggregation, batch aggregation, alpha, beta, gamma,
smoothing, class weights, and empty-class behavior. Four-class inference uses
argmax. Any nested-region correction is a separately configured evaluation
stage.

The development-selected objective is
**{{claim:METHOD.SELECTED_LOSS|raw}}**. Loss selection does not use the legacy
internal subset or external cohort.

**Supplementary Table S4.** Complete executable equations and frozen
hyperparameters for every loss candidate.

### S5. Optimization and convergence

Native models use AdamW with initial learning rate
{{claim:METHOD.INITIAL_LEARNING_RATE_NATIVE_2D|raw}}, weight decay
{{claim:METHOD.WEIGHT_DECAY|raw}}, and effective batch size
{{claim:METHOD.EFFECTIVE_BATCH_SIZE_NATIVE_2D|integer}}. Validation occurs
every {{claim:METHOD.VALIDATION_FREQUENCY_OPTIMIZER_STEPS|integer}} optimizer
steps. Best and terminal checkpoints are both retained.

Early stopping is disabled before
{{claim:METHOD.MINIMUM_OPTIMIZER_STEPS_BEFORE_EARLY_STOPPING|integer}} steps
and then uses minimum delta
{{claim:METHOD.EARLY_STOPPING_MINIMUM_DELTA|raw}} with patience
{{claim:METHOD.EARLY_STOPPING_PATIENCE_CHECKS|integer}} validation checks.
Compute-matched runs use
{{claim:METHOD.COMPUTE_MATCHED_HOURS_PER_RUN|raw}} accelerator hours per run.

**Supplementary Table S5.** Optimizer, schedule, augmentation, convergence,
checkpoint, tuning-opportunity, and compute-budget settings.

**Supplementary Figure S3.** Fold-seed learning curves, best checkpoints,
terminal checkpoints, and ranking stability over the frozen budget
checkpoints.

### S6. Central evaluator

WT, TC, and ET are derived from the four-class label map. Regional overlap,
boundary, surface, rate, and volume endpoints are accompanied by lesion
detection and lesion-wise endpoints. Complete validation and external volumes
are retained, including empty slices. One-empty HD95 is positive infinity and
is summarized by its occurrence rate plus finite distribution summaries.

**Supplementary Table S6.** Metric definitions, aggregation levels,
empty-mask rules, connectivity, lesion matching, minimum-volume settings, and
sensitivity variants.

### S7. Statistical plan

The confirmatory estimand is the paired external patient-level mean of
regional Dice for U-Net+RES versus the parameter-matched plain U-Net.
Uncertainty includes paired patient bootstrap and hierarchical seed-patient
resampling. The paired sign-flip family uses Holm correction. The practical
threshold is an interpretation rule, not a clinical minimal important
difference or equivalence margin.

Bootstrap confidence level is
{{claim:METHOD.CONFIDENCE_LEVEL|percent1}} with
{{claim:METHOD.PAIRED_BOOTSTRAP_RESAMPLES|integer}} paired resamples. The
sign-flip analysis uses
{{claim:METHOD.PAIRED_PERMUTATION_RESAMPLES|integer}} resamples and family
alpha {{claim:METHOD.MULTIPLICITY_ALPHA|raw}}.

**Supplementary Table S7.** Prespecified endpoints, contrasts, multiplicity
family, missingness handling, uncertainty methods, and interpretation rules.

## Supplementary Results

<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The complete development design contains
{{claim:DESIGN.DEVELOPMENT_PATIENTS|integer}} patients,
{{claim:DESIGN.FOLDS|integer}} folds,
{{claim:DESIGN.SEEDS|integer}} common seeds,
{{claim:DESIGN.MODELS|integer}} models, and
{{claim:DESIGN.DEVELOPMENT_RUNS|integer}} development runs. The external
confirmatory and supportive cohorts contain
{{claim:DESIGN.EXTERNAL_CONFIRMATORY_PATIENTS|integer}} and
{{claim:DESIGN.EXTERNAL_SUPPORTIVE_PATIENTS|integer}} patients, respectively.
<!-- END_ARTIFACT_BOUND_RESULTS -->

### S8. Development cross-validation and loss interaction

**Supplementary Table S8.** Fold-seed development estimates, convergence
status, best/terminal checkpoints, budget sensitivity, and failed-run
disposition for all models.

**Supplementary Table S9.** Development-only loss screen and
architecture-by-loss interaction sensitivity.

**Supplementary Figure S4.** Short-, medium-, and converged-budget ranking
stability.

### S9. External model estimates

**Supplementary Table S10.** External confirmatory patient-level WT, TC, ET,
and mean regional Dice for every model with uncertainty.

**Supplementary Table S11.** External regional HD95, surface Dice,
sensitivity, precision, specificity, and relative volume error.

**Supplementary Table S12.** External lesion recall, lesion precision,
lesion-wise Dice, lesion-wise HD95, false-positive lesion count, and
nonfinite-distance rates.

**Supplementary Figure S5.** Regional and lesion-level endpoint distributions
with complete patient denominators.

### S10. Confirmatory contrasts and hierarchical uncertainty

<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The primary contrast includes
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_PATIENT_COUNT|integer}}
paired patients. The paired mean difference is
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEAN_DIFFERENCE|3f}},
with interval
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_LOWER_95|3f}}
to
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_UPPER_95|3f}}
and Holm-adjusted p value
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.HOLM_ADJUSTED_P|pvalue}}.
{{claim:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT|raw}}
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Supplementary Table S13.** All prespecified contrasts with paired mean and
median differences, intervals, standardized effects, raw and adjusted p
values, probability of superiority, practical-threshold status, and exact
interpretation.

**Supplementary Table S14.** Hierarchical bootstrap and mixed-effects
sensitivity results.

**Supplementary Figure S6.** Patient and seed components of uncertainty for
the primary and secondary contrasts.

### S11. Evaluation sensitivities

**Supplementary Table S15.** Minimum-lesion-volume, connectivity, and HD95 cap
sensitivity analyses.

**Supplementary Figure S7.** Finite and infinite HD95 behavior by model and
region.

### S12. External subgroups and domain shift

Subgroup analyses are exploratory. Institution, scanner, field strength,
grade when available, ET presence, development-derived tumor burden, and
resolution cells report explicit denominators. Small cells remain descriptive
and do not receive confirmatory language.

**Supplementary Table S16.** Model estimates by external subgroup.

**Supplementary Table S17.** Primary contrast estimates by external subgroup.

**Supplementary Figure S8.** External subgroup forest plots with small-cell
labels.

### S13. Resource realization

<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
U-Net+RES has {{claim:RESOURCE.UNET_RES.PARAMETER_COUNT|integer}} parameters,
{{claim:RESOURCE.UNET_RES.FLOPS_PER_DECLARED_INPUT|integer}} FLOPs per declared
input, mean training time
{{claim:RESOURCE.UNET_RES.TRAINING_ACCELERATOR_HOURS_MEAN|2f}} accelerator
hours per fold-seed run, and median external end-to-end latency
{{claim:RESOURCE.UNET_RES.INFERENCE_END_TO_END_P50_SECONDS|3f}} seconds per
volume. The parameter-matched control has
{{claim:RESOURCE.UNET_PARAMETER_MATCHED_RES.PARAMETER_COUNT|integer}}
parameters and
{{claim:RESOURCE.UNET_PARAMETER_MATCHED_RES.FLOPS_PER_DECLARED_INPUT|integer}}
FLOPs per declared input.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Supplementary Table S18.** Static graph, training, checkpoint, memory,
inference-stage, latency, and throughput measurements for every model.

**Supplementary Table S19.** Parameter- and compute-matching accuracy and
realized compute-budget adherence.

**Supplementary Figure S9.** Separate accuracy-cost and all-measured-cost
non-dominance views.

### S14. Qualitative selection and panels

Cases are selected after evaluation using prespecified deterministic rules.
The rules cover highest performance, cohort median, lowest finite ET
lesion-wise Dice, greatest false-positive lesion burden, greatest regional
HD95, and greatest model disagreement.

**Supplementary Table S20.** Rule, anonymized patient identifier, slice,
tie-break inputs, selected metric, and source hash for each qualitative case.

**Supplementary Figures S10–S15.** Multimodal images, reference labels,
reference lesion components, all model predictions, and false-positive/
false-negative overlays for each selected role.

### S15. Reproducibility and deviations

**Supplementary Table S21.** Software, hardware, environment, data, split,
configuration, checkpoint, prediction, analysis, figure, and claim-provenance
identifiers.

**Supplementary Table S22.** All failed runs, deviations, missing values, and
their prespecified disposition.

**Supplementary Table S23.** CLAIM checklist with final manuscript page and
line references.

No supplementary item changes the frozen primary endpoint, contrast,
multiplicity family, or external-session rules.
