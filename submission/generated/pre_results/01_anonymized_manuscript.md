> **PRE-RESULTS PREVIEW - NOT FOR SUBMISSION. Scientific result fields and author declarations remain unresolved. Do not upload this file to a journal.**

# Capacity- and Compute-Matched Evaluation of Published BU-Net Components with Multi-Seed Development and Independent External Testing for Multimodal Glioma Segmentation

**Article type:** Original Research

**Summary statement:** [PENDING:SUMMARY_STATEMENT_FROM_FINAL_RESULT]

**Key Points**

1. [PENDING:KEY_POINT_PRIMARY_CAPACITY_CONTROLLED_EFFECT]
2. [PENDING:KEY_POINT_EXTERNAL_FAILURE_OR_DOMAIN_SHIFT]
3. [PENDING:KEY_POINT_RESOURCE_TRADEOFF]

## Abstract

### Purpose

To test whether the published Residual Extended Skip (RES) component retains
a measurable benefit after parameter matching and to characterize uncertainty,
lesion-level failure, external domain shift, and measured resource cost.

### Materials and Methods

This retrospective public-data study used BraTS 2020 for development and
BraTS-Africa for independent external testing. Twelve models were assigned the
same five seeds in five patient-level folds. After development-only model,
loss, checkpoint, endpoint, and analysis freeze, the confirmatory external
cohort was evaluated once. The primary endpoint was patient-level mean whole
tumor, tumor core, and enhancing tumor Dice. The primary contrast compared
U-Net+RES with a parameter-matched plain U-Net using paired bootstrap
intervals, paired sign-flip tests with Holm correction, and hierarchical
seed-patient resampling.

### Results

<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The external confirmatory analysis included
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_PATIENT_COUNT]
paired patients. Mean regional Dice was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.FIRST_MEAN]
for U-Net+RES and
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.SECOND_MEAN]
for the parameter-matched plain U-Net. The paired mean difference was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEAN_DIFFERENCE]
(95% interval,
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_LOWER_95]
to
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_UPPER_95]);
the Holm-adjusted P value was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.HOLM_ADJUSTED_P].
<!-- END_ARTIFACT_BOUND_RESULTS -->

### Conclusion

The result provides capacity-controlled external evidence about a published
BU-Net component and is bounded to the frozen cohorts, models, endpoints, and
practical interpretation threshold.


## Introduction

Glioma segmentation in multimodal magnetic resonance imaging supports
quantitative description of whole tumor (WT), tumor core (TC), and enhancing
tumor (ET), but it is also a setting in which evaluation design can dominate
an apparent architectural improvement. The BraTS benchmark formalized
multimodal subregion segmentation and complementary overlap and boundary
metrics [1]. U-Net established an encoder-decoder reference with lateral skip
connections [2], followed by residual, contextual, self-configuring, and
transformer-based variants.

RES and WC are not new components of the present work. Rehman et al.
introduced both in BU-Net: RES transforms information along extended skip
pathways, whereas WC uses asymmetric contextual convolutions [3]. Conventional
residual blocks have a separate origin in residual learning [4]. Conflating
these elements, or comparing a wider component model only with a smaller plain
U-Net, cannot isolate why performance differs.

Several additional sources of bias matter. Training stochasticity can change
medical-segmentation rankings [5,6]. A self-configuring system such as nnU-Net
can outperform manually assembled networks through coordinated choices in
preprocessing, training, and inference [7], and controlled reevaluation has
shown why modern architectures require strong, fairly tuned baselines [8].
Volumetric and hybrid systems, including TransBTS, nnFormer, and Swin UNETR,
also make a purely bounded 2D comparison an incomplete account of the field
[9–11].

Metric choice further limits interpretation. Regional Dice does not describe
boundary displacement, missed lesions, false-positive lesions, or empty-mask
behavior. Metrics Reloaded recommends choosing complementary measures from the
task’s problem fingerprint [12]. CLAIM 2024 similarly emphasizes transparent
cohort construction, independent evaluation, and complete reporting in
medical imaging AI [13].

We therefore framed the work as a controlled evaluation rather than a
new-network paper. The primary question was whether RES improves external
patient-level mean regional Dice relative to a plain U-Net matched to its
parameter capacity. Prespecified secondary analyses evaluated the same-width
and compute-matched contrasts, WC and BU-Net configurations, strong baselines,
seed-patient uncertainty, lesion and boundary failure, external subgroups, and
measured cost. Negative findings were retained by design.

## Materials and Methods

### Study design and leakage controls

The study was defined before confirmatory external inference. BraTS 2020
training data formed the complete development cohort. BraTS 2019 was used only
for identity and duplicate auditing and contributed no additional patient.
The legacy internal held-out subset was not reopened for any new model,
threshold, loss, post-processing choice, or confirmatory inference.

All splits were patient-level. The
[PENDING:DESIGN.DEVELOPMENT_PATIENTS] development patients were
assigned deterministically to [PENDING:DESIGN.FOLDS] folds stratified
by grade when available, ET presence, and WT burden. Every patient occurred in
one validation fold and in no other validation fold. Slice-level random
splitting was prohibited.

### Development and external cohorts

For each BraTS 2020 patient, the audit recorded identifiers, four modality and
segmentation paths, file hashes, NIfTI geometry, affine/orientation, data type,
finite-value checks, label set, and regional tumor burden. The expected label
set was background, necrotic/non-enhancing tumor, edema, and enhancing tumor.
Raw files were read-only and all caches were written outside dataset roots.

The independent cohort was the processed BraTS-Africa TCIA release [14].
Before model inference, each external patient was checked for four modalities,
readable labels, label conversion, geometry, and metadata. Identifier, raw
hash, normalized content, sampled content, and normalized-volume signatures
were compared against all development patients. The primary external cohort
contained only glioma; other neoplasms were excluded from confirmatory
inference and retained as supportive evidence.

### Preprocessing

The input order was T1, contrast-enhanced T1, T2, and FLAIR. Intensities were
z-normalized independently by modality over nonzero brain voxels. Spatial
augmentation used one transform for all modalities and the segmentation;
intensity augmentation was modality-specific. Training sampling could enrich
tumor-containing slices or patches, but validation and external evaluation
retained the complete volume, including empty slices, so false-positive
behavior remained measurable.

Four-class outputs used softmax with the BraTS label mapping. Region metrics
were derived as WT = labels {1,2,4}, TC = labels {1,4}, and ET = label {4}.
Any nested-region consistency operation was an explicit evaluation stage and
was never hidden inside metric computation.

### Models and component attribution

The frozen matrix included U-Net-Small; plain U-Nets matched to U-Net+RES by
parameters and by declared-input compute; U-Net+RES; U-Net+WC; BU-Net;
ResBlock-U-Net; ResBlock-U-Net+WC; official nnU-Net v2 2D; an official
hardware-feasible nnU-Net v2 3D full-resolution plan; a five-slice 2.5D U-Net;
and MONAI Swin UNETR.

RES and WC followed BU-Net attribution [3]. The ResBlock models used
conventional residual mappings sourced to residual learning [4]; they were
controls, not novel architectures. The parameter-matched and compute-matched
plain U-Nets were chosen by deterministic width/depth search before outcomes.
Parameter difference and compute difference were treated as separate
constraints rather than collapsed into a subjective score.

Official nnU-Net v2 planning, preprocessing, folds, and trainers were retained
where compatible with the frozen data contract [7,8]. The 2.5D comparator
stacked five consecutive slices in modality-major order and predicted the
center slice, with boundary replication [15]. Swin UNETR used the published
hierarchical Swin encoder through the maintained MONAI implementation [11].
No literature score was entered as an experimental observation.

### Loss selection and optimization

Loss was selected using development folds only. The candidates were
cross-entropy plus soft Dice, binary cross-entropy plus focal Tversky, and
cross-entropy plus focal Tversky. Focal Tversky was attributed to Abraham and
Khan [16]. The selected objective was
**[PENDING:METHOD.SELECTED_LOSS]**. Its executable alpha, beta, gamma,
smoothing, channel activation, and background inclusion were stored in the
versioned loss catalog and resolved run configuration.

Native 2D models used AdamW with initial learning rate
[PENDING:METHOD.INITIAL_LEARNING_RATE_NATIVE_2D], weight decay
[PENDING:METHOD.WEIGHT_DECAY], and effective batch size
[PENDING:METHOD.EFFECTIVE_BATCH_SIZE_NATIVE_2D]. One warmup-cosine
schedule was used; no second scheduler modified the same learning rate.
Validation occurred every
[PENDING:METHOD.VALIDATION_FREQUENCY_OPTIMIZER_STEPS] optimizer steps.

Convergence runs allowed at most
[PENDING:METHOD.MAXIMUM_OPTIMIZER_STEPS] optimizer steps. Early
stopping was disabled before
[PENDING:METHOD.MINIMUM_OPTIMIZER_STEPS_BEFORE_EARLY_STOPPING] steps
and then required a minimum patient-level mean regional Dice improvement of
[PENDING:METHOD.EARLY_STOPPING_MINIMUM_DELTA] within
[PENDING:METHOD.EARLY_STOPPING_PATIENCE_CHECKS] validation checks.
Best and terminal checkpoints were both retained. Compute-matched core runs
stopped at a synchronized accelerator budget of
[PENDING:METHOD.COMPUTE_MATCHED_HOURS_PER_RUN] hours per run rather than at
an equal epoch count.

Every main model used the same [PENDING:DESIGN.SEEDS] seeds in every
fold. Failed runs were reported and could not be silently replaced. Model
selection, loss selection, and convergence decisions used development
artifacts only.

### Central evaluation

One evaluator decoded native, MONAI, and nnU-Net predictions and produced
patient-level metrics. The primary endpoint was the arithmetic mean of WT, TC,
and ET Dice for each patient. Secondary endpoints included regional HD95,
surface Dice, sensitivity, precision, specificity, relative volume error,
lesion recall, lesion precision, lesion-wise Dice, lesion-wise HD95, and
false-positive lesion count. Pixel accuracy was not a primary endpoint.

Lesions were three-dimensional connected components under frozen connectivity
and minimum-volume settings. Ground-truth and predicted lesions were paired by
a prespecified one-to-one rule. One-empty HD95 remained positive infinity;
reports separated its rate from finite medians and interquartile ranges.
Evaluation-sensitivity settings changed only connectivity, minimum lesion
volume, and reporting caps and did not replace the primary definition.

### External evaluation

Gate G required all [PENDING:DESIGN.DEVELOPMENT_RUNS] development runs,
required checkpoints, complete validation metrics, frozen loss and nnU-Net
plan, environment hashes, and the prespecified statistical inputs. Only its
passing freeze authorized Gate H.

Gate H was one external session. It evaluated
[PENDING:DESIGN.EXTERNAL_MAIN_CHECKPOINTS] frozen main checkpoints on
all external patients without retraining, threshold selection,
post-processing selection, or label-informed normalization adaptation.
Checkpoint predictions were aggregated within each model by a prespecified
nested-region strict majority vote. All checkpoint failures were retained and
blocked a passing external gate.

### Statistical analysis

The primary contrast was U-Net+RES minus the parameter-matched plain U-Net for
external patient-level mean regional Dice. The paired patient bootstrap used
[PENDING:METHOD.PAIRED_BOOTSTRAP_RESAMPLES] resamples and confidence
level [PENDING:METHOD.CONFIDENCE_LEVEL]. A two-sided paired sign-flip
test used [PENDING:METHOD.PAIRED_PERMUTATION_RESAMPLES] Monte Carlo
resamples. The primary and four prespecified secondary contrasts formed one
Holm family with alpha [PENDING:METHOD.MULTIPLICITY_ALPHA].

The practical interpretation threshold was
[PENDING:METHOD.PRACTICAL_THRESHOLD] mean regional Dice. It was not called
a clinical minimal important difference and was not an equivalence or
noninferiority margin. Effect reporting included the paired mean and median
difference, percentile interval, standardized paired effect, adjusted and raw
p values, and probability of superiority.

Hierarchical uncertainty resampled training seeds and then patients; folds
were averaged in the primary hierarchy and resampled in sensitivity analysis.
A mixed-effects model with model and fold fixed effects and patient and seed
random intercepts was sensitivity analysis only. External institution,
scanner, field strength, grade when available, ET presence, development-based
tumor burden, and resolution subgroups were exploratory and reported with
explicit denominators and no confirmatory interpretation.

### Resource analysis

Parameters, FLOPs, MAC-equivalents, declared input shapes, activation sizes,
and a receptive-field proxy were generated from executable model graphs.
Training time used synchronized optimizer-step measurements after a frozen
warmup. Inference timing separated preprocessing, model forward,
post-processing, and end-to-end latency for each patient volume. On Apple MPS,
memory was reported as framework-allocated and driver-allocated unified
memory, not VRAM.

Accuracy was plotted separately against each cost dimension and in an
all-measured-cost non-dominance view. No subjective efficiency score was
constructed, and “efficient” was not inferred from parameter count alone.

### Qualitative analysis

Case identities were selected after evaluation by six frozen rules: highest
patient-average regional Dice, cohort median, lowest finite ET lesion-wise
Dice, largest false-positive lesion burden, largest regional HD95, and largest
pairwise model disagreement. Ties used larger reference WT volume and then
anonymized patient identifier. The language “prespecified cases” was
prohibited; only the selection rules were prespecified.

Each panel displayed four modalities, reference labels, reference WT
components, predictions from all models, and WT false-positive/false-negative
overlays. The axial slice had the largest reference WT area, with the lowest
slice index breaking ties. This analysis performed no new inference.

### Reproducibility and claim provenance

Each run stored the Git commit, resolved configuration and hash, fold and data
hashes, environment and hardware records, random seed, optimizer/scheduler
state, best and terminal checkpoint hashes, patient metrics, resource profile,
and explicit failure status. Interrupted native MPS runs restored Python,
NumPy, and PyTorch CPU RNG state; bitwise claims were restricted to tested
contexts.

Gate I rehashed the complete development, external, analysis, table, and
figure bundle and reran code checks in a data-free clean clone. Numerical
analyses and standard result figures had to regenerate byte-identically.
Qualitative panels were hash-verified; their rerender required the ignored
derived image cache and was identified as a separate scope boundary. This
artifact audit was not described as full retraining.

Gate J generated manuscript values from a machine-readable claim registry.
Every substituted value retained its source file hash, row selector, column,
format, registry hash, and rendered-document hash. Manual numeric result entry
inside artifact-bound Results blocks was prohibited.

## Results

<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
### Cohort and execution integrity

All [PENDING:DESIGN.DEVELOPMENT_PATIENTS] development patients were
assigned to exactly one validation fold. The external confirmatory analysis
contained [PENDING:DESIGN.EXTERNAL_CONFIRMATORY_PATIENTS] glioma
patients, and the supportive analysis contained
[PENDING:DESIGN.EXTERNAL_SUPPORTIVE_PATIENTS] other-neoplasm patients.
The complete main external matrix comprised
[PENDING:DESIGN.EXTERNAL_MAIN_CHECKPOINTS] checkpoint evaluations.

### Primary capacity-controlled comparison

U-Net+RES had mean external patient-level regional Dice
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.FIRST_MEAN],
compared with
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.SECOND_MEAN]
for the parameter-matched plain U-Net. The paired mean difference was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEAN_DIFFERENCE];
the median difference was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEDIAN_DIFFERENCE].
The paired percentile interval extended from
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_LOWER_95]
to
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_UPPER_95].
The standardized paired effect was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.STANDARDIZED_PAIRED_EFFECT_DZ],
the Holm-adjusted p value was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.HOLM_ADJUSTED_P],
and the probability of superiority was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PROBABILITY_OF_SUPERIORITY].
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT]

### Resource realization

U-Net+RES contained
[PENDING:RESOURCE.UNET_RES.PARAMETER_COUNT] parameters and required
[PENDING:RESOURCE.UNET_RES.FLOPS_PER_DECLARED_INPUT] FLOPs per declared
input. Mean training time was
[PENDING:RESOURCE.UNET_RES.TRAINING_ACCELERATOR_HOURS_MEAN] accelerator
hours per fold-seed run, and median external end-to-end latency was
[PENDING:RESOURCE.UNET_RES.INFERENCE_END_TO_END_P50_SECONDS] seconds per
volume. Peak framework-allocated unified memory was
[PENDING:RESOURCE.UNET_RES.TRAINING_PEAK_FRAMEWORK_ALLOCATED_UNIFIED_MEMORY_BYTES_MAX]
bytes.

The parameter-matched plain U-Net contained
[PENDING:RESOURCE.UNET_PARAMETER_MATCHED_RES.PARAMETER_COUNT]
parameters, required
[PENDING:RESOURCE.UNET_PARAMETER_MATCHED_RES.FLOPS_PER_DECLARED_INPUT]
FLOPs per declared input, and had median end-to-end latency
[PENDING:RESOURCE.UNET_PARAMETER_MATCHED_RES.INFERENCE_END_TO_END_P50_SECONDS]
seconds per volume. Accuracy-cost non-dominance was reported separately for
parameters, FLOPs, training time, allocated unified memory, and end-to-end
latency.

### Secondary, lesion, subgroup, and qualitative results

All prespecified model contrasts and all frozen models were retained in the
generated tables, including negative and failed outcomes. Regional overlap,
boundary, empty-mask, lesion detection, lesion-wise, and false-positive lesion
endpoints were reported without replacing the primary endpoint. Supportive
other-neoplasm results and development-budget sensitivities were separated
from the external confirmatory analysis.

Subgroup estimates were exploratory. Cells below the frozen minimum
denominator were labeled descriptive, confidence intervals were withheld for
those cells, and no multiplicity-adjusted subgroup claim was made.
Qualitative panels were chosen by the frozen rules and not by manual case
preference.
<!-- END_ARTIFACT_BOUND_RESULTS -->

## Discussion

The primary result was:
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT]
This statement is intentionally narrower than a generic claim that residual or
context modules improve glioma segmentation. It applies to the exact published
RES implementation, its parameter-matched control, the common seeds and folds,
the frozen loss and convergence rules, and the independent confirmatory
cohort.

The same-width contrast addresses the behavior of the historical base-width
comparison but remains capacity-confounded. The parameter-matched contrast is
the primary component estimand, whereas the compute-matched contrast tests a
different constraint. Agreement among these views would strengthen a component
interpretation; disagreement would show that the answer depends on the
resource constraint. The study therefore does not collapse them into one
ranking.

The strong baselines provide context rather than a tournament assembled from
literature scores. nnU-Net tests whether coordinated pipeline configuration
outweighs a handcrafted component change [7,8]. The 2.5D and 3D systems test
whether through-plane context changes the bounded 2D conclusion [11,15].
Because all predictions pass through one evaluator and one external cohort,
differences are not attributed to unmatched datasets or metric code.

Lesion and boundary results are central to interpretation. A favorable mean
Dice can coexist with missed ET lesions, infinite one-empty HD95, or
false-positive components. We therefore report the complete endpoint family
and use qualitative panels to expose success, central tendency, hard failure,
false-positive burden, boundary failure, and inter-model disagreement. These
panels illustrate measured behavior; they do not estimate prevalence or
replace patient-level statistics.

External evaluation on BraTS-Africa introduces population, institution, and
acquisition shift that is absent from an internal split [14]. It does not,
however, establish universal transportability or clinical utility. The
processed cohort and reference labels have their own selection and annotation
processes, and the supportive other-neoplasm cases are not part of the
confirmatory estimand.

Resource interpretation is similarly bounded. Parameter count, graph compute,
training time, allocated unified memory, and latency measure different costs.
A model can be favorable on one axis and unfavorable on another. The Pareto
views preserve those tradeoffs and avoid converting heterogeneous costs into
an arbitrary efficiency score.

### Limitations

First, this is a retrospective public-data study and does not evaluate a
prospective clinical workflow. Second, the confirmatory cohort is independent
of BraTS 2020 by the implemented overlap audit, but it is not representative
of every institution, scanner, tumor phenotype, or annotation practice.
Third, the primary practical threshold is a prespecified interpretation rule,
not a clinical minimal important difference. Fourth, some comparators differ
in dimensionality and official training systems; equal patients, folds, seeds,
evaluator, and reporting do not make their optimization procedures identical.
Fifth, Apple MPS results describe the realized unified-memory device and should
not be generalized to CUDA throughput or memory without new measurements.
Sixth, artifact-level clean-clone reproduction does not constitute full
data-and-compute retraining.

### Conclusion
This study evaluates published BU-Net components under patient-level,
equal-seed, capacity- and compute-aware development with a single frozen
external session.
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT]
The contribution is controlled evidence and traceability, not a claim that RES
or WC was invented here. Clinical utility, universal generalization, and
state-of-the-art performance are not established.

## Data availability

BraTS 2020 and BraTS-Africa remain subject to their source access terms. Raw
medical images and labels are not redistributed by this repository. Versioned
manifests, hashes, split assignments, and data-setup instructions are provided
to permit authorized reconstruction without exposing source data.

## Code and artifact availability

The source code, frozen configurations, tests, and nonrestricted audit
artifacts are available at the project repository. A release identifier and
archival DOI will be inserted only after the final passing artifact bundle is
created. Model checkpoint redistribution is subject to author confirmation and
the applicable data-derived artifact terms.

## Anonymized Acknowledgments

[AUTHOR CONFIRMATION REQUIRED: enter an anonymized acknowledgment or state none.]

## Figure Legends

[GENERATED DURING FINAL BUILD: no more than six figures.]

## References

1. Menze BH, Jakab A, Bauer S, et al. The Multimodal Brain Tumor Image
   Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*.
   2015;34:1993–2024. doi:10.1109/TMI.2014.2377694.
2. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for
   Biomedical Image Segmentation. *MICCAI*. 2015:234–241.
   doi:10.1007/978-3-319-24574-4_28.
3. Rehman MU, Cho S, Kim JH, Chong KT. BU-Net: Brain Tumor Segmentation Using
   Modified U-Net Architecture. *Electronics*. 2020;9:2203.
   doi:10.3390/electronics9122203.
4. He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition.
   *CVPR*. 2016:770–778. doi:10.1109/CVPR.2016.90.
5. Renard F, Guedria S, De Palma N, Vuillerme N. Variability and
   reproducibility in deep learning for medical image segmentation.
   *Scientific Reports*. 2020;10:13724.
   doi:10.1038/s41598-020-69920-0.
6. Åkesson J, Töger J, Heiberg E. Random effects during training:
   implications for deep learning-based medical image segmentation.
   *Computers in Biology and Medicine*. 2024;180:108944.
   doi:10.1016/j.compbiomed.2024.108944.
7. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
   self-configuring method for deep learning-based biomedical image
   segmentation. *Nature Methods*. 2021;18:203–211.
   doi:10.1038/s41592-020-01008-z.
8. Isensee F, Wald T, Ulrich C, et al. nnU-Net Revisited: A Call for Rigorous
   Validation in 3D Medical Image Segmentation. In: *Medical Image Computing
   and Computer Assisted Intervention—MICCAI 2024*. 2024:488–498.
   doi:10.1007/978-3-031-72114-4_47.
9. Wang W, Chen C, Ding M, Li J, Yu H, Zha S. TransBTS: Multimodal Brain
   Tumor Segmentation Using Transformer. In: *Medical Image Computing and
   Computer Assisted Intervention—MICCAI 2021*. LNCS 12901. 2021:109–119.
   doi:10.1007/978-3-030-87193-2_11.
10. Zhou H-Y, Guo J, Zhang Y, Han X, Yu L, Wang L, Yu Y. nnFormer: Volumetric
    Medical Image Segmentation via a 3D Transformer. *IEEE Transactions on
    Image Processing*. 2023;32:4036–4045.
    doi:10.1109/TIP.2023.3293771.
11. Hatamizadeh A, Nath V, Tang Y, Yang D, Roth HR, Xu D. Swin UNETR: Swin
    Transformers for Semantic Segmentation of Brain Tumors in MRI Images. In:
    *BrainLes 2021*. 2022:272–284.
    doi:10.1007/978-3-031-08999-2_22.
12. Maier-Hein L, et al. Metrics Reloaded: recommendations for image analysis
    validation. *Nature Methods*. 2024;21:195–212.
    doi:10.1038/s41592-023-02151-z.
13. Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial
    Intelligence in Medical Imaging: 2024 Update. *Radiology: Artificial
    Intelligence*. 2024;6(4):e240300.
    doi:10.1148/ryai.240300.
14. Adewole M, Rudie JD, Gbadamosi A, et al. The BraTS-Africa Dataset:
    Expanding the Brain Tumor Segmentation Data to Capture African
    Populations. *Radiology: Artificial Intelligence*. 2025;7(4):e240528.
    doi:10.1148/ryai.240528.
15. Avesta A, Hossain S, Lin M, Aboian M, Krumholz HM, Aneja S. Comparing 3D,
    2.5D, and 2D Approaches to Brain Image Auto-Segmentation.
    *Bioengineering*. 2023;10(2):181.
    doi:10.3390/bioengineering10020181.
16. Abraham N, Khan NM. A Novel Focal Tversky Loss Function With Improved
    Attention U-Net for Lesion Segmentation. *IEEE ISBI*. 2019:683–687.
    doi:10.1109/ISBI.2019.8759329.


## Tables

[GENERATED DURING FINAL BUILD: no more than four tables; each table will begin on a separate page and contain no merged cells.]
