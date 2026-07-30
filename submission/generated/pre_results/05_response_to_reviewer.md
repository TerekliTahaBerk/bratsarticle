> **PRE-RESULTS PREVIEW - NOT FOR SUBMISSION. Scientific result fields and author declarations remain unresolved. Do not upload this file to a journal.**

# Response to the reviewer

**Revised manuscript:** *Capacity- and Compute-Matched Evaluation of Published
BU-Net Components with Multi-Seed Development and Independent External Testing
for Multimodal Glioma Segmentation*

We agree that the rejected manuscript could not be repaired through language
editing or a few additional experiments. The study was rebuilt as a controlled
evaluation of published components. This response is a result-bound template:
it may be rendered only after the development matrix, statistical freeze, and
single external testing session are complete. Page and line references remain
layout placeholders until the final proof is generated.

## R01 — BU-Net attribution and novelty

**Concern.** RES and WC were not attributed to BU-Net and appeared to be
positioned as new contributions.

**Direct response.** We agree. RES and WC are now attributed to Rehman et al.
throughout. The revised work does not claim a new architecture.

**Exact action.** We replaced the novelty framing with a capacity-controlled
evaluation question, cited BU-Net as the primary source for both components,
and separated conventional residual blocks from RES pathways.

**New evidence.** The model matrix contains plain, RES, WC, BU-Net,
conventional residual-block, parameter-matched, and compute-matched controls.

**Manuscript location.** [FINAL PAGE/LINES: title, Introduction, Models, and
Discussion]; [FINAL TABLE/FIGURE: architecture and model-configuration table].

**Repository evidence.** `literature/q1q2_verified_sources.yaml`;
`configs/q1q2_v2/model_matrix.yaml`;
`reports/q1q2_v2/model_matrix_validation.json`.

**Remaining limitation.** The contribution is controlled evidence about
published components, not architectural invention.

## R02 — BraTS edition overlap and canonical cohort

**Concern.** Combining BraTS editions without patient deduplication could
duplicate patients and create leakage.

**Direct response.** We did not pool BraTS editions. BraTS 2020 is the sole
development cohort; BraTS 2019 is used only for identity and content auditing.

**Exact action.** We created read-only inventories, file/content hashes,
cross-edition mappings, and a canonical patient manifest.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The revised development cohort contains
[PENDING:DESIGN.DEVELOPMENT_PATIENTS] unique labeled patients.
BraTS-Africa contributes
[PENDING:DESIGN.EXTERNAL_CONFIRMATORY_PATIENTS] confirmatory glioma
patients and [PENDING:DESIGN.EXTERNAL_SUPPORTIVE_PATIENTS] separately
reported other-neoplasm patients.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Datasets and Overlap Audit];
[FINAL FIGURE: cohort identity and overlap flow].

**Repository evidence.** `manifests/canonical/brats2020_canonical_manifest.csv`;
`manifests/audit/duplicate_mapping.csv`;
`reports/q1q2_v2/external_gate_c_summary.json`.

**Remaining limitation.** Content-based auditing reduces identifiable overlap
risk but cannot prove that every historical data lineage is documented.

## R03 — Patient-level partitions and legacy-test isolation

**Concern.** The original split was unspecified and might have been made at
slice level; the meaning of “test set” was unclear.

**Direct response.** Every revised partition is patient-level. The previously
opened internal subset is retained only as legacy evidence and is unavailable
to all new model selection and inference paths.

**Exact action.** We replaced the old split for v2 development with
deterministic patient-level cross-validation and an independent external test.
External and legacy accesses are separately guarded and logged.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
Each development patient appears in exactly one of
[PENDING:DESIGN.FOLDS] validation folds, and every main model uses the
same [PENDING:DESIGN.SEEDS] training seeds.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Study Design and Leakage Controls].

**Repository evidence.** `splits/q1q2_v2/split_metadata.json`;
`configs/q1q2_v2/gate_g_freeze.yaml`; `artifacts/test_access_log.jsonl`.

**Remaining limitation.** The folds support development estimation; they are
not relabeled as an official BraTS test.

## R04 — Common and fair experimental protocol

**Concern.** Models were previously compared across different data subsets,
losses, schedules, and training opportunities.

**Direct response.** The heterogeneous historical comparison was removed.
All revised comparisons use the same cohort contracts, folds, seed list,
evaluator, preprocessing boundaries, and prespecified budget definitions.

**Exact action.** We froze a shared native protocol, retained official
framework behavior where required for nnU-Net, and record every deliberate
model-specific exception.

**New evidence.** All planned runs and failures are retained in the run
registry; literature-derived scores are excluded from experimental tables.

**Manuscript location.** [FINAL PAGE/LINES: Training and Compute Protocol];
[FINAL TABLE: training and tuning budgets].

**Repository evidence.** `configs/q1q2_v2/training_protocol.yaml`;
`configs/q1q2_v2/model_matrix.yaml`; `configs/q1q2_v2/gate_g_freeze.yaml`.

**Remaining limitation.** Fair comparison does not mean that official
self-configuring systems and native implementations have identical
optimization internals.

## R05 — Convergence and prior undertraining

**Concern.** Short or unequal training, especially the prior Swin experiment,
could not support conclusions about architecture.

**Direct response.** The prior short-run curves are legacy evidence only.
They do not appear in revised architecture comparisons.

**Exact action.** We prespecified convergence checks, validation frequency,
early-stopping behavior, maximum steps, terminal and best checkpoints, and
separate compute-matched runs. Failed convergence remains visible.

**New evidence.** The final convergence table and ranking-stability figure are
generated only from complete run artifacts; an incomplete model cannot be
interpreted as an architectural failure.

**Manuscript location.** [FINAL PAGE/LINES: Training and Compute Protocol];
[FINAL FIGURE: convergence and ranking stability].

**Repository evidence.** `configs/q1q2_v2/training_protocol.yaml`;
`configs/protocols/convergence_matched_mps.yaml`;
`configs/q1q2_v2/gate_g_freeze.yaml`.

**Remaining limitation.** Convergence is operationally defined under the
frozen device and optimization protocol, not proven in an asymptotic sense.

## R06 — Strong experimental baselines

**Concern.** Direct BU-Net, residual U-Net, nnU-Net, volumetric, and modern
hybrid baselines were missing.

**Direct response.** The revised matrix contains the component core, official
nnU-Net configurations, a controlled five-slice model, and a maintained Swin
UNETR implementation.

**Exact action.** We validate all models against one data/evaluator contract
and prohibit replacing an unexecuted comparator with a literature score.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The frozen comparison matrix contains
[PENDING:DESIGN.MODELS] models and
[PENDING:DESIGN.DEVELOPMENT_RUNS] development runs.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Models and Component Attribution];
[FINAL TABLE: model configurations].

**Repository evidence.** `configs/q1q2_v2/model_matrix.yaml`;
`reports/q1q2_v2/model_matrix_validation.json`;
`reports/q1q2_v2/nnunet_planning_summary.json`.

**Remaining limitation.** The chosen modern baselines do not exhaust the
rapidly changing segmentation literature.

## R07 — Component ablation and capacity attribution

**Concern.** RES, WC, residual blocks, and model capacity were not isolated.

**Direct response.** We agree that the historical comparison was
capacity-confounded. The revised primary estimand compares U-Net+RES with a
plain parameter-matched U-Net.

**Exact action.** We added same-width, parameter-matched, compute-matched, WC,
BU-Net, conventional residual-block, and interaction controls.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
For the primary external contrast, U-Net+RES achieved
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.FIRST_MEAN] and
the matched plain U-Net achieved
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.SECOND_MEAN].
The paired difference was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.MEAN_DIFFERENCE]
with interval
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_LOWER_95]
to
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_BOOTSTRAP_UPPER_95].
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT]
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Primary Capacity-Controlled
Comparison]; [FINAL TABLE: primary contrasts].

**Repository evidence.** `reports/q1q2_v2/model_matching_summary.json`;
`configs/q1q2_v2/statistical_analysis_plan.yaml`;
`artifacts/q1q2_v2/statistics/primary_contrasts.csv`.

**Remaining limitation.** The primary conclusion applies to the implemented
RES definition and frozen capacity target, not every residual architecture.

## R08 — Loss definition, output semantics, and loss ablation

**Concern.** The loss equation, activations, class inclusion, and selection
procedure were incomplete.

**Direct response.** The revised loss catalog defines every activation,
channel, reduction, class inclusion rule, parameter, smoothing term, and
empty-class behavior.

**Exact action.** Development-only comparison covers softmax-consistent and
hybrid candidates. One loss is frozen before main runs, and a prespecified
architecture-by-loss sensitivity is kept separate from the primary estimand.

**New evidence.** The selected objective is
**[PENDING:METHOD.SELECTED_LOSS]**; external labels and the legacy internal
subset are prohibited from loss selection.

**Manuscript location.** [FINAL PAGE/LINES: Loss Selection and Optimization];
[FINAL SUPPLEMENT: executable loss definitions and interaction analysis].

**Repository evidence.** `configs/losses/catalog.yaml`;
`configs/q1q2_v2/loss_protocol.yaml`;
`reports/q1q2_v2/loss_methods.json`.

**Remaining limitation.** The loss screen is intentionally restricted and
does not establish that the selected loss is universally optimal.

## R09 — Complete central evaluation and empty-mask behavior

**Concern.** Dice/IoU reporting was inconsistent, empty slices were unclear,
and boundary and lesion endpoints were missing.

**Direct response.** All revised predictions pass through one evaluator.
Validation and external testing retain complete volumes, including empty
slices.

**Exact action.** We froze regional overlap, boundary, surface, rate, volume,
lesion-detection, lesion-wise, and false-positive endpoints with explicit
empty-mask and infinite-distance rules.

**New evidence.** The final tables report WT, TC, and ET separately and retain
one-empty HD95 as infinity with its rate shown alongside finite summaries.

**Manuscript location.** [FINAL PAGE/LINES: Central Evaluation];
[FINAL TABLES: regional and lesion-wise endpoints].

**Repository evidence.** `configs/q1q2_v2/evaluation.yaml`;
`configs/q1q2_v2/evaluation_sensitivity.yaml`;
`src/evaluation`.

**Remaining limitation.** Reference annotations remain imperfect proxies for
the underlying biological boundaries.

## R10 — Equal seeds and patient-level statistical inference

**Concern.** Replication counts, uncertainty, paired tests, effect sizes, and
multiplicity control were absent.

**Direct response.** Every main model now shares the same folds and seeds.
The patient is the inferential unit.

**Exact action.** We prespecified paired patient bootstrap intervals,
hierarchical seed-patient resampling, paired sign-flip tests, Holm correction,
effect size, and probability-of-superiority reporting.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The primary contrast used
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PAIRED_PATIENT_COUNT]
paired external patients. Its Holm-adjusted p value was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.HOLM_ADJUSTED_P]
and probability of superiority was
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.PROBABILITY_OF_SUPERIORITY].
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Statistical Analysis and Primary
Results]; [FINAL FIGURE: hierarchical uncertainty].

**Repository evidence.** `configs/q1q2_v2/seeds.yaml`;
`configs/q1q2_v2/statistical_analysis_plan.yaml`;
`artifacts/q1q2_v2/statistics/completion.json`.

**Remaining limitation.** Statistical precision is bounded by the available
independent external patients and does not imply clinical importance.

## R11 — Measured resource and efficiency evidence

**Concern.** Resource-efficiency claims lacked parameters, computation,
memory, and latency measurements.

**Direct response.** Unqualified efficiency and deployment language was
removed. Costs are reported as separate measured dimensions.

**Exact action.** We profile parameters, graph computation, accelerator time,
preprocessing, forward, post-processing, end-to-end latency, and
backend-appropriate memory counters under a frozen protocol.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
U-Net+RES contains [PENDING:RESOURCE.UNET_RES.PARAMETER_COUNT]
parameters, requires
[PENDING:RESOURCE.UNET_RES.FLOPS_PER_DECLARED_INPUT] FLOPs per declared
input, and has median external end-to-end latency
[PENDING:RESOURCE.UNET_RES.INFERENCE_END_TO_END_P50_SECONDS] seconds per
volume. Peak framework-allocated unified memory is
[PENDING:RESOURCE.UNET_RES.TRAINING_PEAK_FRAMEWORK_ALLOCATED_UNIFIED_MEMORY_BYTES_MAX]
bytes.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: Resource Analysis and Resource
Realization]; [FINAL FIGURES: capacity and compute Pareto views].

**Repository evidence.** `configs/q1q2_v2/resource_profile_protocol.yaml`;
`configs/q1q2_v2/resource_execution.yaml`;
`artifacts/q1q2_v2/resources/accuracy_cost_pareto.csv`.

**Remaining limitation.** Apple MPS measurements describe one
unified-memory system and are not CUDA throughput or VRAM measurements.

## R12 — Controlled dimensionality and transformer context

**Concern.** The original manuscript compared two-dimensional and volumetric
or transformer systems without a common protocol.

**Direct response.** Heterogeneous literature scores and subjective rankings
were removed from experimental figures.

**Exact action.** Native two-dimensional, controlled five-slice, official
nnU-Net volumetric, and maintained Swin UNETR systems are evaluated on common
patients through the central evaluator.

**New evidence.** Dimensionality and architecture results are reported as
model-specific estimates and paired comparisons where prespecified, not as a
claim that one family is universally preferable.

**Manuscript location.** [FINAL PAGE/LINES: Models, Results, and Discussion].

**Repository evidence.** `configs/q1q2_v2/model_matrix.yaml`;
`reports/q1q2_v2/nnunet_planning_summary.json`;
`reports/q1q2_v2/m1_execution_decision.json`.

**Remaining limitation.** Dimensionality, implementation ecosystem, and
optimization cannot be separated by a single contrast.

## R13 — Independent external testing and domain shift

**Concern.** The original work lacked an independent hospital/domain test and
could not support generalization or clinical claims.

**Direct response.** The legacy internal subset is not reused. BraTS-Africa
provides the independent domain-shift cohort after a zero-overlap audit.

**Exact action.** We permit one audited external session only after model,
checkpoint, endpoint, multiplicity, and analysis freeze. No external tuning,
retraining, or label-informed adaptation is allowed.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The external session evaluates
[PENDING:DESIGN.EXTERNAL_MAIN_CHECKPOINTS] frozen checkpoints. The
confirmatory cohort contains
[PENDING:DESIGN.EXTERNAL_CONFIRMATORY_PATIENTS] glioma patients; other
neoplasms are reported separately as supportive evidence.
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: External Evaluation and External
Results]; [FINAL FIGURE: external subgroup analysis].

**Repository evidence.** `reports/q1q2_v2/external_gate_c_summary.json`;
`configs/q1q2_v2/gate_h_external.yaml`;
`artifacts/q1q2_v2/external_access_log.jsonl`.

**Remaining limitation.** External testing on one public African cohort does
not establish universal transportability or clinical utility.

## R14 — Qualitative success and failure analysis

**Concern.** The original work showed no segmentation examples, overlays, or
hard failures.

**Direct response.** The revised design includes success, median, lesion,
false-positive, boundary, and model-disagreement cases.

**Exact action.** Case identities are selected after evaluation by frozen
deterministic rules. Panels include all modalities, reference labels, all
models, lesion components, and false-positive/false-negative overlays.

**New evidence.** Both favorable and adverse cases remain in the final panel
set; identities are not manually substituted.

**Manuscript location.** [FINAL PAGE/LINES: Qualitative Analysis];
[FINAL FIGURE: qualitative panels].

**Repository evidence.** `configs/q1q2_v2/qualitative_protocol.yaml`;
`configs/q1q2_v2/qualitative_execution.yaml`;
`artifacts/q1q2_v2/qualitative/selected_cases.json`.

**Remaining limitation.** Selected examples illustrate measured failure
modes; they do not estimate their population prevalence.

## R15 — Artifact-derived figures and tables

**Concern.** Heterogeneous scores and subjective cost or architecture ratings
were presented as quantitative comparisons.

**Direct response.** Those figures and tables were discarded.

**Exact action.** Every revised quantitative panel is generated from the
central metrics, statistical, subgroup, resource, or qualitative artifacts.
No literature score is plotted beside experimental output.

**New evidence.** Figure manifests retain input hashes, output hashes, and
generation metadata; table values are generated rather than manually copied.

**Manuscript location.** [FINAL PAGE/LINES: Results and figure captions];
[FINAL SUPPLEMENT: output manifest].

**Repository evidence.** `configs/q1q2_v2/figure_execution.yaml`;
`src/bratsarticle/analysis/q1q2_figures.py`;
`artifacts/q1q2_v2/figures/completion.json`.

**Remaining limitation.** Visual presentation choices remain editorial even
when the plotted values and selected cases are artifact-bound.

## R16 — Primary literature coverage

**Concern.** Essential primary sources and critical comparisons were missing.

**Direct response.** The literature ledger now includes BU-Net, Focal Tversky,
nnU-Net and its controlled reevaluation, TransBTS, nnFormer, Swin UNETR,
dimensionality studies, Metrics Reloaded, CLAIM, seed variability, and the
BraTS-Africa descriptor.

**Exact action.** Bibliographic metadata and contribution fields were checked
against primary publisher, proceedings, author-manuscript, or official data
pages. Unverified fields remain explicitly marked.

**New evidence.** The literature matrix separates datasets, dimensionality,
seed reporting, overlap auditing, external evaluation, metrics, code, and
license instead of treating published scores as a benchmark.

**Manuscript location.** [FINAL PAGE/LINES: Introduction, Related Work, and
References]; [FINAL SUPPLEMENT: novelty matrix].

**Repository evidence.** `literature/q1q2_verified_sources.yaml`;
`reports/q1q2_v2/novelty_matrix.csv`;
`reports/q1q2_v2/novelty_assessment.md`.

**Remaining limitation.** The manuscript is an experimental methods study,
not a systematic review, and does not claim exhaustive literature coverage.

## R17 — Reproducibility, code, environment, and license

**Concern.** Code, splits, configurations, environment details, and a
reproducibility package were absent.

**Direct response.** The repository now contains versioned code,
configurations, manifests, tests, environment locks, audit contracts, and an
Apache license.

**Exact action.** Gate I verifies source cleanliness, tests, static analysis,
hashes, downstream numerical regeneration, standard figures, and qualitative
panel hashes. Full retraining and artifact-only reproduction are described as
different scopes.

**New evidence.** Final reproduction status is reported from the Gate I
completion artifact; a clean-clone audit is not represented as full
retraining.

**Manuscript location.** [FINAL PAGE/LINES: Reproducibility and Code/Data
Availability]; [FINAL TABLE: reproducibility checklist].

**Repository evidence.** `REPRODUCIBILITY.md`;
`configs/q1q2_v2/reproducibility_execution.yaml`; `LICENSE`.

**Remaining limitation.** Authorized raw data remain subject to source access
terms and are not redistributed.

## R18 — Writing, organization, and bounded claims

**Concern.** The original text was repetitive, unnatural, and made causal,
clinical, or architecture-wide claims beyond the evidence.

**Direct response.** The manuscript was rewritten as one experimental study
with a structured abstract, explicit estimand, compact Methods, result-led
Discussion, and bounded Conclusion.

**Exact action.** We removed tutorial taxonomy, pseudo-quantification,
literature-score rankings, unsupported mechanism claims, and clinical
applicability language. Gate J audits result provenance and inferential
wording.

**New evidence.**
<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->
The final primary interpretation is:
[PENDING:CONTRAST.UNET_RES_VS_UNET_PARAMETER_MATCHED_RES.INTERPRETATION_TEXT]
<!-- END_ARTIFACT_BOUND_RESULTS -->

**Manuscript location.** [FINAL PAGE/LINES: complete revised manuscript].

**Repository evidence.** `manuscript/q1q2_v2_manuscript.template.md`;
`configs/q1q2_v2/claim_execution.yaml`;
`claims/q1q2_v2_claim_ledger.csv`.

**Remaining limitation.** Professional language editing cannot substitute for
author review of scientific meaning, and all authors must approve the final
wording.

## R19 — Submission declarations and exact response parity

**Concern.** The revised response must not claim results or resource fields
that are absent from the manuscript and submission package.

**Direct response.** The response and manuscript use the same claim registry.
Every rendered value records its source file, hash, selector, column, format,
and rendered-document hash.

**Exact action.** Final page/line references are added only after layout.
Author order, affiliations, contributions, ethics, funding, conflicts,
acknowledgements, correspondence details, and disclosure text require author
confirmation.

**New evidence.** Gate J must pass for both the manuscript and this response;
Gate K remains blocked until the author declarations and journal-specific
files are complete.

**Manuscript location.** [FINAL PAGE/LINES: declarations and availability
sections].

**Repository evidence.** `configs/q1q2_v2/claim_execution.yaml`;
`submission/AUTHOR_CONFIRMATION_REQUIRED.md`;
`artifacts/q1q2_v2/claims/completion.json`.

**Remaining limitation.** No author declaration, journal choice, release DOI,
or page/line citation is inferred from repository code.
