# CLAIM 2024 checklist — Q1/Q2 v2 study

Guideline: Tejani AS, Klontzas ME, Gatti AA, et al. *Checklist for Artificial
Intelligence in Medical Imaging: 2024 Update*. Radiology: Artificial
Intelligence. 2024;6(4):e240300. doi:10.1148/ryai.240300.

This is a pre-results compliance template. “Yes” and “Partial” entries remain
pending until real results are rendered and final page/line references are
inserted. Repository paths identify implementation evidence; they do not
replace reporting in the manuscript. Author-dependent declarations are not
inferred from code.

| ID | CLAIM 2024 item | Status before final layout | Final manuscript location | Repository evidence / explanation |
|---|---|---|---|---|
| C01 | Identify the work as an AI methodology study and name the technology in the title or abstract | yes_pending_final_page_lines | [FINAL PAGE/LINES: title and abstract] | `manuscript/q1q2_v2_manuscript.template.md`; deep-learning segmentation and model families are stated without architectural novelty claims. |
| C02 | Provide a structured summary of design, methods, results, conclusions, population, partitions, and availability | partial_pending_final_results | [FINAL PAGE/LINES: structured abstract] | The template covers design and methods; result values and availability identifiers remain Gate J/K outputs. |
| C03 | State scientific and clinical background, intended use, and role of the AI approach | yes_pending_final_page_lines | [FINAL PAGE/LINES: Introduction] | Intended role is methodological segmentation evaluation; no clinical deployment claim is made. |
| C04 | State aims, objectives, and hypotheses | yes_pending_final_page_lines | [FINAL PAGE/LINES: Introduction] | The capacity-controlled RES question and prespecified secondary analyses are explicit. |
| C05 | State whether the study is prospective or retrospective | yes_pending_final_page_lines | [FINAL PAGE/LINES: Study Design] | Retrospective public-data methodology study. |
| C06 | Describe the study goal and predictive task | yes_pending_final_page_lines | [FINAL PAGE/LINES: Study Design and Models] | Four-class multimodal MRI segmentation with regional and lesion-level evaluation. |
| C07 | Describe data sources and their alignment with intended use | yes_pending_final_page_lines | [FINAL PAGE/LINES: Development and External Cohorts] | BraTS 2020 development and BraTS-Africa external testing; `reports/q1q2_v2/external_dataset_selection.md`. |
| C08 | State inclusion and exclusion criteria | yes_pending_final_page_lines | [FINAL PAGE/LINES: Development and External Cohorts] | Cohort roles and exclusion of other neoplasms from confirmatory inference are explicit. |
| C09 | Describe preprocessing in reproducible detail | yes_pending_final_page_lines | [FINAL PAGE/LINES: Preprocessing] | `configs/data/preprocessing_pilot_cached.yaml`; fold and external boundaries are declared. |
| C10 | Describe any selected subset, crop, or extracted image portion | yes_pending_final_page_lines | [FINAL PAGE/LINES: Preprocessing] | Training sampling may enrich tumor content; complete validation/external volumes are retained. |
| C11 | Describe de-identification | partial_pending_final_results | [FINAL PAGE/LINES: Data Availability and Ethics] | Public deidentified sources are used; source-specific de-identification is not repeated locally. Final ethics language requires author confirmation. |
| C12 | Describe handling of missing data | yes_pending_final_page_lines | [FINAL PAGE/LINES: Cohorts and Statistical Analysis] | Modality completeness, metadata missingness, subgroup missingness, and failed predictions have explicit dispositions. |
| C13 | Report relevant image-acquisition information | partial_pending_final_results | [FINAL PAGE/LINES: Cohorts and Supplement] | Available scanner, field-strength, spacing, geometry, and resolution metadata are reported; unavailable acquisition fields remain missing rather than imputed. |
| C14 | Define the reference standard | yes_pending_final_page_lines | [FINAL PAGE/LINES: Cohorts and Central Evaluation] | Source-provided tumor subregion labels and exact WT/TC/ET mapping are stated. |
| C15 | Explain the rationale for the reference standard | yes_pending_final_page_lines | [FINAL PAGE/LINES: Cohorts and Limitations] | BraTS-compatible labels support the frozen task while annotation uncertainty remains a limitation. |
| C16 | State the source of reference annotations and annotator information | partial_pending_final_results | [FINAL PAGE/LINES: Cohorts] | Primary dataset descriptors are cited; the present study performs no new annotation and reports only source-available annotation details. |
| C17 | Describe annotation of the testing dataset | not_applicable | [FINAL PAGE/LINES: External Cohort] | No new test annotation is performed; processed BraTS-Africa reference labels are used under source terms. |
| C18 | Report inter- and intrarater variability or mitigation | no | [FINAL PAGE/LINES: Limitations] | Rater-level repeat annotations are unavailable; this is retained as a limitation and no variability estimate is invented. |
| C19 | Describe assignment to training, tuning, and testing partitions | yes_pending_final_page_lines | [FINAL PAGE/LINES: Study Design and External Evaluation] | `splits/q1q2_v2/split_metadata.json`; `splits/q1q2_v2/external_test.csv`. |
| C20 | State the level at which partitions are disjoint | yes_pending_final_page_lines | [FINAL PAGE/LINES: Study Design and Leakage Controls] | Patient-level disjointness, cross-edition identity audit, and external overlap audit are explicit. |
| C21 | Explain the intended testing sample size | yes_pending_final_page_lines | [FINAL PAGE/LINES: Statistical Analysis] | Fixed public external cohort with precision/power planning; `reports/q1q2_v2/external_precision_analysis.json`. |
| C22 | Describe model inputs, outputs, layers, and modifications in reconstructable detail | yes_pending_final_page_lines | [FINAL PAGE/LINES: Models; Supplementary Table S3] | `configs/q1q2_v2/model_matrix.yaml` and executable implementations. |
| C23 | Report software libraries, frameworks, packages, and versions | yes_pending_final_page_lines | [FINAL PAGE/LINES: Reproducibility] | `environment/q1q2_v2-environment.json`; `environment/q1q2_v2-requirements-lock.txt`. |
| C24 | Describe parameter initialization and starting weights | yes_pending_final_page_lines | [FINAL PAGE/LINES: Models and Training] | Run-resolved model initialization and pretrained-weight status are recorded by configuration and metadata. |
| C25 | Describe training, augmentation, optimization, stopping, and hyperparameters | yes_pending_final_page_lines | [FINAL PAGE/LINES: Loss Selection and Optimization] | `configs/q1q2_v2/training_protocol.yaml`; best and terminal checkpoints are required. |
| C26 | Describe selection of the final model or models | yes_pending_final_page_lines | [FINAL PAGE/LINES: Loss Selection and External Evaluation] | Development-only loss/checkpoint rules and the no-external-selection guard are explicit. |
| C27 | Describe ensembling | yes_pending_final_page_lines | [FINAL PAGE/LINES: External Evaluation] | Fold-seed checkpoint predictions use a prespecified strict-majority nested-region aggregation. |
| C28 | Define performance metrics and their task relevance | yes_pending_final_page_lines | [FINAL PAGE/LINES: Central Evaluation] | `configs/q1q2_v2/evaluation.yaml`; regional, boundary, surface, and lesion endpoints. |
| C29 | Describe statistical uncertainty and comparison methods | yes_pending_final_page_lines | [FINAL PAGE/LINES: Statistical Analysis] | `configs/q1q2_v2/statistical_analysis_plan.yaml`; paired and hierarchical uncertainty with multiplicity control. |
| C30 | Report robustness or sensitivity analyses | partial_pending_final_results | [FINAL PAGE/LINES: Results and Supplement] | Evaluation thresholds, connectivity, HD95 handling, budget, loss interaction, and external subgroup sensitivities are prespecified. |
| C31 | Describe and validate explainability methods, if used | not_applicable | [FINAL PAGE/LINES: Methods or checklist note] | No saliency or post hoc explainability method is used or claimed. |
| C32 | Report evaluation on internal data | yes_pending_final_page_lines | [FINAL PAGE/LINES: Development Results] | Patient-level cross-validation is reported; the previously opened legacy internal subset is not reused for v2 inference. |
| C33 | Report testing on external data | partial_pending_final_results | [FINAL PAGE/LINES: External Evaluation and Results] | Gate H is pending; `configs/q1q2_v2/gate_h_external.yaml` prohibits tuning and requires one audited session. |
| C34 | Provide clinical-trial registration when applicable | not_applicable | [FINAL PAGE/LINES: Study Design] | Retrospective public-data methodology study; no prospective clinical trial is conducted. |
| C35 | Report included and excluded patient counts | partial_pending_final_results | [FINAL PAGE/LINES: Cohort flow and Results] | Counts are claim-bound and the final flow diagram is generated from manifests. |
| C36 | Report demographic and clinical characteristics of each dataset and partition | partial_pending_final_results | [FINAL PAGE/LINES: Cohort table and external subgroups] | All source-available characteristics and explicit missingness are reported; absent demographics are not imputed. |
| C37 | Report model performance and statistical uncertainty on all relevant datasets | partial_pending_final_results | [FINAL PAGE/LINES: Results and Supplement] | Final Gate J values are pending; every frozen model and failed outcome must remain visible. |
| C38 | Report diagnostic-performance estimates and precision for classification tasks | not_applicable | [FINAL PAGE/LINES: checklist note] | The task is semantic/lesion segmentation, not patient-level diagnostic classification; segmentation uncertainty is addressed under C29 and C37. |
| C39 | Provide failure analysis and examples of incorrect results | partial_pending_final_results | [FINAL PAGE/LINES: Lesion and Qualitative Results] | Prespecified post-evaluation selection rules include hard, false-positive, boundary, and disagreement cases. |
| C40 | Discuss study limitations | yes_pending_final_page_lines | [FINAL PAGE/LINES: Limitations] | Public retrospective data, annotation uncertainty, one external cohort, system differences, MPS scope, and reproduction boundaries are explicit. |
| C41 | Discuss implications for practice and the intended clinical role | yes_pending_final_page_lines | [FINAL PAGE/LINES: Discussion and Conclusion] | The manuscript states that clinical utility and universal transportability are not established. |
| C42 | Provide the full protocol or additional technical details | yes_pending_final_page_lines | [FINAL PAGE/LINES: Reproducibility and Supplement] | Versioned configs, supplement, code, and gate documentation provide technical detail. |
| C43 | State availability of software, model, and data | partial_pending_final_results | [FINAL PAGE/LINES: Data and Code Availability] | Code is versioned; raw data remain under source terms; final release/checkpoint availability awaits Gate K and author confirmation. |
| C44 | State funding, support, and funder role | no | [FINAL PAGE/LINES: Funding] | `submission/AUTHOR_CONFIRMATION_REQUIRED.md`; no funding statement is inferred from repository artifacts. |

## Finalization rule

The PDF may be produced only after the result-bearing manuscript is rendered,
the final document is paginated, every Yes/Partial row has a real page/line
reference, unresolved result tokens are absent, and the authors confirm all
declaration-dependent entries.
