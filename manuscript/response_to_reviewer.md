# Response to the reviewer

**Manuscript:** *Leakage-Safe Multi-Seed Evaluation of Published BU-Net Components for Resource-Constrained 2D Glioma Segmentation*

We thank the reviewer for identifying problems that could not be corrected by editorial changes alone. We rebuilt the study around a canonical cohort, patient-level partitions, matched training, guarded test access, reproducible artifacts, and appropriately limited claims. The old cross-architecture ranking and clinical or state-of-the-art language were removed.

## 1. Attribution and novelty

**Concern:** RES and WC appeared to be presented as new contributions although they were introduced in BU-Net.

**Response:** Corrected throughout. RES, WC, and their combination are explicitly attributed to Rehman et al. (2020). The title, abstract, architecture figure, Methods, Discussion, and Conclusion describe this as an evaluation and reimplementation study. No architectural novelty is claimed. A dedicated implementation note records deliberate deviations from the source paper.

## 2. BraTS 2019/2020 overlap and leakage

**Concern:** Pooling BraTS editions could duplicate patients and contaminate splits.

**Response:** We did not pool the editions. BraTS 2020 training is the only canonical labeled cohort. BraTS 2019 is used only for content-based identity auditing. All 335 BraTS 2019 cases map to BraTS 2020; 34 cases are new in 2020. One mapped segmentation contains a two-voxel annotation revision. The final 369 patients were split at patient level into 258/37/74 training/validation/internal-test cases, with zero identifier or content-signature overlap.

## 3. Slice-level leakage

**Concern:** A slice-level split would allow images from one patient in several subsets.

**Response:** All partitions and statistical analyses now use the patient as the unit. Slices are sampled only within an already assigned training patient. The split manifests and hashes are frozen and tested automatically.

## 4. Unfair or inconsistent protocols

**Concern:** Architectures appeared to receive different losses, schedules, or training opportunities.

**Response:** Every final candidate now uses the same inputs, normalization, augmentation, BCE+FTL objective, AdamW optimizer, learning rate, weight decay, batch size, scheduler, device, and 2,000-step/0.5-hour cap. The protocol is intentionally bounded and this limitation is stated. Development screens, confirmation runs, and test evaluation have distinct roles and machine-readable registries.

## 5. Unsupported transformer or Swin conclusions

**Concern:** Undertrained transformer comparisons cannot support architectural conclusions.

**Response:** Removed. No Swin, transformer, attention, or literature score appears as an experimental comparator. nnU-Net, TransBTS, and nnFormer are cited only as important untested context. The manuscript explicitly states that no 3D, transformer, self-configuring, or external-validation comparison was performed.

## 6. Missing ablation

**Concern:** The contribution of individual components was unclear.

**Response:** The rebuilt design screens standard U-Net, U-Net+RES, U-Net+WC, BU-Net (RES+WC), residual-block U-Net, and residual-block U-Net+WC under one protocol. The final frozen comparison includes standard U-Net, U-Net+RES, and BU-Net. On the internal test, mean regional Dice is 0.736, 0.756, and 0.752, respectively. The paired U-Net+RES versus standard U-Net difference is +0.020 (Holm p=0.004); the BU-Net versus standard U-Net difference is +0.017 (Holm p=0.014). We interpret this as evidence for RES under the present protocol, not architectural novelty.

## 7. Loss definition and selection

**Concern:** The loss formulation, class handling, and selection process were insufficiently specified.

**Response:** The Methods now provide the complete BCE+FTL formula and its parameters: equal term weights, alpha 0.3, beta 0.7, gamma 0.75, smoothing 1e-5, foreground-only FTL, and no class weights. Seven loss arms were screened once on the validation subset. Their full results are reported as development-only evidence.

## 8. Inconsistent metrics and aggregation

**Concern:** Metric definitions, empty-mask rules, and aggregation could change model rankings.

**Response:** The evaluator now fixes WT/TC/ET label mapping, Dice, HD95, surface Dice at 1 mm, lesion metrics, 26-connectivity, maximum-total-IoU lesion matching, and explicit empty-mask behavior. Seeds are averaged within patient before inference; patients are never multiplied into slice- or seed-level pseudo-samples. Infinite and undefined secondary values are retained and counted.

## 9. Statistics and uncertainty

**Concern:** Point estimates without patient-level uncertainty or multiplicity control were inadequate.

**Response:** The primary outcome and three contrasts were frozen before test access. We use 10,000 patient-level bootstrap resamples, 100,000 two-sided paired sign-flip permutations, paired effect size dz, and Holm correction. Regional and secondary endpoints are identified as estimation-only.

## 10. Resource and efficiency claims

**Concern:** “Lightweight,” “efficient,” or clinical deployment language lacked measurement.

**Response:** We removed those unqualified claims. The paper reports parameters, checkpoint size, MACs, FLOPs, p50/p95 volume latency, throughput, allocated/reserved memory, and development accelerator-hours from the implemented candidates on one host. BU-Net is explicitly shown to require more resources than standard U-Net and U-Net+RES. No clinical deployment claim remains.

## 11. Qualitative analysis and failure cases

**Concern:** The paper lacked image-level evidence and failure analysis.

**Response:** Artifact-derived figures now show modalities, ground truth, all frozen predictions, false-positive/false-negative overlays, and preselected success, hard, and failure cases. The failure case is discussed alongside modest ET lesion-recall and lesion-wise Dice estimates.

## 12. Reproducibility and code

**Concern:** The reported results could not be reproduced.

**Response:** The repository now contains configurations, model and loss implementations, evaluator tests, patient manifests and hashes, run registries, statistical scripts, derived tables and figures, environment lock, claim ledger, and reproduction instructions. A clean-clone audit verified 230 tracked-artifact hashes, static checks, the test suite, two byte-identical manuscript-input generations, and a clean worktree.

## 13. Requested 3D, nnU-Net, transformer, and external validation

**Concern:** Strong conclusions would require modern 3D/self-configuring baselines and external validation.

**Response:** We agree with the evidentiary requirement and narrowed the paper instead of fabricating or importing incomparable results. The revised manuscript does not claim superiority over these systems, state of the art, external generalization, or clinical applicability. These experiments remain future work and are prominent limitations.

## 14. Manuscript-wide claim correction

**Concern:** The original framing was broader than the evidence.

**Response:** The manuscript was rewritten. The conclusion is limited to a leakage-safe internal comparison: RES improved the primary overlap endpoint over standard U-Net under this bounded 2D protocol; adding WC in the full BU-Net did not improve that endpoint over RES alone and increased resource demand. Boundary and lesion-level findings are reported as mixed.
