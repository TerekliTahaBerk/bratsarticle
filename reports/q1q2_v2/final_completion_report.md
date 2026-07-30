# Q1/Q2 v2 completion report

Report date: 2026-07-30
Decision: **NOT READY — M1 DEVELOPMENT IN PROGRESS**

This is the required hard-blocker completion report. It records completed
design, implementation, and reporting-template work and explicitly does not
claim that the experimental study or submission package is complete.

## 1. Repository identity

- Legacy source commit:
  `ab60a79d8a49a2fe1adb000546f653485796bab1`
- Immutable annotated tag: `v1-bounded-2d-component-study`
- Tag target:
  `ab60a79d8a49a2fe1adb000546f653485796bab1`
- v2 branch: `q1q2-capacity-external-validation`
- Study/package version: `v2.0.0`
- The delivery commit is reported by Git in the final handoff; a tracked file
  cannot contain the hash of the commit that contains itself.

## 2. Gate status

| Gate | Status | Evidence |
|---|---|---|
| A — Legacy snapshot | PASS | Immutable tag and legacy pointers |
| B — Gap audit | PASS | Repository, code–Methods and claim corrections |
| C — External data | PASS | 146 complete patients; zero overlap |
| D — Capacity matching | DESIGN PASS | Parameter and compute controls selected within 2%; final realization pending |
| E — Equal-seed design | DESIGN PASS / EXECUTION PENDING | Twelve models share five seeds and five folds |
| F — Convergence | IN PROGRESS | Bounded development loss screen is active; main matrix follows loss freeze |
| G — Statistical freeze | PRESPECIFIED / NOT FROZEN | Loss and checkpoints do not exist |
| H — External test | NOT STARTED | No external model inference or metrics |
| I — Reproducibility | CONTRACT PASS / EXECUTION PENDING | Clean-clone and artifact regeneration contract implemented |
| J — Claim audit | CONTRACT PASS / FINAL RENDER PENDING | Manuscript and reviewer response share one provenance registry |
| K — Submission | TEMPLATE PARTIAL | Scientific results, layout references and author declarations absent |

## 3. Cohorts

- Development: all 369 unique BraTS 2020 training patients.
- Development design: deterministic patient-level five-fold CV with validation
  sizes 74/74/74/74/73; stratified by grade, ET presence and WT burden.
- External release: processed BraTS-Africa TCIA v1, DOI
  `10.7937/V8H6-8X67`, CC BY 4.0.
- Primary external: 95 glioma patients.
- Supportive external: 51 other-neoplasm patients, excluded from confirmatory
  inference.
- Legacy 74-patient internal subset: not opened and prohibited for v2 models.

## 4. Overlap audit

All 146 external patients were compared with 369 development patients: 53,874
pairs. Identifier, raw file hash, exact normalized content, sampled content and
normalized-volume near-match counts were all zero. Minimum normalized-signature RMS
distance was `0.03083456` versus the frozen `1e-5` threshold. Gate C passes.

## 5. Model and seed matrix

The 12 models are U-Net-Small, parameter-matched U-Net, compute-matched U-Net,
U-Net+RES, U-Net+WC, BU-Net, ResBlock-U-Net, ResBlock-U-Net+WC, nnU-Net v2 2D,
nnU-Net v2 3D full-resolution, five-slice 2.5D U-Net and Swin UNETR.

Every model is assigned seeds `20260730`–`20260734` in all five folds. The
convergence matrix contains 300 runs. The eight core models add 200
compute-matched runs. Failed seeds cannot be silently replaced.

## 6. Parameter/compute matching

At input `1×4×240×240`:

| Control | Width/depth | Parameters | MAC/slice | Difference from U-Net+RES |
|---|---|---:|---:|---:|
| U-Net+RES target | 16/4 | 4,450,452 | 9,459,302,400 | — |
| Parameter-matched plain U-Net | 24/4 | 4,368,364 | 5,994,086,400 | 1.8445% parameters |
| Compute-matched plain U-Net | 30/4 | 6,823,774 | 9,348,480,000 | 1.1716% MAC |

Both controls meet the 2% static target. Real training memory, latency and
accelerator-hours remain pending, so Gate D is not fully passed.

## 7. Convergence

The authorized 15-job development-only loss screen is active on the companion
experiment branch at one immutable training commit. Its frozen rule allows up
to 10,000 optimizer steps per job and validation every 500 steps. The main
convergence rule allows up to 50,000 optimizer steps, minimum delta 0.001 and
patience 12 checks. Best and terminal checkpoints are both required. Legacy
2,000-step runs remain pilot evidence.

## 8. Primary external result

Does not exist. No model has produced external predictions. No effect,
confidence interval or p value is reported.

## 9. Hierarchical statistics

Not run. The plan prespecifies patient bootstrap, seed-then-patient
hierarchical bootstrap, paired sign-flip tests with Holm correction, and a
mixed-effects sensitivity. The practical interpretation threshold is 0.020
mean-regional Dice; it is not a clinical MCID or equivalence margin.

For n=95 and the legacy planning SD 0.05888, the expected 95% CI half-width is
0.0120 and approximate two-sided power for a 0.020 true paired difference is
0.906. This is planning evidence only.

## 10. Lesion-level results

No v2 lesion result exists. The frozen evaluator covers regional Dice/HD95/
surface Dice, rates, relative volume error, lesion recall/precision, lesion-wise
Dice/HD95 and false-positive lesion count. Sensitivities cover 1/10 voxels,
6/26 connectivity and 100 mm³. One-empty HD95 remains infinity.

## 11. Resource results

- Host: Apple M1 Max, 32 GPU cores, 32 GB unified memory.
- Backend: MPS available; CUDA unavailable.
- Terminology: MPS framework-reported allocated unified memory.
- Swin UNETR 64³ forward/backward smoke: PASS; 15,705,646 parameters.
- Free disk at preflight: approximately 395 GiB.
- Mandatory development runs before external inference: 615, including the
  15-job loss screen and 600-run frozen development matrix.
- Known proxy excluding nnU-Net: approximately 4,032 accelerator-hours or 168
  serial days.
- Unbenchmarked: 25 nnU-Net 2D and 50 nnU-Net 3D/interaction runs.

## 12. Negative outcomes

The negative feasibility result is retained: a single MPS device cannot
complete the mandatory design in a credible bounded execution. This is a
compute/scheduling blocker, not a failed MPS-operator test or evidence that any
architecture performs poorly.

## 13. Reproduction

Legacy v1 remains recoverable from its immutable tag. The legacy integrity test
now validates each tracked entry against that tag rather than against the
modified v2 working tree. No v2 full-training rerun, fold×seed reproduction or
external-inference reproduction is possible before training.

The realized Python 3.11/Apple dependency snapshot is frozen in
`environment/q1q2_v2-requirements-lock.txt` with machine-readable hashes in
`environment/q1q2_v2-environment.json`. Docker and Apptainer definitions are
present for audit/test reproduction. They are not represented as validated
CUDA training images; the cluster-specific image and lock must be frozen after
the allocation is known. Authorized-data setup instructions and the
superseded/unrealized legacy A100 boundary are also documented.

## 14. Tests

Executed on the updated environment:

- Ruff: PASS.
- Strict mypy: PASS for 76 source files.
- Pytest: **212 passed, 2 skipped**.
- The skips are the CUDA-only metric-invariance test and the integration check
  requiring final Gate G checkpoints.
- Real MPS matrix and Swin UNETR forward/backward smokes: PASS.

These results cover code/invariants, not segmentation accuracy or convergence.

## 15. License and release

Original repository code is Apache-2.0. `LICENSE`, `NOTICE`,
`THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `.zenodo.json`, contributor/security
documents, model/data cards, container definitions, an exact-version local
environment lock and prerelease notes are present. nnU-Net v2 2.8.1 and MONAI
1.6.0 are Apache-2.0 dependencies.

No v2 tag, GitHub Release or Zenodo DOI is created because the scientific v2
release is incomplete.

## 16. Target journal rationale

The verified scope matrix retains Medical Image Analysis as the leading
methodology target if the controlled result yields a transferable lesson, and
Radiology: Artificial Intelligence if independent domain-shift/failure
analysis becomes central. The conditional rule and current official
Radiology: Artificial Intelligence Original Research limits are recorded in
`submission/journal_target_rationale.md`. Current quartile was not guessed.
Final selection, APC, institutional agreement and live policies require
recheck after results.

## 17. Manuscript and supplement

A complete v2 manuscript template is present at
`manuscript/q1q2_v2_manuscript.template.md`. It contains no fabricated v2
result and binds all numerical Results text to the Gate J registry. A
23-table/15-figure supplementary template is present at
`manuscript/q1q2_v2_supplement.template.md` and uses the same registry. Final
Results, figures, tables, abstract effect estimates and the journal-formatted
documents remain unavailable until the 615 development runs and external
inference complete. Legacy manuscript files remain legacy evidence.

The Gate K CLAIM 2024 template covers all 44 guideline items: 27 are
Yes pending final page/line references, 11 are Partial pending results, two
remain No, and four are Not Applicable. It is not represented as a completed
submission checklist before final results, layout, and author confirmation.

A guarded pre-results Radiology: Artificial Intelligence package is also
available in `submission/generated/pre_results/`. It contains seven documents
in Markdown, DOCX, and PDF form, including the anonymized manuscript, title
page, cover letter, supplement, reviewer response, CLAIM checklist, and
data/code statement. The package passes the journal-format source audit and a
52-page visual layout audit, but its manifest is explicitly
`preview_not_for_submission`. Final-mode generation is blocked by Gates H–J,
confirmed author metadata, real page/line references, unresolved result
markers, the live journal-policy recheck, and the supplement page limit.

## 18. Reviewer response

A 19-concern v2 response template is present at
`manuscript/q1q2_v2_response_to_reviewer.template.md`. The audit requires each
concern exactly once, all response fields, existing repository evidence, and
the same Gate J result registry used by the manuscript. A final response is not
claimed: result substitutions and page/line references remain blocked until
the external result and final layout exist.

## 19. Missing author declarations

Author order/affiliations, CRediT roles, funding, conflicts, ethics/waiver
interpretation, corresponding-author details, acknowledgements, checkpoint
distribution permission and journal-specific AI disclosure require author
confirmation. They were not invented.

## 20. Final decision

**NOT READY — M1 DEVELOPMENT IN PROGRESS**

The user has authorized execution on the selected Apple M1 Max. No immediate
user action is required for the active bounded loss screen. The frozen full
matrix remains a multi-month serial MPS workload; five patient folds and five
common seeds cannot be silently reduced. Gates F–H, the external result, final
result-bearing manuscript, figures/tables, reviewer-response substitutions,
and genuinely submission-ready package remain incomplete until real artifacts
exist. The generated pre-results package is a format-verified preview and
cannot satisfy Gate K.
