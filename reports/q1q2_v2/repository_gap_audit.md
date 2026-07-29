# Q1/Q2 v2 repository gap audit

Audit date: 2026-07-30
Audited commit: `ab60a79d8a49a2fe1adb000546f653485796bab1`
Working branch: `q1q2-capacity-external-validation`
Target semantic version: `v2.0.0`

## Decision

Gate A passes: the complete Gate 0–14 study is recoverable from the immutable
tag `v1-bounded-2d-component-study`, and its principal integrity anchors are
recorded under `legacy/v1/`.

Gate B passes as an audit: every material reviewer/prompt concern found in the
repository is mapped below. Gate C now passes on official BraTS-Africa TCIA v1
data. MPS is available on the Apple M1 Max, and Apache-2.0 resolves the code
license. The repository is still **not ready for full v2 training** because the
mandatory 615-run matrix requires at least 4,032 measured/extrapolated serial
accelerator-hours before the unbenchmarked nnU-Net runs.

## Repository state

| Check | Evidence | Status |
|---|---|---|
| Exact legacy commit recorded | `ab60a79d8a49a2fe1adb000546f653485796bab1` | PASS |
| Immutable annotated tag | `v1-bounded-2d-component-study` resolves to the legacy commit | PASS |
| Dedicated v2 branch | `q1q2-capacity-external-validation` | PASS |
| Worktree before v2 audit | Clean | PASS |
| Legacy integrity manifest retained | Frozen tag plus SHA-256 anchors in `legacy/v1/` | PASS |
| Separate v2 artifact namespace | `artifacts/q1q2_v2/` designated | PASS |
| External cohort locally configured | 146 complete BraTS-Africa patients; 95 primary glioma | PASS |
| External overlap | 0 matches across 53,874 patient pairs | PASS |
| Suitable accelerator visible | Apple M1 Max MPS smoke and Swin UNETR forward/backward pass | PASS |
| Full mandatory compute allocation | One MPS device; ≥4,032 h proxy excluding nnU-Net | BLOCKED |
| Final software license | Apache-2.0 plus NOTICE/third-party notices | PASS |
| Local environment snapshot | 114-line exact-version lock plus machine-readable hashes | PASS |
| Container definitions | Docker and Apptainer audit/test definitions; CUDA promotion pending allocation | PARTIAL |

## Legacy study inventory

The audited repository contains patient-level data discovery and split code,
preprocessing, evaluator tests, 2D U-Net-family models, seven loss
configurations, Gate 8 single-seed screening, Gate 9 multi-seed development,
Gate 10 statistical freezing, one guarded Gate 11 legacy internal-test access,
artifact-derived figures/tables, reproducibility reports, and final manuscript
files. These assets remain usable as implementation provenance and legacy
evidence, subject to the boundaries below.

The canonical cohort is 369 BraTS 2020 labeled patients. The legacy split is
258/37/74. BraTS 2019 was used for identity auditing; all 335 BraTS 2019 cases
overlap the canonical cohort and therefore do not form an independent external
cohort. The 74-patient subset cannot be reused for v2 model evaluation.

## Gaps mapped to v2 tasks

| Gap | Repository evidence | Required v2 task | Gate |
|---|---|---|---|
| No independent confirmatory cohort | Resolved with official BraTS-Africa v1 and zero-overlap audit | Keep results locked until Gate G | C PASS |
| No capacity-matched reference | Plain width-24/depth-4 U-Net is 1.8445% from RES parameters | Train the frozen control in all folds/seeds | D PARTIAL |
| No compute-matched reference | Plain width-30/depth-4 U-Net is 1.1716% from RES MAC | Train/profile the frozen control under the common budget | D PARTIAL |
| Unequal replication | Common seeds `20260730`–`20260734` frozen for all 12 models/folds | Complete every common fold×seed run; no substitutions | E DESIGN PASS |
| Bounded runs are not convergence evidence | Legacy runs stop at 2,000 steps/0.5 h | Define convergence-matched and compute-matched regimes with prespecified stopping diagnostics | F |
| No external statistical freeze | Gate 10 applies only to the legacy internal analysis | Freeze v2 endpoint, contrasts, multiplicity and missingness before external access | G |
| No external result | Legacy test is internal and already opened | Make one audited external opening after Gates C–G | H |
| Model matrix was incomplete | Official nnU-Net v2 2.8.1 and MONAI 1.6 are installed; 2.5D extraction and Swin MPS smoke pass | Generate nnU-Net plans and execute all frozen runs on an allocated cluster | Design resolved; D–F execution pending |
| Resource terminology was device-inaccurate | Legacy aliases remain for compatibility; v2 registry now adds backend-neutral fields | Use `accelerator_hours` and backend-neutral memory fields in every new report | I DESIGN PASS |
| License unresolved | Resolved as Apache-2.0 under explicit rights-holder authorization | Retain dependency and data notices | I/K PASS |
| Qualitative language overstates prespecification | Manuscript calls identities “prespecified” | State that cases were selected after evaluation using prespecified deterministic rules | J |
| Response letter overstates table contents | It lists checkpoint size, throughput, and reserved memory not present in the manuscript table | Correct the response or add artifact-derived columns in a future manuscript | J |
| A100/MPS protocol mismatch | Gate 7 configs target A100; realized Gate 8–11 runs used Apple MPS | Resolved by the superseded legacy note and separate q1q2 v2 hardware protocol | I DESIGN PASS |
| No final author/declaration package | No confirmed target journal or author confirmations for v2 | Obtain author declarations after scientific gates pass | K |

## Exact legacy model/resource facts

All values below are artifact-derived at input `4 x 240 x 240` per slice:

| Candidate | Parameters | MAC/slice | FLOP/slice | Legacy seeds |
|---|---:|---:|---:|---:|
| Standard 2D U-Net | 1,942,772 | 2,676,326,400 | 5,352,652,800 | 3 |
| U-Net+RES | 4,450,452 | 9,459,302,400 | 18,918,604,800 | 5 |
| BU-Net | 8,384,660 | 10,344,038,400 | 20,688,076,800 | 5 |

These differences prohibit interpreting the legacy U-Net+RES comparison as a
capacity-controlled component effect.

## Access and compute preflight

- Disk available at updated preflight: approximately 395 GiB.
- Python: 3.11.14.
- PyTorch: 2.13.0 in the current virtual environment.
- CUDA: unavailable.
- MPS: built and available outside the application sandbox on Apple M1 Max.
- Swin UNETR 64³ forward/backward smoke: PASS.
- External release: verified outside Git; external result access remains zero.
- Full matrix: blocked at 615 runs and at least 4,032 accelerator-hours before
  nnU-Net timing.
- No full v2 training, external prediction evaluation, or legacy internal-test
  access was performed during this audit.

## Gate B task disposition

Every issue above has a target gate and correction entry. Gate B passes as a
repository and claim audit; Gate C also passes. These do not waive the active
full-compute blocker or authorize external inference.
