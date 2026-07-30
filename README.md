# bratsarticle

Reproducible research pipeline for leakage-safe, patient-level evaluation of
U-Net-family models on multimodal BraTS glioma segmentation data.

## Current status

The completed bounded internal study is preserved at the immutable
`v1-bounded-2d-component-study` tag. Its Gates 0–14, manuscript, figures, and
74-patient internal evaluation are legacy evidence only.

The active Q1/Q2 v2 study is not yet a completed paper. Development-only loss
screening is running on the frozen Apple M1 Max configuration. The v2 design
requires 600 development runs, 300 frozen main checkpoints in the single
external session, all 12 models, five patient folds, and the same five seeds.
No v2 performance, external-generalization, clinical-utility, or superiority
claim exists until the corresponding artifacts pass Gates G–J. New inference
on the legacy 74-patient subset is prohibited.

Legacy v1 submission artifacts:

- [`manuscript/final_manuscript.docx`](manuscript/final_manuscript.docx)
- [`manuscript/final_manuscript.pdf`](manuscript/final_manuscript.pdf)
- [`manuscript/final_manuscript.tex`](manuscript/final_manuscript.tex)
- [`manuscript/response_to_reviewer.docx`](manuscript/response_to_reviewer.docx)
- [`manuscript/response_to_reviewer.pdf`](manuscript/response_to_reviewer.pdf)
- [`manuscript/claim_2024_checklist.md`](manuscript/claim_2024_checklist.md)
- [`reports/gate14_completion.md`](reports/gate14_completion.md)

The initial audit is available at
[`reports/phase0_repository_audit.md`](reports/phase0_repository_audit.md).
The verified data audit is available at
[`reports/data_audit_summary.md`](reports/data_audit_summary.md).
The provisional split audit is available at
[`reports/split_balance_report.md`](reports/split_balance_report.md).
The evaluator contract is available at
[`reports/evaluator_specification.md`](reports/evaluator_specification.md).
The preprocessing contract is available at
[`reports/preprocessing_specification.md`](reports/preprocessing_specification.md).
The baseline contract is available at
[`reports/unet2d_baseline_specification.md`](reports/unet2d_baseline_specification.md).
The BU-Net fidelity decisions are available at
[`reports/bunet_implementation_notes.md`](reports/bunet_implementation_notes.md),
and the generated model/loss summaries are
[`reports/gate6_model_summary.md`](reports/gate6_model_summary.md) and
[`reports/gate6_loss_methods.md`](reports/gate6_loss_methods.md).
The frozen fairness and registry contract is
[`reports/fairness_and_registry_protocol.md`](reports/fairness_and_registry_protocol.md).
The complete gate history is recorded under [`reports/`](reports/), and the
verified primary literature sources used by the rebuilt manuscript are in
[`literature/verified_sources.yaml`](literature/verified_sources.yaml).

The v2 gate state is
[`reports/q1q2_v2/gate_status.yaml`](reports/q1q2_v2/gate_status.yaml), its
verified literature ledger is
[`literature/q1q2_verified_sources.yaml`](literature/q1q2_verified_sources.yaml),
and Gate I/J provenance controls are documented in
[`docs/q1q2_v2/gates_i_j.md`](docs/q1q2_v2/gates_i_j.md).

## Scientific scope

The core study will:

- treat BraTS 2020 training subjects as the canonical labeled cohort;
- use BraTS 2019 only for identity and duplicate auditing;
- use patient-level five-fold development without a new internal held-out
  claim;
- reserve BraTS-Africa for one frozen external session after Gate G;
- evaluate RES and WC as previously published BU-Net components, not as novel
  modules;
- generate metrics, tables, and figures from machine-readable experiment
  artifacts;
- prevent test-set use during training, model selection, threshold selection,
  and post-processing development.

## Data safety

Raw MRI and segmentation files are not versioned and must never be modified,
renamed, or written in place. Dataset roots will be supplied through
environment variables:

```text
BRATS2020_ROOT
BRATS2019_ROOT
BRATS_CACHE_ROOT
```

The `data/` directory and medical-image files are excluded from Git.

## Development

Create the locked Python 3.11 environment:

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r environment/requirements-lock.txt
./.venv/bin/pip install -e .
```

Run code-quality checks:

```bash
./.venv/bin/ruff format --check src tests
./.venv/bin/ruff check src tests
./.venv/bin/mypy src/bratsarticle
./.venv/bin/pytest -q
```

Run the read-only data audit after exporting the dataset roots:

```bash
./.venv/bin/brats-data-audit --config configs/data/audit.yaml
```

Regenerate the provisional patient-level split:

```bash
./.venv/bin/brats-generate-split --config configs/data/split.yaml
```

Run a bounded baseline smoke:

```bash
BRATS2020_ROOT=/authorized/path/to/brats2020 \
./.venv/bin/brats-train-unet2d \
  --config configs/training/unet2d_gate5_smoke.yaml \
  --smoke-steps 1
```

Reportable v2 training requires an eligible frozen host and its explicit
runner authorization. External inference remains separately guarded until
Gate G; legacy internal held-out inference is prohibited for v2.

Regenerate the manuscript Markdown and audit its exact long-phrase overlap
against the user-supplied source materials:

```bash
python3 scripts/generate_gate14_manuscript.py
python3 scripts/audit_gate14_originality.py
```

Build DOCX and LaTeX sources with the document runtime, then compile the PDF
with Tectonic:

```bash
python3 scripts/build_gate14_documents.py --target all
```

The manuscript generator reads scientific values from tracked CSV/JSON
artifacts. It must not be replaced with manual result transcription.
