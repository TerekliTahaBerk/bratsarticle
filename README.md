# bratsarticle

Reproducible research pipeline for leakage-safe, patient-level evaluation of
U-Net-family models on multimodal BraTS glioma segmentation data.

## Current status

Gates 0-14 are complete. The repository now contains the audited data and
patient split, tested U-Net-family implementations, matched development runs,
multi-seed confirmation, a frozen analysis plan, one guarded internal
held-out test evaluation, artifact-derived figures and tables, a clean-clone
reproduction audit, and the rebuilt manuscript package.

The primary internal result is deliberately narrow: under the bounded 2D
protocol, both BU-Net and U-Net+RES improved patient-level mean regional Dice
over standard U-Net, while U-Net+RES slightly exceeded the full BU-Net and
used fewer measured resources. The study does not establish clinical utility,
external generalization, state of the art, or superiority over untested 3D,
transformer, or self-configuring systems.

Final submission artifacts:

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

## Scientific scope

The core study will:

- treat BraTS 2020 training subjects as the canonical labeled cohort;
- use BraTS 2019 only for identity and duplicate auditing;
- enforce patient-level train, validation, and internal held-out test splits;
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

Full training requires both a suitable CUDA host and the explicit
`--allow-full-training` flag.

Full training and internal held-out test evaluation remain separately guarded
by the protocol and test-access audit.

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
