# Phase 0 Repository and Environment Audit

**Gate:** Gate 0 — Repository audit

**Audit date:** 2026-07-29 (Europe/Istanbul)

**Workspace:** `/Users/berk.terekli/Documents/braintumorarticle`

**Status:** Completed with prerequisites and training-environment blockers

## 1. Scope and safety statement

This audit was limited to read-only inspection of the existing workspace, runtime,
hardware, documents, figures, and dataset directory structure. No MRI volume,
segmentation, CSV metadata file, existing figure, LaTeX source, or PDF was
modified, renamed, or moved.

The only artifact created during Gate 0 is this report under `reports/`.

## 2. Executive summary

The workspace is a data-and-manuscript folder rather than an initialized
scientific software repository. It contains approximately 78 GB of data, one
LaTeX source, one compiled manuscript PDF, and 17 legacy figure/reference
assets. It contains no Python source, notebooks, tests, experiment
configurations, dependency declaration, `.gitignore`, or Git metadata.

The local data are sufficient to begin a CPU-based read-only inventory:

- BraTS 2019 contains 335 conventionally named segmentation files.
- BraTS 2020 training contains 369 subject directories.
- BraTS 2020 validation contains 125 subject directories.
- A separate BraTS 2021 directory occupies approximately 12 GB but is outside
  the core-paper cohort and was not inspected beyond top-level inventory.

The machine is an Apple M1 Max system with 32 GB unified memory. No NVIDIA
CUDA runtime or GPU is available. PyTorch 2.11.0 is installed in the default
Python 3.12 environment; its MPS backend is built but not currently available
to the process. CPU tensor execution succeeds. This environment is suitable
for manifest discovery, hashing, NIfTI metadata inspection, evaluator unit
tests, and small smoke tests, but not for definitive full-model training.

## 3. Existing workspace tree

Top-level existing items before this report was created:

```text
.
├── .DS_Store
├── data/
│   ├── brats2019/
│   ├── brats2020/
│   └── brats2021/
├── figuresandfiles/
├── main (10).pdf
└── main (3).tex
```

Observed counts before adding this report:

| Item | Count |
|---|---:|
| Total files | 4,049 |
| Files under `data/` | 4,029 |
| Files under `figuresandfiles/` | 17 |
| Python source files | 0 |
| Jupyter notebooks | 0 |
| YAML/JSON/TOML configuration files | 0 |

## 4. Repository state

- The workspace is **not a Git repository**.
- There is no `.gitignore`.
- There is no `pyproject.toml`, requirements file, environment file, lockfile,
  container definition, package configuration, or experiment registry.
- Git commit, dirty-state, configuration-hash, and artifact provenance cannot
  yet be recorded.

### Required action before scientific code is committed

1. Create `.gitignore` before repository initialization or staging.
2. Explicitly exclude `data/`, caches, checkpoints, artifacts, credentials,
   local environment files, and generated manuscript build intermediates.
3. Initialize version control only after the exclusion rules are reviewed.
4. Never add the 78 GB raw-data tree to Git.

## 5. Python and dependency state

### Python 3.11

- Executable: `/opt/homebrew/bin/python3.11`
- Version: 3.11.14
- Available scientific packages include NumPy, SciPy, pandas, and
  scikit-learn.
- PyTorch, MONAI, nibabel, OmegaConf/Hydra, pytest, ruff, mypy, and
  TensorBoard are not installed in this interpreter.

### Default Python 3.12

- Executable: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`
- Version: 3.12.10
- Installed relevant packages:
  - PyTorch 2.11.0
  - nibabel 5.4.2
  - NumPy 2.4.4
  - SciPy 1.17.1
  - pandas 3.0.2
  - scikit-learn 1.8.0
  - pytest 9.0.3
- Missing relevant packages:
  - MONAI
  - OmegaConf/Hydra
  - ruff
  - mypy/pyright
  - TensorBoard

The requested stack specifies Python 3.11. A dedicated Python 3.11 environment
and a locked dependency declaration should therefore be created before the
training pipeline is implemented. Gate 1 data discovery could technically run
with the existing Python 3.12 environment, but mixing ad-hoc interpreters would
weaken reproducibility and is not recommended.

No `uv`, conda, mamba, pyenv, or Poetry executable was found. `pip3` is
available.

## 6. Compute environment

| Component | Observed state |
|---|---|
| Host | Apple MacBook Pro |
| CPU/SoC | Apple M1 Max, 10 CPU cores |
| Memory | 32 GB unified memory |
| Integrated GPU | Apple M1 Max, 32 GPU cores |
| NVIDIA GPU | Not present |
| `nvidia-smi` | Not available |
| CUDA build in PyTorch | No |
| CUDA available | No |
| PyTorch MPS built | Yes |
| PyTorch MPS available | No |
| PyTorch CPU tensor smoke test | Passed |

### Consequence

- Read-only data audit and CPU-based correctness tests are feasible locally.
- Small model forward/backward and overfit smoke tests may be feasible on CPU,
  subject to measured runtime.
- Compute-matched, convergence-matched, multi-seed, nnU-Net, or transformer
  training must not be scheduled on this host as the definitive experiment
  environment.
- A consistent NVIDIA training environment must be selected and recorded
  before Gate 5 full baseline work or any later full training.
- Full training remains prohibited until explicit approval regardless of
  hardware availability.

## 7. Dataset inventory

### Storage

| Path | Approximate size |
|---|---:|
| Entire workspace | 78 GB |
| `data/` | 78 GB |
| `data/brats2019/` | 26 GB |
| `data/brats2020/` | 40 GB |
| `data/brats2021/` | 12 GB |
| `figuresandfiles/` | 4.4 MB |

The host has approximately 467 GiB free on the relevant filesystem.

### Top-level cohort observations

| Dataset area | Observed count |
|---|---:|
| BraTS 2019 conventionally named `*_seg.nii` files | 335 |
| BraTS 2020 training subject directories | 369 |
| BraTS 2020 validation subject directories | 125 |
| BraTS 2021 files | 3 |

BraTS 2020 training includes `name_mapping.csv` and `survival_info.csv`.
BraTS 2020 validation includes `name_mapping_validation_data.csv` and
`survival_evaluation.csv`. BraTS 2019 includes `name_mapping.csv` and
`survival_data.csv`.

These counts are preliminary directory-level observations, not a completed
data-integrity conclusion. Gate 1 must independently verify modalities,
segmentations, identities, labels, hashes, shapes, affines, and duplicate
relationships.

### Environment variables

The required variables are not currently set:

- `BRATS2020_ROOT`
- `BRATS2019_ROOT`
- `BRATS_CACHE_ROOT`

No source code should embed the absolute paths observed during this audit.

### Raw-data immutability risk

The dataset directories are technically writable by the current OS user.
Scientific policy must therefore enforce logical read-only behavior:

- inventory code must open inputs read-only;
- outputs and caches must resolve outside `data/`;
- startup checks must reject an output/cache path nested under a raw-data root;
- no rename, normalization-in-place, or metadata rewrite operation is allowed.

## 8. Existing manuscript and visual assets

### Manuscript

- LaTeX source: `main (3).tex`
- Compiled PDF: `main (10).pdf`
- PDF pages: 10
- PDF page size: US Letter
- Bibliography items embedded in LaTeX: 18
- Sections: 6
- Figures declared: 11
- Tables declared: 5

### Figure assets

There are 17 existing figure/reference assets under `figuresandfiles/`.
They include performance charts, training curves, architecture diagrams, and
one unrelated/reference PDF.

These assets are legacy, unverified artifacts. They must not be copied into the
new result pipeline or treated as evidence until their source data and
generation process are verified.

### Rebuild blocker

The LaTeX source declares `\graphicspath{{figures/}}` and references files
under `figures/`, while the current workspace contains `figuresandfiles/` and
filenames with suffixes such as ` (1)`. The current LaTeX source tree therefore
does not reproduce the existing PDF without path/file reconciliation.

This is a manuscript-support issue for a later gate. It must not be resolved by
silently renaming or overwriting legacy assets.

### Document toolchain

- `tectonic` is available.
- `pandoc` is available.
- `pdflatex` and `latexmk` were not found in the active shell path.

## 9. Write-access boundaries

Observed OS-level write access:

- Workspace root: writable.
- `data/`: writable, but scientifically immutable by policy.
- `figuresandfiles/`: writable, but legacy assets should remain untouched.
- `/tmp`: writable.

The research implementation should write only to controlled repository
directories such as `manifests/`, `reports/`, `splits/`, `artifacts/`,
`figures/`, and a separately configured cache root.

## 10. Gate 0 risks and blockers

### Blocking before reproducible implementation

1. **No Git repository:** commit provenance and clean/dirty state are
   unavailable.
2. **No ignore policy:** raw data could be staged accidentally if Git were
   initialized prematurely.
3. **No locked Python 3.11 environment:** the requested stack is split across
   incompatible/ad-hoc interpreters.
4. **No config system:** required dataset roots are unset and no path-safe
   configuration exists.
5. **No NVIDIA execution environment:** definitive training cannot run
   locally.

### Non-blocking for a CPU read-only Gate 1 audit

1. Python 3.12 already has nibabel, NumPy, pandas, and pytest.
2. There is sufficient free disk space for small CSV/JSON manifests and audit
   reports.
3. The BraTS 2019 and 2020 directory structures and mapping CSV files are
   present.

### Scientific risks to preserve in later gates

1. Raw directories are writable despite the immutability requirement.
2. The BraTS 2021 directory is present but is outside the approved canonical
   cohort.
3. Legacy figures and old manuscript values lack reproducible provenance.
4. The existing LaTeX source has unresolved graphics paths.
5. Internal test access logging cannot be enforced until repository code and
   manifests exist.

## 11. Proposed next action after Gate 0 approval

On explicit approval to enter Gate 1:

1. Create the requested repository scaffold and immutable scientific rules in
   `AGENTS.md`.
2. Add `.gitignore`, `.env.example`, `pyproject.toml`, data-usage and
   reproducibility documentation before initializing/staging Git content.
3. Establish a dedicated Python 3.11 environment and locked dependencies.
4. Implement a read-only, config-driven inventory command with path-safety
   checks and controlled segmentation filename fallbacks.
5. Test the inventory on a very small subject subset.
6. Run the complete read-only BraTS 2020/2019 Gate 1 audit only after the smoke
   test passes.
7. Produce the required manifests and audit summaries without altering raw
   data.

No split, evaluator, model implementation, training, or internal-test
evaluation should begin during Gate 1.
