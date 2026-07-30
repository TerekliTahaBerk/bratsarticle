# Official nnU-Net v2 baseline protocol

Status: infrastructure prepared; training not started

This protocol covers the required official plain nnU-Net v2 2D and residual
encoder 3D full-resolution baselines. It uses all 369 BraTS 2020 development patients,
the frozen five patient-level folds, and the same five seeds used by every
main model. It does not access the legacy 74-patient internal subset or the
external cohort.

## Scientific boundary

The adapter changes data layout and label encoding only. It does not copy or
modify raw MRI data. The four raw modality files are represented by relative
symbolic links in a derived nnU-Net directory. Segmentations are copied to the
derived directory after the official BraTS mapping:

| Meaning | BraTS raw | nnU-Net training/export |
|---|---:|---:|
| Background | 0 | 0 |
| NCR/NET | 1 | 2 |
| Edema | 2 | 1 |
| Enhancing tumor | 4 | 3 |

The inverse mapping is applied before the common repository evaluator. The
nnU-Net region definitions are WT `(1,2,3)`, TC `(2,3)`, and ET `(3)`, with
region class order `(1,2,3)`, following the official nnU-Net BraTS converter.

The seed subclasses preserve the selected official nnU-Net architecture,
loss, optimizer, learning-rate schedule, augmentation definitions, and
default 1,000-epoch duration. Each epoch contains 250 optimizer steps. This
means a completed default run contains 250,000 optimizer steps; it must not be
described as compute-matched to the native 50,000-step ceiling. The official
run is a strong full-duration baseline. Compute-matched nnU-Net claims require
a separately frozen and validated trainer budget.

## Derived local roots

All generated data remain ignored by Git:

```bash
export REPO_ROOT=/path/to/bratsarticle
export BRATS2020_ROOT=/authorized/BraTS2020_TrainingData
export NNUNET_WORK_ROOT="$REPO_ROOT/work/q1q2_v2/nnunet"
export nnUNet_raw="$NNUNET_WORK_ROOT/raw"
export nnUNet_preprocessed="$NNUNET_WORK_ROOT/preprocessed"
export nnUNet_results="$NNUNET_WORK_ROOT/results"
export nnUNet_extTrainer="$REPO_ROOT/nnunet_ext"
export nnUNet_n_proc_DA=0
export nnUNet_compile=false
```

Single-process augmentation is mandatory for the seeded M1 Max runs because
upstream nnU-Net otherwise selects `NonDetMultiThreadedAugmenter`. MPS
deterministic algorithms use warning-only mode because the installed PyTorch
version lacks a deterministic implementation for at least one required MPS
operator. Tolerance-based fresh-run reproduction is therefore required; no
bitwise MPS claim is permitted.

## Audited preparation

Reportable preparation recomputes every source SHA-256 from the canonical
manifest. The optional skip flag is for local diagnostics only.

```bash
cd "$REPO_ROOT"
PYTHONPATH=src .venv/bin/python scripts/prepare_q1q2_nnunetv2.py \
  --dataset-root "$BRATS2020_ROOT" \
  --nnunet-raw-root "$nnUNet_raw" \
  --nnunet-preprocessed-root "$nnUNet_preprocessed"
```

The command is idempotent. An existing mismatch is rejected rather than
overwritten. `derivation_manifest.json` records source and derived hashes
without an absolute raw-data path. `splits_final.json` is generated from the
five tracked fold CSVs and uses nnU-Net folds `0..4` for repository folds
`1..5`.

## Planning and preprocessing

Planning first records the legacy plain default because the installed official
tool requests this baseline before a custom memory target. The tool itself
warns that this planner is no longer the recommended benchmarking default.
Both requested configurations are integrity-checked:

```bash
nnUNetv2_plan_and_preprocess \
  -d 501 \
  --verify_dataset_integrity \
  -c 2d 3d_fullres \
  -npfp 4 \
  -np 4 2
```

Before any model result exists, the official ResEnc-L and ResEnc-M presets are
also planned without changing their memory targets:

```bash
nnUNetv2_plan_experiment -d 501 -pl nnUNetPlannerResEncL
nnUNetv2_plan_experiment -d 501 -pl nnUNetPlannerResEncM
```

The official documentation recommends ResEnc-L as the new default and
ResEnc-M for a budget similar to the standard U-Net. The predeclared 3D choice
is therefore ResEnc-L, with ResEnc-M allowed only if the untouched ResEnc-L
plan fails a one-batch MPS feasibility test. Published NVIDIA memory targets
are not reported as M1 memory measurements. Selection never uses Dice, loss,
or convergence behavior.

After preprocessing and while no other MPS queue lock exists, the real
one-step feasibility command is:

```bash
PYTHONHASHSEED=20260730 PYTHONPATH=src \
  .venv/bin/python scripts/run_q1q2_nnunetv2_mps_preflight.py \
  --configuration 3d_fullres \
  --plans-identifier nnUNetResEncUNetLPlans \
  --assert-no-active-mps-lock \
    artifacts/q1q2_v2/queue_runtime/loss_screen.lock \
  --output work/q1q2_v2/nnunet/preflight/resenc_l_3d.json
```

The command uses an actual preprocessed training batch and the official
training loss/backward/optimizer step. It polls framework- and driver-reported
MPS allocated unified memory. It is a hardware diagnostic, not a model
evaluation, and it neither predicts nor computes metrics.

The resulting fingerprint, plans, patch shapes, batch sizes, parameter counts,
and estimated preprocessing disk use are frozen before training. A one-batch
forward/backward MPS feasibility run is required for each configuration.
Failure of ResEnc-L activates only the predeclared official ResEnc-M fallback.
Failure of ResEnc-M is a model feasibility blocker; the GPU-memory target is
not silently changed.

## Frozen job matrix

The job generator creates 50 jobs: two configurations × five folds × five
seeds. The 25 3D jobs remain hardware-blocked until ResEnc-L or the
predeclared ResEnc-M fallback passes.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/generate_q1q2_nnunetv2_jobs.py
```

Each job encodes its trainer, zero- and one-indexed fold, seed, device,
environment, source manifest hash, and five split hashes. Example:

```bash
PYTHONHASHSEED=20260730 nnUNetv2_train \
  501 2d 0 \
  -tr nnUNetTrainerSeed20260730 \
  -p nnUNetPlans \
  -device mps
```

Do not add `--c` to a reportable run without recording an interruption.
Upstream nnU-Net checkpoints do not serialize every augmentation RNG state, so
continuation is not claimed to be bitwise equivalent to uninterrupted
training.

## Acceptance before main execution

Training may enter the main queue only after all of the following are true:

1. Dataset integrity and exact 369-case identity pass.
2. `splits_final.json` matches all five tracked folds byte-for-byte after
   canonical serialization.
3. Default, ResEnc-L, and ResEnc-M plans have recorded hashes.
4. One-batch MPS forward/backward passes for plain 2D and either ResEnc-L 3D
   or the predeclared ResEnc-M 3D fallback.
5. Measured step time, peak MPS allocated unified memory, checkpoint size, and
   total serial-hours estimate are recorded.
6. The running native loss screen has completed or is deliberately paused so
   feasibility profiling cannot contaminate its timing.
7. The selected architecture-attribution loss is frozen from development CV.

No external prediction or metric computation is authorized by this protocol.
