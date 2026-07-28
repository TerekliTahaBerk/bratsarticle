# Standard 2D U-Net Baseline Specification

**Gate:** 5  
**Status:** PASS for implementation and bounded diagnostics  
**Full cohort training:** Not started

## Architecture

The baseline is a plain 2D U-Net following the contracting path, symmetric
expanding path, and skip-concatenation pattern introduced by
[Ronneberger, Fischer, and Brox](https://arxiv.org/abs/1505.04597).

The reportable configuration uses:

- four input channels in T1/T1ce/T2/FLAIR order;
- four output logits for background and BraTS labels 1, 2, and 4;
- four encoder levels;
- 32 base channels, doubled at each level;
- two same-padded 3-by-3 convolutions plus ReLU per block;
- 2-by-2 max pooling;
- transposed-convolution upsampling and skip concatenation;
- a final 1-by-1 classifier.

No residual block, RES component, WC component, attention, transformer, or
other proposed module is present. Same padding is an explicit implementation
choice relative to the cropped valid-convolution layout in the original U-Net.

## Optimization contract

The baseline configuration is
`configs/training/unet2d_baseline.yaml`. It provisionally declares AdamW,
learning rate 0.001, weight decay 0.00001, and a 0.5/0.5 weighted sum of
four-class cross-entropy and foreground soft Dice loss. The training Dice
smooth is 0.00001 and affects only optimization; the central evaluator uses no
Dice smoothing.

Automatic mixed precision is configurable. CUDA uses float16 autocast with
gradient scaling; CPU diagnostic mode uses bfloat16 autocast without gradient
scaling. PyTorch documents autocast and gradient scaling as the standard AMP
components
([PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)).

## Reproducibility and recovery

Python, NumPy, PyTorch CPU/CUDA, and DataLoader workers are seeded. Deterministic
algorithms are enabled and cuDNN benchmarking is disabled, consistent with
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness).

Atomic checkpoints contain:

- model and optimizer state;
- AMP scaler state;
- epoch and global step;
- Python, NumPy, PyTorch CPU, and CUDA RNG state;
- run metadata.

A unit test compares uninterrupted two-step training with a one-step
checkpoint followed by resume. Loss and every model-state tensor are
bit-identical on the current CPU environment. A real one-step CLI smoke was
also resumed to step two; its metadata records the resume source.

Every run metadata file records the configuration hash, train/validation split
hashes, Git commit, seed, Python/package versions, device, CPU/memory
information, CUDA/MPS availability, mixed-precision state, run kind, and
completion status.

## Validation path

Validation enumerates every slice, reconstructs each complete patient volume,
and calls the Gate 3 central evaluator. A synthetic integration test produces a
perfect two-slice volume and obtains mean regional Dice 1.0 and WT HD95 0 mm
through that path. Any batch marked `test` is rejected by the validation
function.

## Controlled real-data overfit diagnostic

The diagnostic script automatically chose the slice with the largest ET voxel
count from the first provisional training subject. The selected slice contains
all three target regions:

| Field | Value |
|---|---:|
| Subject | `BraTS20_Training_002` |
| Slice | 45 |
| WT target voxels | 1,489 |
| TC target voxels | 539 |
| ET target voxels | 238 |
| Steps | 200 |
| Initial loss | 1.2698 |
| Final loss | 0.0644 |
| Initial mean regional Dice | 0.0230 |
| Final mean regional Dice | 0.9208 |
| Final WT Dice | 0.9369 |
| Final TC Dice | 0.9523 |
| Final ET Dice | 0.8733 |

All predeclared acceptance conditions passed: loss fell by more than 50%,
mean regional Dice improved, and final WT/TC/ET Dice each exceeded 0.80.
The machine-readable source is
`reports/gate5_real_overfit_metrics.json`, generated from Git commit
`09f516284753dfcf9fa5e67c842fa74ad65894ec`.

This diagnostic demonstrates pipeline capacity to memorize one real training
slice. It is not an estimate of validation performance, generalization,
clinical utility, or model superiority.

## Full-training guard

The training CLI requires `--allow-full-training` for a non-smoke run. The
current protocol additionally requires CUDA for full training. The present
Apple M1 Max host exposes neither CUDA nor available MPS through the installed
PyTorch build, so no full-cohort training was started. Bounded smoke steps
remain permitted and never load the internal-test manifest.

