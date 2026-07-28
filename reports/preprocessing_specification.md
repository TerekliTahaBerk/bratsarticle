# Preprocessing and Slice-Loader Specification

**Gate:** 4  
**Status:** PASS  
**Configuration:** `configs/data/preprocessing.yaml`

## Input contract

Each patient is loaded from four co-registered MRI volumes in the fixed channel
order:

1. T1
2. T1ce
3. T2
4. FLAIR

The segmentation remains an integer `{0,1,2,4}` mask. Runtime loading checks
that all modalities and the label share the same shape, spacing, and affine.
Manifest paths are resolved below the audited BraTS 2020 training root and
path traversal is rejected.

## Intensity preprocessing

Each patient and modality is normalized independently. Mean and population
standard deviation are calculated only from nonzero voxels of that modality;
zero background remains zero. No train-, validation-, or test-cohort
normalization statistic is fitted.

Percentile clipping is disabled. The code permits clipping only through fixed
per-modality numeric bounds whose provenance is explicitly
`development_train_only_fixed_bounds`. Enabling clipping without four complete
fixed bounds fails configuration validation. No internal-test statistic can
therefore be used to derive a preprocessing rule.

## Training sampling and augmentation

The provisional 2D training policy samples tumor-containing slices with
probability 0.67 and non-tumor slices with probability 0.33, with 16 samples
per patient per epoch. These values are configuration parameters, not hidden
constants. If a requested pool is absent, sampling falls back to the available
pool rather than dropping the patient.

Spatial augmentation uses flips and 90-degree rotations. One transform plan is
drawn and applied identically to all four channels and the segmentation mask;
there is no interpolation or channel-specific geometry. Intensity augmentation
draws scale and shift independently for each modality and applies them only to
nonzero voxels. The mask is never passed through an intensity transform.

Training sampling and augmentation are deterministic functions of the global
seed, epoch, and dataset index. Repeating an index in one epoch returns the
same sample; changing the epoch changes the deterministic random stream.

## Validation and internal-test behavior

Validation and internal-test policies:

- include every patient;
- enumerate every slice along the declared axis;
- retain empty slices;
- apply no random augmentation;
- produce deterministic output.

For the audited 240×240×155 data and axis 2, each output image has shape
`[4,240,240]` and each label has shape `[240,240]`. Patient ID, slice index,
slice axis, spacing, empty-slice indicator, and split are returned with every
sample so predictions can be reassembled into full patient volumes before the
central evaluator is called.

The normal builder accepts only train or validation manifests. Direct
construction of a test dataset raises `PermissionError`. The internal-test
builder first invokes the explicit `allow_test_evaluation` guard and append-only
access log defined in Gate 2.

## Cache safety

Normalized-volume disk caching is implemented but disabled by default. When
enabled:

- `BRATS_CACHE_ROOT` or an explicit external path is required;
- any cache path at or below the raw-data root is rejected;
- cache keys include the subject, manifest paths and hashes, modality order,
  normalization, and clipping policy;
- writes are atomic compressed NPZ files;
- spatial and intensity augmentation are never cached.

## Verification

The loader tests cover fixed modality order, nonzero z-score, image/label
spatial synchronization, per-modality intensity augmentation, configurable
tumor/non-tumor sampling, deterministic validation, deterministic
epoch-indexed training, empty-slice preservation, channel/shape/dtype
conventions, cache isolation, and direct test-construction rejection.

A read-only real-data smoke test loaded one training subject
(`BraTS20_Training_002`) from the provisional train manifest:

| Property | Observed |
|---|---|
| Dataset samples per epoch | 4,128 (= 258 × 16) |
| Sampled slice | 78 |
| Image | `float32 [4,240,240]` |
| Label | `int64 [240,240]` |
| Finite image values | Yes |
| Split | train |

No training loop or internal-test access was performed.

