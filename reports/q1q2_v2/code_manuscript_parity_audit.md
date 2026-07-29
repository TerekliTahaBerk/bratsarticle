# Q1/Q2 v2 code–manuscript parity audit

Audit scope: the frozen legacy implementation and Gate 14 manuscript at commit
`ab60a79d8a49a2fe1adb000546f653485796bab1`.

## Summary

The principal data split and patient-level inference descriptions match the
legacy code. Four material parity problems require correction before reuse in
v2: the background status of BCE is under-specified, MPS measurements use
VRAM-named registry fields, qualitative identities are called prespecified
although only selection rules were frozen, and the reviewer response lists
resource metrics absent from the manuscript.

## Loss formula and output semantics

For the selected `binary_cross_entropy_plus_focal_tversky` configuration,
`include_background: false`.

The implementation first computes element-wise
`binary_cross_entropy_with_logits(logits.float(), one_hot)` across four
channels, then removes channel 0 before reduction. Thus the selected BCE term
is **foreground-only**, not four-channel BCE. It uses independent sigmoid
Bernoulli probabilities internally.

The FTL term applies a four-class softmax to the same logits, computes
classwise Tversky quantities, then removes channel 0 before reduction. It is
also foreground-only. The raw logits are shared, but the probability mappings
are deliberately different:

```text
raw logits
  ├─ sigmoid per channel → foreground BCE
  └─ softmax over 4 classes → foreground focal Tversky
```

The legacy Methods correctly says that BCE uses sigmoid and FTL uses softmax,
and it explicitly calls FTL foreground-only. It does not explicitly say that
the selected BCE term also excludes background. That omission is material
because a reader could reasonably reconstruct a different objective.

For v2, the primary softmax-consistent loss candidates must be implemented and
tested separately:

- four-class CE + foreground soft Dice;
- foreground BCE + foreground FTL, explicitly labeled as a hybrid
  multi-label/multiclass objective if retained;
- four-class CE + foreground FTL.

Each equation must state the included channels, activation, reduction axes,
weights, smoothing, empty-class behavior, and inference mapping. Unit tests
must compare code against direct tensor equations.

## Capacity and compute parity

The manuscript describes matched preprocessing and training limits, not
matched capacity. Artifact-derived values show:

| Candidate | Parameters | Relative to U-Net | MAC/slice | Relative to U-Net |
|---|---:|---:|---:|---:|
| Standard U-Net | 1,942,772 | 1.00x | 2.676 G | 1.00x |
| U-Net+RES | 4,450,452 | 2.29x | 9.459 G | 3.53x |
| BU-Net | 8,384,660 | 4.32x | 10.344 G | 3.87x |

The legacy component result is therefore protocol-matched but not
parameter-matched or compute-matched. A v2 component-effect statement requires
both controls.

## Seed parity

The legacy Methods says all three reportable candidates were frozen, but their
replication counts differ. Standard U-Net has seeds 20260729–20260731; BU-Net
and U-Net+RES additionally have 20260732–20260733. This is accurately
recoverable from Gate 9/10 artifacts but fails the v2 equal-seed rule. All v2
main arms must use exactly the same five seeds in each of five folds.

## Resource terminology

`ResourceTracker` samples:

- CUDA: PyTorch peak allocated and reserved device memory;
- MPS: `torch.mps.current_allocated_memory()` and
  `torch.mps.driver_allocated_memory()`.

Both are serialized under `peak_allocated_vram_bytes` and
`peak_reserved_vram_bytes`. That schema is inaccurate for Apple unified
memory. The legacy manuscript mostly says “allocated memory,” but the protocol
and registry still say VRAM. Future schema and prose must distinguish:

- CUDA peak allocated/reserved device memory; and
- **MPS framework-reported allocated unified memory** plus driver-allocated
  memory.

The legacy numbers are framework counters, not a measurement of total physical
unified-memory consumption.

## Qualitative case prespecification

Gate 10 froze deterministic role-selection rules and a prediction seed before
test access. It did not know or freeze the identities of success, hard, and
failure patients before evaluation. The caption “Prespecified BU-Net ET
success, hard, and failure cases” overstates what was prespecified.

Required wording:

> Cases were selected after evaluation using prespecified deterministic rules.

The identities must not be described as prespecified.

## Reviewer-response resource claims

The response states that the paper reports parameter count, checkpoint size,
MACs, FLOPs, p50/p95 latency, throughput, allocated/reserved memory, and
accelerator-hours. Manuscript Table 7 contains parameters, MACs, FLOPs,
p50/p95 latency, peak allocated MB, accelerator-hours, and Dice. It omits
checkpoint size, throughput, and reserved/driver memory. The response must be
corrected unless a future artifact-generated table adds those columns.

## A100 versus realized Apple MPS

Gate 7 protocol files target `NVIDIA A100-SXM4-80GB`. Gate 8–11 reportable runs
were realized on Apple M1 Max through MPS. The protocol later records this
change, but retains A100 requirements and VRAM language in places. The A100
plan must remain archived as unrealized rather than being presented as the
execution environment.

The current audit environment exposes neither CUDA nor available MPS. It
cannot be assumed equivalent to the legacy execution environment.

## Parity disposition

| Area | Legacy parity | v2 action |
|---|---|---|
| Patient-level split | Matched | Replace legacy split with 5-fold development only after Gate C |
| Internal-test role | Matched for v1 | Prohibit all new evaluation on the 74 patients |
| Selected loss activation | Partly stated | Explicitly state foreground exclusion for both BCE and FTL |
| Inference mapping | Matched | Retain four-class argmax where applicable |
| Parameters/MACs | Artifact-derived | Add parameter- and compute-matched controls |
| Seed count | Accurately recoverable but unequal | Enforce identical five-seed lists |
| Resource language | Partly inaccurate | Use backend-neutral schema and exact MPS wording |
| Qualitative selection | Rules frozen; identities post-evaluation | Correct caption/methods |
| Reviewer resource list | Over-claims manuscript contents | Correct response or table |
