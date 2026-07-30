# Gate H single frozen external evaluation

Gate H remains physically and logically blocked until
`gate_g_analysis_freeze.json` authorizes external inference. Its contract is
`configs/q1q2_v2/gate_h_external.yaml`.

The external session evaluates the 300 main best checkpoints only: 12 models,
five folds, and five common seeds. The 200 compute-matched and 100
architecture-by-loss runs remain development sensitivity analyses and are not
used to multiply external hypotheses. Each checkpoint is evaluated on all 95
confirmatory glioma patients and the 51 supportive other-neoplasm patients.
The two cohorts are always labelled separately.

At the first opening the queue:

1. validates the immutable Gate G checkpoint and analysis hashes;
2. refuses any earlier external model-inference event;
3. re-hashes all 730 frozen source files;
4. creates a provenance-bound normalized mmap cache for native/Swin inference;
5. derives raw-valued, uncompressed NIfTI inputs for official nnU-Net without
   substituting the native normalized cache;
6. appends the irreversible external-session start event; and
7. runs every best checkpoint without changing normalization, thresholds,
   postprocessing, loss, or model parameters.

Run the queue only after Gate G passes:

```bash
nnUNet_raw="$PWD/work/q1q2_v2/nnunet/raw" \
nnUNet_preprocessed="$PWD/work/q1q2_v2/nnunet/preprocessed" \
nnUNet_results="$PWD/work/q1q2_v2/nnunet/results" \
nnUNet_extTrainer="$PWD/nnunet_ext" \
PYTHONPATH=src .venv/bin/python scripts/run_q1q2_gate_h_external.py \
  --external-root /authorized/BraTS-Africa \
  --allow-frozen-external-inference
```

Operational interruption may resume the same deterministic session when no
completed metric artifact exists. A checkpoint recorded as failed is not
retried or replaced. The gate reports the failure and cannot pass.

Per-checkpoint patient tables retain model, training fold, seed, checkpoint
hash, cohort role, scanner metadata, and all common metrics. The prespecified
model-level table is the arithmetic mean of the 25 frozen fold-seed
checkpoint metrics for each patient. Gate H passes only if all 300 checkpoints
complete and every model–patient row has exactly 25 replicates. The raw
checkpoint tables remain available for seed/fold uncertainty analyses.

The official nnU-Net adapter uses `nnUNetv2_predict`, the exact seeded trainer,
selected plans identifier, training fold, and `checkpoint_best.pth`. Temporary
prediction volumes are evaluated and discarded only after their patient
metrics are written and hash-verified; this avoids retaining hundreds of
gigabytes of redundant label maps. Native and Swin adapters also reload the
Gate G best-checkpoint hash before inference.

No outcome in Gate H can trigger a new seed, checkpoint, threshold,
normalization, or postprocessing decision.
