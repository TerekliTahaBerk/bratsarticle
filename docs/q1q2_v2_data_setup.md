# q1q2 v2 authorized-data setup

Raw MRI data are never stored in this repository. Users must obtain BraTS 2020
and the processed BraTS-Africa TCIA v1 release from their official sources,
accept the applicable terms and keep the files outside the Git checkout.

## Development data

1. Obtain the authorized BraTS 2020 training release.
2. Preserve the four native modalities and segmentation files without editing
   the raw tree.
3. Point the audit tooling to that authorized root.
4. Verify the resulting development inventory against the published
   patient-level fold manifests under `splits/q1q2_v2/`.

The legacy 74-patient internal subset must not be exposed to a new v2 model.

## External data

The selected external release is processed BraTS-Africa TCIA v1, DOI
`10.7937/V8H6-8X67`, licensed CC BY 4.0. The primary confirmatory cohort is the
95-patient glioma group; 51 other-neoplasm cases are supportive only.

The official transfer contained 13 zero-byte placeholders in the realized
download. The repair script assembles a separate verified copy and never
modifies the original download:

```bash
PYTHONPATH=src .venv/bin/python \
  work/q1q2_v2/assemble_verified_external_release.py --help
```

Run the audit tooling against the verified copy and compare its inventory and
overlap outputs to:

- `manifests/q1q2_v2/external_inventory.csv`
- `manifests/q1q2_v2/external_overlap_audit.csv`
- `reports/q1q2_v2/external_gate_c_summary.json`

## Access boundary

Data-integrity and overlap inspection are audit activities. Model prediction,
metric computation, threshold choice, post-processing choice and retraining on
the external cohort remain prohibited until the analysis freeze passes. Every
external access must be appended to
`artifacts/q1q2_v2/external_access_log.jsonl`.
