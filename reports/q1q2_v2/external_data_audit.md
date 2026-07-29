# External data audit

Audit date: 2026-07-30
Gate C status: **PASS**

## Release and access

- Cohort: processed BraTS-Africa TCIA v1
- TCIA collection DOI: `10.7937/V8H6-8X67`
- License: CC BY 4.0
- Released patients: 146
- Primary confirmatory glioma patients: 95
- Supportive other-neoplasm patients: 51
- Modalities: T1, T1ce, T2 and FLAIR
- Labels: expert tumor subregions, mapped
  `0→0; 1→1 (NCR/NET); 2→2 (ED); 3→4 (ET)`

The first Faspex transfer contained 13 zero-byte placeholders. The original
download was not modified. Those files were selectively reacquired from the
same official package, checked as readable 240×240×155 NIfTI files, hashed,
and assembled into a separate verified release. The repair manifest contains
13 data rows and SHA-256 values.

## Integrity and label-semantic results

| Requirement | Result |
|---|---:|
| Patients with four modalities and label | 146 / 146 |
| Integrity failures | 0 |
| Primary eligible glioma patients | 95 |
| Supportive other-neoplasm patients | 51 |
| Convertible WT/TC/ET semantics | PASS |
| External predictions or metrics accessed | No |

Observed source label sets were subsets of `{0,1,2,3}`. The fixed mapping
produces internal BraTS labels `{0,1,2,4}`. Derived regions are WT =
1∪2∪4, TC = 1∪4 and ET = 4. Geometry, orientation, spacing and file
readability were checked from the actual files.

## Content-based overlap result

Every external patient was compared with every one of the 369 canonical
BraTS 2020 development patients (53,874 pairs). The audit used canonicalized
identifiers, raw SHA-256 values, exact normalized-array hashes, sampled image
hashes, robust normalized-volume descriptors, geometry and institution
metadata.

| Signal | Matches |
|---|---:|
| Patient identifier | 0 |
| Raw file SHA-256 | 0 |
| Exact normalized image content | 0 |
| Sampled content signature | 0 |
| Normalized volume signature within `1e-5` RMS | 0 |

The nearest robust normalized-signature RMS distance was `0.03083456`, well
above the frozen near-match threshold `1e-5`. Median and maximum nearest
distances were `0.04992062` and `0.11541344`. No institution-mapping signal
linked the Sub-Saharan African centers to a canonical BraTS 2020 identifier.
Zero overlap is established for this audit.

## Machine-readable evidence

- Gate summary SHA-256:
  `dba96e850ba4877840fbf85c50819b468e9c94d067db24314dd43251366f4bc3`
- Inventory SHA-256:
  `82710419cf7277e0ae1e78ef1b7931e5bc4eef01bf6a2e9b09cea1dd455517ec`
- Overlap audit SHA-256:
  `3133594306c63059aaff8c2a6cf14c58c7bba481de37abe269198d02e9c41edb`
- Signature index SHA-256:
  `3b464abb59f08b32a67174467bb20c88c22db187f2dfb34ca98879d2ce7333dd`
- Transfer-repair manifest SHA-256:
  `7d53cbdede099ddb1fc3292617e868a8216620d4f9af113c12c98c66f2b2c754`

The append-only log contains one
`external_identity_integrity_label_audit` event with
`model_inference=false` and `prediction_metrics_accessed=false`. This Gate C
access is not the confirmatory result opening.

## Scientific boundary

Gate C passing authorizes development design; it does not authorize external
inference. The 95-patient external manifest may be opened for predictions only
after loss selection, all fold×seed checkpoints, configuration hashes and the
statistical plan pass Gate G. The 51 other-neoplasm cases cannot enter the
primary endpoint.
