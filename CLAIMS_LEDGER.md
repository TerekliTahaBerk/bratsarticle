# Claim–Evidence Ledger

No manuscript claim is reportable until its status is `supported`.

| ID | Claim | Type | Evidence | Limitation | Citation | Status |
|---|---|---|---|---|---|---|
| C001 | BraTS 2020 training contains 369 complete and eligible canonical labeled subjects. | Data | `reports/data_audit_summary.json`; canonical manifest | Local authorized dataset copy | Official dataset metadata to be verified in literature gate | supported |
| C002 | BraTS 2019 contributes no independent subjects beyond the mapped BraTS 2020 cohort; 335 identities overlap and BraTS 2020 adds 34 subjects. | Data | `manifests/audit/duplicate_mapping.csv`; all four MRI modalities content-equivalent for mapped pairs | One mapped segmentation has a two-voxel annotation revision | Official BraTS mapping metadata | supported |
| C003 | RES and WC are components of BU-Net rather than novel components of this project. | Attribution | BU-Net methods and architecture description | Reimplementation details may differ | Rehman et al., 2020, DOI: 10.3390/electronics9122203 | supported |
| C004 | A model is more resource-efficient than a comparator. | Performance/resource | Accuracy, FLOPs/MACs, VRAM, GPU-hour, and latency artifacts | Hardware-specific | N/A | unsupported |
| C005 | A model is clinically applicable or generalizable. | Clinical/generalization | External validation and/or prospective evidence | No external validation yet | N/A | unsupported |
| C006 | The provisional 258/37/74 partition is patient-level, deterministic, duplicate-free by audited same-role file hashes, and balanced within the declared tolerances. | Data/method | `splits/provisional/split_metadata.json`; `reports/split_balance_report.md`; split tests | Split remains provisional until the analysis plan is frozen | N/A | supported |
