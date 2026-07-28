# Provisional Patient-Level Split Balance

**Status:** PASS

## Split definition

- Seed: `20260729`
- Candidate search size: 256
- Selected candidate index: 160
- Balance objective: 0.01545875
- Train subjects: 258
- Validation subjects: 37
- Internal held-out test subjects: 74

All partitions are patient-level. The internal held-out test subset is not available through the development loader.
Exact same-role file hashes were checked globally before partitioning; no cross-patient duplicate was found.

## Balance acceptance

- Maximum categorical prevalence deviation: 0.0439 (limit 0.0800)
- Maximum absolute SMD: 0.0484 (limit 0.3500)

## Manifest hashes

| Manifest | SHA-256 |
|---|---|
| train | `49a8d6836aba5f69bb39d5eb513e29c3746adaae72f02ecdc0df094cd86d7425` |
| validation | `95df721b5b013475b2100224279247e53afcc19c6ab79debf54283d27dc3d5bf` |
| test | `455b3b661be73a84fc99458798ee9a5cbbf9c70deac0b425397220fbbab7a525` |

## Continuous balance

| Split | Feature | SMD |
|---|---|---:|
| train | wt_volume_mm3 | 0.0034 |
| validation | wt_volume_mm3 | -0.0484 |
| test | wt_volume_mm3 | 0.0124 |
| train | tc_volume_mm3 | -0.0141 |
| validation | tc_volume_mm3 | 0.0234 |
| test | tc_volume_mm3 | 0.0376 |
| train | et_volume_mm3 | -0.0084 |
| validation | et_volume_mm3 | -0.0227 |
| test | et_volume_mm3 | 0.0406 |
