# Data card

## Development cohort

The v2 development cohort contains 369 unique labeled BraTS 2020 training
patients. Five patient-level folds are stratified by grade, enhancing-tumor
presence, and whole-tumor burden. Raw data are not included.

## External confirmatory cohort

The selected release is the processed BraTS-Africa TCIA v1 collection
(`10.7937/V8H6-8X67`, CC BY 4.0). All 146 released patients contain T1, T1ce,
T2, FLAIR, and expert tumor-subregion labels. The primary confirmatory cohort
is the 95-patient glioma sheet. Fifty-one other-neoplasm cases are supportive
only.

The audit compared all 146 external patients against 369 development patients
using identifiers, file hashes, sampled content, normalized volumes, robust
signatures, geometry, and institution metadata. No overlap was detected across
53,874 patient pairs.

## Access and redistribution

Users obtain each dataset from its official source and accept its terms. This
repository stores relative manifests and hashes, not the MRI files. The
separately downloaded external release is treated as read-only after verified
assembly.

## Known limitations

BraTS-Africa primary n is 95, grade metadata are incomplete, and scanner/site
subgroups may be small. The prespecified precision analysis therefore
emphasizes confidence intervals and marks subgroup cells below ten patients as
descriptive/insufficient.
