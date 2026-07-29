# Q1/Q2 v2 contribution assessment

## Bottom line

A credible v2 paper should not be positioned as a new-network paper. RES and WC
were introduced by Rehman et al. in BU-Net, and modern 3D/self-configuring
baselines already set a high comparison standard. The defensible contribution
is a **controlled falsification-oriented component study**:

1. isolate RES and WC under parameter- and compute-matched controls;
2. compare all principal models with identical seeds and patient-level
   five-fold development;
3. freeze the external endpoint and analysis before access;
4. test on a patient-overlap-free domain-shift cohort; and
5. connect overlap, boundary, lesion, failure, and resource results through a
   fully traceable artifact/claim ledger.

That contribution is potentially publishable only if it produces new
controlled evidence. The legacy study alone does not establish it.

## What the literature changes

BU-Net already claims contextual aggregation from RES and WC. Therefore the v2
question is not whether the repository can reimplement those blocks; it is
whether any observed gain remains after controlling capacity, computation,
seed variability, convergence, dimensional context, and external domain.

nnU-Net is a mandatory baseline because it demonstrates that pipeline
configuration can dominate handcrafted architectural changes. nnU-Net
Revisited further weakens claims based on under-controlled architecture
comparisons. TransBTS, nnFormer, and Swin UNETR show that a purely 2D
U-Net-family comparison is no longer an adequate state-of-field context.

Seed-variability studies show that a numerically favorable initialization can
change apparent rankings even under cross-validation. Using three seeds for a
reference and five for finalists is therefore not acceptable for v2
confirmatory inference.

Metrics Reloaded supports using complementary metric families. For glioma
subregions this means that regional Dice alone is insufficient: boundary
metrics, lesion detection/lesion-wise scores, explicit empty-mask behavior,
and patient-level aggregation are needed.

## External-data implications

- **BraTS-Africa** was selected after local audit: 146/146 patients have all
  four modalities and convertible labels, the primary glioma cohort is 95,
  and no identity/content overlap was found across 53,874 comparisons.
- **UCSF-PDGM** is large and technically compatible, but part of the collection
  entered BraTS 2021. It cannot be declared independent of BraTS 2020 without
  content-based overlap auditing.
- **UPenn-GBM** is large and four-modality, but its data descriptor states that
  173 cases were previously used in BraTS/FeTS. A clean external subset would
  require positive identification and exclusion of every overlapping case.
- **FeTS/BraTS 2021** cannot be used wholesale: its public training/validation
  data are identical to BraTS 2021 and inherit prior BraTS cases; protected
  institutional test labels are unavailable.
- Newer BraTS editions are not independent merely because their edition number
  is newer. They require the same identity/content audit, and hidden-label
  challenge tests are incompatible with the required per-patient confirmatory
  analysis unless official evaluation returns sufficient artifacts.

## Claim ceiling before new experiments

The maximum honest current claim is:

> The legacy study provides bounded internal motivation; v2 has established
> an overlap-free external cohort and frozen capacity/compute controls, but no
> v2 performance or external-generalization result exists before full training.

The following wording is unsupported: “component effect” and “external generalization.”
“State of the art,” “clinically robust,” and “Q1/Q2 ready” are unsupported.

## Contribution decision

Status: **design strengthened, empirically blocked by compute allocation**.

The study becomes a strong methods/evaluation contribution only after Gates
C–H. A negative component result remains valuable if controls, external
validation, and complete reporting are maintained.
