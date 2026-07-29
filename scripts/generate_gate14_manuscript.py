#!/usr/bin/env python3
# ruff: noqa: E501, RUF001
"""Generate the Gate 14 manuscript package from tracked scientific artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = ROOT / "manuscript"
REPORTS_DIR = ROOT / "reports"

MODEL_LABELS = {
    "unet_reference": "Standard U-Net",
    "bunet": "BU-Net",
    "unet_res": "U-Net+RES",
}
REGION_LABELS = {"wt": "WT", "tc": "TC", "et": "ET"}


def _read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _metric(rows: list[dict[str, str]], candidate: str, metric: str) -> dict[str, str]:
    for row in rows:
        if row["candidate_id"] == candidate and row["metric"] == metric:
            return row
    raise KeyError((candidate, metric))


def _comparison(
    rows: list[dict[str, str]], comparison_id: str, metric: str
) -> dict[str, str]:
    for row in rows:
        if row["comparison_id"] == comparison_id and row["metric"] == metric:
            return row
    raise KeyError((comparison_id, metric))


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    top = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([top, separator, *body])


def _cohort_table() -> str:
    rows = _read_csv("tables/final/table01_cohort_characteristics.csv")
    return _markdown_table(
        [
            "Partition",
            "Patients",
            "HGG",
            "LGG",
            "ET present",
            "WT volume median (IQR), mm3",
        ],
        [
            [
                row["Partition"],
                row["Patients"],
                row["HGG"],
                row["LGG"],
                row["ET present"],
                row["WT volume median (IQR), mm3"],
            ]
            for row in rows
        ],
    )


def _resource_table(
    resource_rows: list[dict[str, str]], metric_rows: list[dict[str, str]]
) -> str:
    table_rows: list[list[str]] = []
    for candidate in ("unet_reference", "bunet", "unet_res"):
        subset = [row for row in resource_rows if row["candidate_id"] == candidate]
        table_rows.append(
            [
                MODEL_LABELS[candidate],
                f"{statistics.mean(_float(row, 'parameter_count') for row in subset) / 1e6:.3f}",
                f"{statistics.mean(_float(row, 'macs_per_slice') for row in subset) / 1e9:.3f}",
                f"{statistics.mean(_float(row, 'flops_per_slice') for row in subset) / 1e9:.3f}",
                f"{statistics.mean(_float(row, 'latency_p50_seconds') for row in subset):.3f}",
                f"{statistics.mean(_float(row, 'latency_p95_seconds') for row in subset):.3f}",
                f"{statistics.mean(_float(row, 'development_peak_allocated_vram_bytes') for row in subset) / 1e6:.1f}",
                f"{statistics.mean(_float(row, 'development_gpu_hours') for row in subset):.3f}",
                _fmt(
                    _float(
                        _metric(metric_rows, candidate, "mean_regional_dice"),
                        "mean_finite",
                    )
                ),
            ]
        )
    return _markdown_table(
        [
            "Candidate",
            "Parameters, M",
            "MAC/slice, G",
            "FLOP/slice, G",
            "p50 s/volume",
            "p95 s/volume",
            "Peak allocated MB",
            "Development GPU-h/run",
            "Mean regional Dice",
        ],
        table_rows,
    )


def _test_dice_table(metric_rows: list[dict[str, str]]) -> str:
    rows: list[list[str]] = []
    for candidate in ("unet_reference", "bunet", "unet_res"):
        mean_row = _metric(metric_rows, candidate, "mean_regional_dice")
        rows.append(
            [
                MODEL_LABELS[candidate],
                _fmt(_float(mean_row, "mean_finite")),
                f"{_fmt(_float(mean_row, 'bootstrap_lower_finite'))}-{_fmt(_float(mean_row, 'bootstrap_upper_finite'))}",
                _fmt(_float(_metric(metric_rows, candidate, "wt_dice"), "mean_finite")),
                _fmt(_float(_metric(metric_rows, candidate, "tc_dice"), "mean_finite")),
                _fmt(_float(_metric(metric_rows, candidate, "et_dice"), "mean_finite")),
            ]
        )
    return _markdown_table(
        [
            "Candidate",
            "Mean regional Dice",
            "95% bootstrap CI",
            "WT Dice",
            "TC Dice",
            "ET Dice",
        ],
        rows,
    )


def _comparison_table(comparison_rows: list[dict[str, str]]) -> str:
    rows: list[list[str]] = []
    for comparison_id in (
        "bunet_vs_unet_reference",
        "unet_res_vs_unet_reference",
        "bunet_vs_unet_res",
    ):
        row = _comparison(comparison_rows, comparison_id, "mean_regional_dice")
        label = (
            f"{MODEL_LABELS[row['first_candidate']]} - "
            f"{MODEL_LABELS[row['second_candidate']]}"
        )
        rows.append(
            [
                label,
                _fmt_signed(_float(row, "paired_mean_difference")),
                f"{_fmt_signed(_float(row, 'paired_bootstrap_lower'))} to "
                f"{_fmt_signed(_float(row, 'paired_bootstrap_upper'))}",
                _fmt_signed(_float(row, "standardized_paired_effect_dz")),
                _fmt_p(_float(row, "permutation_p_value")),
                _fmt_p(_float(row, "holm_adjusted_p_value")),
            ]
        )
    return _markdown_table(
        ["Contrast", "Paired difference", "95% bootstrap CI", "dz", "Raw p", "Holm p"],
        rows,
    )


def _secondary_table(metric_rows: list[dict[str, str]]) -> str:
    rows: list[list[str]] = []
    for candidate in ("unet_reference", "bunet", "unet_res"):
        values: list[str] = [MODEL_LABELS[candidate]]
        for region in ("wt", "tc", "et"):
            surface = _metric(metric_rows, candidate, f"{region}_surface_dice")
            values.append(_fmt(_float(surface, "mean_finite")))
        for region in ("wt", "tc", "et"):
            hd = _metric(metric_rows, candidate, f"{region}_hd95_mm")
            text = _fmt(_float(hd, "mean_finite"), 1)
            inf_count = int(hd["positive_infinity_count"])
            if inf_count:
                text += f" ({inf_count} inf)"
            values.append(text)
        et_recall = _metric(metric_rows, candidate, "et_lesion_recall")
        et_lwd = _metric(metric_rows, candidate, "et_lesion_wise_dice")
        values.extend(
            [
                _fmt(_float(et_recall, "mean_finite")),
                _fmt(_float(et_lwd, "mean_finite")),
            ]
        )
        rows.append(values)
    return _markdown_table(
        [
            "Candidate",
            "WT surface Dice",
            "TC surface Dice",
            "ET surface Dice",
            "WT HD95 mm",
            "TC HD95 mm",
            "ET HD95 mm",
            "ET lesion recall",
            "ET lesion-wise Dice",
        ],
        rows,
    )


def _subgroup_table(subgroup_rows: list[dict[str, str]]) -> str:
    rows: list[list[str]] = []
    requested = [
        ("grade", "HGG"),
        ("grade", "LGG"),
        ("enhancing_tumor_reference", "present"),
        ("enhancing_tumor_reference", "absent"),
        ("whole_tumor_burden", "small"),
        ("whole_tumor_burden", "medium"),
        ("whole_tumor_burden", "large"),
    ]
    for subgroup, category in requested:
        subset = [
            row
            for row in subgroup_rows
            if row["record_type"] == "candidate_estimate"
            and row["subgroup"] == subgroup
            and row["category"] == category
        ]
        by_model = {row["candidate_id"]: row for row in subset}
        display_category = {
            "enhancing_tumor_reference": "ET reference",
            "whole_tumor_burden": "WT burden",
            "grade": "Grade",
        }[subgroup]
        rows.append(
            [
                f"{display_category}: {category}",
                by_model["bunet"]["patient_count"],
                _fmt(_float(by_model["unet_reference"], "mean_regional_dice")),
                _fmt(_float(by_model["bunet"], "mean_regional_dice")),
                _fmt(_float(by_model["unet_res"], "mean_regional_dice")),
                by_model["bunet"]["reportability"],
            ]
        )
    return _markdown_table(
        ["Subgroup", "n", "Standard U-Net", "BU-Net", "U-Net+RES", "Interpretation"],
        rows,
    )


def _development_architecture_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["screen"] == "architecture"]
    display = {
        "architecture_unet": "Standard U-Net",
        "architecture_bunet": "BU-Net",
        "architecture_unet_res": "U-Net+RES",
        "architecture_unet_wc": "U-Net+WC",
        "architecture_resunet": "Residual-block U-Net",
        "architecture_resunet_wc": "Residual-block U-Net+WC",
    }
    return _markdown_table(
        ["Architecture arm", "Mean regional Dice", "Eliminated"],
        [
            [
                display[row["arm_id"]],
                _fmt(_float(row, "mean_regional_dice")),
                "yes" if row["eliminated"] == "True" else "no",
            ]
            for row in selected
        ],
    )


def _development_loss_table(rows: list[dict[str, str]]) -> str:
    selected = [row for row in rows if row["screen"] == "loss"]
    display = {
        "architecture_unet": "Cross-entropy + soft Dice",
        "loss_binary_cross_entropy": "Binary cross-entropy",
        "loss_binary_cross_entropy_plus_focal_tversky": "Binary cross-entropy + focal Tversky",
        "loss_binary_cross_entropy_plus_soft_dice": "Binary cross-entropy + soft Dice",
        "loss_cross_entropy": "Cross-entropy",
        "loss_focal_tversky": "Focal Tversky",
        "loss_soft_dice": "Soft Dice",
    }
    return _markdown_table(
        ["Loss arm", "Mean regional Dice", "Eliminated"],
        [
            [
                display[row["arm_id"]],
                _fmt(_float(row, "mean_regional_dice")),
                "yes" if row["eliminated"] == "True" else "no",
            ]
            for row in selected
        ],
    )


def _figure(path: str, caption: str, label: str, width: str = "92%") -> str:
    return f"![{caption}]({path}){{#{label} width={width}}}"


def _manuscript() -> str:
    audit = _read_json("reports/data_audit_summary.json")
    split = _read_json("splits/frozen/split_metadata.json")
    gate11 = _read_json("reports/gate11_analysis.json")
    metric_rows = _read_csv("reports/gate11_metric_summary.csv")
    comparison_rows = _read_csv("reports/gate11_comparisons.csv")
    resource_rows = _read_csv("reports/gate11_resource_summary.csv")
    subgroup_rows = _read_csv("reports/gate11_subgroups.csv")
    gate8_rows = _read_csv("reports/gate8_arm_summary.csv")
    gate9_confirm = _read_csv("reports/gate9_confirmation_summary.csv")
    gate9_final = _read_csv("reports/gate9_final_summary.csv")

    unet_mean = _float(
        _metric(metric_rows, "unet_reference", "mean_regional_dice"), "mean_finite"
    )
    bunet_mean = _float(
        _metric(metric_rows, "bunet", "mean_regional_dice"), "mean_finite"
    )
    res_mean = _float(
        _metric(metric_rows, "unet_res", "mean_regional_dice"), "mean_finite"
    )
    bu_ref = _comparison(
        comparison_rows, "bunet_vs_unet_reference", "mean_regional_dice"
    )
    res_ref = _comparison(
        comparison_rows, "unet_res_vs_unet_reference", "mean_regional_dice"
    )
    bu_res = _comparison(comparison_rows, "bunet_vs_unet_res", "mean_regional_dice")
    resource_means = {
        candidate: {
            key: statistics.mean(
                _float(row, key)
                for row in resource_rows
                if row["candidate_id"] == candidate
            )
            for key in (
                "parameter_count",
                "macs_per_slice",
                "latency_p50_seconds",
                "latency_p95_seconds",
                "development_peak_allocated_vram_bytes",
                "development_gpu_hours",
            )
        }
        for candidate in ("unet_reference", "bunet", "unet_res")
    }
    gate9_by_candidate = {row["candidate_id"]: row for row in gate9_confirm}
    gate9_final_by_candidate = {row["candidate_id"]: row for row in gate9_final}
    qualitative = gate11["qualitative_cases"]

    title = (
        "Leakage-Safe Multi-Seed Evaluation of Published BU-Net Components "
        "for Resource-Constrained 2D Glioma Segmentation"
    )

    return f"""# {title}

Taha Berk Terekli^1^, Livanur Mengeş^2^, Volkan Yusuf Hal^3^, Ali Emre Döşer^4^

^1^ Department of Mathematics, Yıldız Technical University, Istanbul, Turkey<br>
^2^ Department of Computer Engineering, Istanbul Beykent University, Istanbul, Turkey<br>
^3^ Department of Software Engineering, Istanbul Beykent University, Istanbul, Turkey<br>
^4^ Department of Computer Engineering, Haliç University, Istanbul, Turkey

**Corresponding author:** Taha Berk Terekli<br>
**Running title:** Controlled evaluation of BU-Net components<br>
**Article type:** Original research - methodological evaluation

## Abstract

**Background:** Architectural comparisons in glioma segmentation are vulnerable to patient leakage, inconsistent training budgets, incomplete attribution, and selective reporting. We evaluated previously published BU-Net components under a single guarded protocol rather than proposing a new architecture.

**Methods:** BraTS 2020 training cases formed the canonical labeled cohort. BraTS 2019 was used only to audit identity overlap. All {audit["brats2020"]["subject_count"]} canonical patients were split at patient level into {split["counts"]["train"]} training, {split["counts"]["validation"]} validation, and {split["counts"]["test"]} internal held-out test cases. A standard 2D U-Net, the published BU-Net reimplementation with residual extended skip (RES) and wide context (WC) modules, and U-Net+RES were trained with identical preprocessing, loss, optimization, and 2,000-step per-run limits. Development screening preceded a five-seed finalist stage. The test manifest was opened once after candidate and analysis freezing. The primary outcome was patient-level mean Dice across whole tumor, tumor core, and enhancing tumor. Three paired contrasts used 10,000 bootstrap resamples, 100,000 sign-flip permutations, and Holm correction.

**Results:** On the internal held-out test subset, mean regional Dice was {_fmt(unet_mean)} for standard U-Net, {_fmt(bunet_mean)} for BU-Net, and {_fmt(res_mean)} for U-Net+RES. Relative to standard U-Net, paired differences were {_fmt_signed(_float(bu_ref, "paired_mean_difference"))} for BU-Net (95% CI {_fmt_signed(_float(bu_ref, "paired_bootstrap_lower"))} to {_fmt_signed(_float(bu_ref, "paired_bootstrap_upper"))}; Holm p={_fmt_p(_float(bu_ref, "holm_adjusted_p_value"))}) and {_fmt_signed(_float(res_ref, "paired_mean_difference"))} for U-Net+RES (95% CI {_fmt_signed(_float(res_ref, "paired_bootstrap_lower"))} to {_fmt_signed(_float(res_ref, "paired_bootstrap_upper"))}; Holm p={_fmt_p(_float(res_ref, "holm_adjusted_p_value"))}). BU-Net was lower than U-Net+RES by {_fmt(abs(_float(bu_res, "paired_mean_difference")))} (95% CI {_fmt_signed(_float(bu_res, "paired_bootstrap_lower"))} to {_fmt_signed(_float(bu_res, "paired_bootstrap_upper"))}; Holm p={_fmt_p(_float(bu_res, "holm_adjusted_p_value"))}). BU-Net used {resource_means["bunet"]["parameter_count"] / 1e6:.3f} million parameters versus {resource_means["unet_res"]["parameter_count"] / 1e6:.3f} million for U-Net+RES and had higher median latency and peak allocated memory.

**Conclusions:** Under this bounded 2D protocol, the published RES component was associated with a small improvement over standard U-Net. Adding WC in the full BU-Net did not improve the primary endpoint over RES alone and increased resource demand. These internal, single-dataset findings do not establish clinical utility, external generalization, or superiority over 3D, transformer, or self-configuring systems.

**Keywords:** brain tumor segmentation; BraTS; U-Net; BU-Net; reproducibility; patient-level evaluation; resource profiling

\\pagebreak

## 1. Introduction

Glioma segmentation supports quantitative analysis of multimodal magnetic resonance imaging (MRI), but model rankings can be distorted by experimental design. Slice-level partitioning can place images from one patient in multiple subsets. Reusing overlapping BraTS editions can duplicate subjects. A comparison can also favor one architecture through a different loss, augmentation policy, run duration, or stopping rule. Finally, an overlap score alone does not describe boundary error, lesion detection, failure modes, or computational demand.

The BraTS benchmark standardized multimodal glioma MRI and the evaluation of whole tumor (WT), tumor core (TC), and enhancing tumor (ET) [1-4]. U-Net remains a useful reference architecture [5]. Rehman et al. later described BU-Net, a 2D U-Net modification containing residual extended skip (RES) and wide context (WC) modules [6]. These modules are prior work; we do not claim either component as novel. Our question is narrower: what evidence do RES and WC provide when implemented in one codebase and evaluated with matched data, training, statistical, and resource protocols?

Contemporary segmentation systems include self-configuring pipelines and volumetric transformer designs [9,14,15]. They are important context, but they were not trained here. Comparing our internal scores with literature values produced under different cohorts and protocols would not be a controlled benchmark. We therefore position this work as a component and reproducibility study within a 2D U-Net family.

The study makes four contributions:

1. a content-based audit of the overlap between BraTS 2019 and BraTS 2020 before selecting one canonical cohort;
2. a patient-level development and test workflow with a frozen, one-time internal-test access event;
3. a multi-seed, matched-protocol evaluation of standard U-Net, U-Net+RES, and the full BU-Net reimplementation; and
4. artifact-derived statistical, resource, subgroup, and qualitative reports released with executable checks.

No contribution is framed as a new network block, clinical device, or state-of-the-art result.

## 2. Related work

BraTS combines T1, post-contrast T1, T2, and FLAIR MRI with expert tumor annotations [1-4]. The official tasks define WT as labels 1, 2, and 4; TC as labels 1 and 4; and ET as label 4 [2]. Dice and 95th-percentile Hausdorff distance (HD95) are established challenge measures, but each captures a different property of the segmentation.

U-Net introduced an encoder-decoder design with skip connections for biomedical segmentation [5]. BU-Net retained the 2D U-Net form while adding RES modules to the skip pathways and a WC module at the bottleneck [6]. The large separable kernels in these modules aim to increase contextual support. Because RES and WC originated in BU-Net, the present implementation and all component names are attributed to Rehman et al.

Loss design also matters for imbalanced lesions. Tversky and focal Tversky objectives modify the relative weighting of false-positive and false-negative errors [7,8]. We screened these objectives under the same development protocol rather than assigning different losses to different final models.

Methodological guidance recommends choosing metrics for the task, preserving the correct unit of analysis, reporting the split level, declaring primary outcomes, and making software available [10,12]. We therefore used the patient as the statistical unit, treated secondary endpoints as estimation-only analyses, and retained undefined or infinite outcomes instead of silently replacing them. Surface Dice was included at a predeclared 1 mm tolerance [11]. That tolerance is an analytic setting, not a claim of clinical acceptability or interobserver calibration.

## 3. Materials and methods

### 3.1 Study design and reporting scope

This was a retrospective computational study of a public, de-identified challenge dataset. The workflow was organized into sequential gates: data integrity and identity audit; patient-level split; preprocessing and evaluator validation; model and loss implementation tests; single-seed development screening; multi-seed confirmation; analysis freezing; one guarded internal-test evaluation; and artifact-derived reporting. CLAIM 2024 informed the reporting checklist [12].

The study did not use the official BraTS 2020 validation set because its reference labels were withheld. In this paper, “internal held-out test subset” refers only to the labeled partition created from the BraTS 2020 training cohort.

{_figure("figures/final/fig02_split_and_analysis_flow.png", "Figure 1. Gated development, freezing, and one-time internal-test analysis flow.", "fig:flow")}

### 3.2 Cohort audit, canonicalization, and splitting

We inventoried all expected modalities and segmentations by patient and computed file- and image-content signatures. BraTS 2020 training was selected as the canonical labeled cohort. BraTS 2019 contributed no additional training observation and was used only for identity and duplicate auditing.

The audit found {audit["brats2019"]["subject_count"]} complete BraTS 2019 cases and {audit["brats2020"]["subject_count"]} complete BraTS 2020 cases. All {audit["mapping"]["mapped_overlap_count"]} BraTS 2019 patients mapped to BraTS 2020 by image content; {audit["mapping"]["new_in_brats2020_count"]} cases were new in the 2020 edition. The mapped image modalities were content-equivalent. One mapped segmentation differed by {audit["mapping"]["segmentation_revision_subjects"][0]["differing_voxel_count"]} voxels, which was recorded as an annotation revision rather than a new patient. The canonical cohort therefore contained {audit["brats2020"]["subject_count"]} unique patients.

Patients, not slices, were partitioned with seed {split["seed"]}. The selected candidate minimized imbalance across grade, ET presence, and WT-volume strata while satisfying frozen tolerances. Patient identifiers did not overlap between subsets.

**Table 1. Cohort characteristics by patient-level partition.**

{_cohort_table()}

{_figure("figures/final/fig01_cohort_flow.png", "Figure 2. Cohort identity audit and canonical patient flow.", "fig:cohort")}

### 3.3 MRI preprocessing and sampling

Each case used T1, T1ce, T2, and FLAIR in that channel order. Volumes supplied by BraTS were co-registered, skull stripped, and sampled axially at 240 x 240 pixels. For each patient and modality, nonzero voxels were standardized with that volume's mean and standard deviation. Intensity clipping was disabled.

Training sampled 16 slices per patient per epoch; the probability of selecting a tumor-containing slice was 0.67 and at least one tumor voxel defined a positive slice. Spatial augmentation used independent flips with probability 0.5 and rotations by multiples of 90 degrees. Per-modality intensity augmentation, applied with probability 0.5, used scale 0.9-1.1 and shift -0.1 to 0.1 in standardized units. Validation and test evaluation traversed all slices, including empty slices, deterministically. Cached arrays were memory-mapped outside the raw-data roots.

### 3.4 Architectures and attribution

All candidates accepted four MRI channels and produced four logits representing background and BraTS labels 1, 2, and 4. The common encoder widths were 16, 32, 64, and 128, followed by a 256-channel bottleneck. Batch normalization and dropout probability 0.3 were shared.

Standard U-Net used ordinary encoder-decoder skip connections. U-Net+RES added the published BU-Net RES pathways but omitted WC. The full BU-Net reimplementation combined RES with the published WC bottleneck module. RES used separable N x 1 and 1 x N branches with N in 9, 11, 13, and 15 across resolution levels, followed by fusion convolutions. WC used two oppositely ordered separable 15-pixel paths whose outputs were summed. Figure 3 identifies publication provenance directly.

The implementation followed the BU-Net prose where the original schematic was ambiguous. Deliberate implementation choices were four mutually exclusive output classes, base width 16, the stated dropout placement, and no imported external code. These choices make the present work a reimplementation study, not an exact reproduction of the original paper.

{_figure("figures/final/fig03_model_architectures.png", "Figure 3. Compared 2D architectures. RES and WC are published BU-Net components from Rehman et al. [6].", "fig:architectures")}

### 3.5 Loss function and development screen

The selected loss was an equal-weight combination of channel-wise binary cross-entropy (BCE) and foreground focal Tversky loss (FTL):

$$
\\mathcal{{L}} = 0.5\\,\\mathcal{{L}}_{{BCE}} + 0.5\\,\\mathcal{{L}}_{{FTL}},
$$

$$
\\mathcal{{L}}_{{FTL}} =
\\frac{{1}}{{|C_f|}}\\sum_{{c\\in C_f}}
\\left(1-
\\frac{{TP_c+\\epsilon}}{{TP_c+\\alpha FP_c+\\beta FN_c+\\epsilon}}
\\right)^\\gamma ,
$$

where $C_f$ contains the three foreground classes, $\\alpha=0.3$, $\\beta=0.7$, $\\gamma=0.75$, and $\\epsilon=10^{{-5}}$. BCE used sigmoid probabilities and one-hot targets; FTL used softmax probabilities. Inference used argmax over the four output classes. No class weights were applied.

The development screen evaluated six architecture arms and seven loss arms once each on the validation subset. Its role was shortlisting, not hypothesis testing or reporting a generalization estimate.

**Table 2. Single-seed architecture development screen (n=37 validation patients).**

{_development_architecture_table(gate8_rows)}

**Table 3. Single-seed loss development screen (n=37 validation patients).**

{_development_loss_table(gate8_rows)}

### 3.6 Matched training and multi-seed confirmation

Every reportable run used the same Apple M1 Max MPS device, inputs, batch size 16, AdamW optimizer, learning rate 0.001, weight decay 0.00001, augmentation, and loss. Mixed precision and pretraining were disabled. The realized bounded protocol stopped each run at 2,000 optimizer steps or 0.5 accelerator-hours, whichever occurred first. A 200-step linear warmup preceded cosine decay to 0.01 of the initial rate. Validation occurred at step 2,000, and the highest patient-level mean regional Dice checkpoint was retained.

Three seeds ({gate9_by_candidate["bunet"]["seed_count"]} per candidate) confirmed standard U-Net, BU-Net, U-Net+RES, and U-Net+WC. U-Net+WC was eliminated by the predeclared rule. BU-Net and U-Net+RES then received two additional seeds each. Five-seed validation means were {_fmt(_float(gate9_final_by_candidate["bunet"], "mean_regional_dice"))} and {_fmt(_float(gate9_final_by_candidate["unet_res"], "mean_regional_dice"))}, respectively; the U-Net+RES minus BU-Net paired difference was {_fmt_signed(_float(gate9_final_by_candidate["unet_res"], "paired_mean_difference"))} (95% bootstrap CI {_fmt_signed(_float(gate9_final_by_candidate["unet_res"], "paired_bootstrap_lower"))} to {_fmt_signed(_float(gate9_final_by_candidate["unet_res"], "paired_bootstrap_upper"))}). Because this interval included zero, the development ranking was not interpreted as superiority. Standard U-Net, BU-Net, and U-Net+RES were frozen for internal-test evaluation.

### 3.7 Guarded internal-test evaluation

Before test access, candidate identities, 13 checkpoint hashes, the split hashes, the endpoint, three paired contrasts, inference rules, and statistical procedures were frozen. Test evaluation required an explicit authorization flag and an append-only access log. The test manifest was opened once. No checkpoint, threshold, post-processing stage, or model-selection decision changed afterward.

For each candidate, per-seed endpoint values were averaged within patient before patient-level statistical inference. No test-time augmentation or post-processing was used.

### 3.8 Outcomes and metric rules

The primary outcome was each patient's arithmetic mean of WT, TC, and ET Dice. Region Dice values were also reported separately. Secondary estimates included HD95, surface Dice at 1 mm, lesion recall, lesion precision, lesion-wise Dice, false-positive lesion count, and relative volume error.

Lesions used 26-connectivity with a one-voxel minimum. Predicted and reference lesions were paired by maximum-total-IoU matching. If both masks were empty, overlap and surface Dice were 1 and HD95 was 0 mm; if only one mask was empty, overlap and surface Dice were 0 and HD95 was infinity. Undefined rates remained missing. Tables state finite denominators and infinity counts where applicable.

### 3.9 Statistical analysis

The patient was the only inferential unit. For each candidate, 95% confidence intervals used 10,000 patient-level bootstrap resamples with seed 20260729. Three frozen paired comparisons of mean regional Dice were tested with 100,000 two-sided sign-flip permutations using seed 20260730. Holm's sequential procedure controlled the family-wise error rate at 0.05 [13]. We report paired mean differences, bootstrap intervals, paired standardized effect $d_z$, raw p values, and Holm-adjusted p values. Regional and secondary endpoints were estimation-only; no unplanned multiplicity-adjusted claims were made.

Grade, reference ET presence, and training-derived WT burden tertiles were exploratory. The ET-absent group was descriptive because it contained only five patients.

### 3.10 Resource profiling and reproducibility

Parameter counts and multiply-accumulate operations (MACs) were computed from the implemented models at a 4 x 240 x 240 slice input. One MAC was reported as two floating-point operations (FLOPs). Per-volume inference latency and allocated memory were measured on the same Apple M1 Max host and summarized across seeds. Development accelerator-hours were retained from each run. These measurements compare the present implementations on one host; they are not deployment benchmarks.

All tables and figures were generated from machine-readable run artifacts. A tracked manifest records hashes for reportable files. A clean-clone audit rebuilt the Gate 12 outputs twice, confirmed byte-identical results, ran static checks and tests, and verified a clean worktree.

## 4. Results

### 4.1 Cohort integrity and development selection

The content audit identified complete imaging and labels for all {audit["brats2020"]["subject_count"]} canonical patients, no file-integrity errors, and no patient overlap across partitions. The maximum absolute standardized mean difference across continuous split features was {split["maximum_absolute_standardized_mean_difference"]:.3f}, below the frozen tolerance of {split["tolerances"]["max_absolute_standardized_mean_difference"]:.2f}.

In the single-seed screen, BU-Net had the highest architecture-screen mean regional Dice ({_fmt(_float(next(row for row in gate8_rows if row["arm_id"] == "architecture_bunet"), "mean_regional_dice"))}); U-Net+RES was within the practical screening margin. BCE+FTL had the highest loss-screen mean ({_fmt(_float(next(row for row in gate8_rows if row["arm_id"] == "loss_binary_cross_entropy_plus_focal_tversky"), "mean_regional_dice"))}). These development observations only determined which arms advanced.

In the three-seed stage, mean regional Dice was {_fmt(_float(gate9_by_candidate["unet_reference"], "mean_regional_dice"))} for standard U-Net, {_fmt(_float(gate9_by_candidate["bunet"], "mean_regional_dice"))} for BU-Net, {_fmt(_float(gate9_by_candidate["unet_res"], "mean_regional_dice"))} for U-Net+RES, and {_fmt(_float(gate9_by_candidate["unet_wc"], "mean_regional_dice"))} for U-Net+WC. Only U-Net+WC met the elimination rule. The finalist interval spanning zero justified carrying both BU-Net and U-Net+RES into the frozen test analysis.

### 4.2 Primary internal-test outcome

All 74 test patients had finite primary outcomes. Mean regional Dice was {_fmt(unet_mean)} for standard U-Net, {_fmt(bunet_mean)} for BU-Net, and {_fmt(res_mean)} for U-Net+RES.

**Table 4. Internal held-out test Dice estimates (n=74 patients).**

{_test_dice_table(metric_rows)}

BU-Net exceeded standard U-Net by {_fmt_signed(_float(bu_ref, "paired_mean_difference"))} (95% CI {_fmt_signed(_float(bu_ref, "paired_bootstrap_lower"))} to {_fmt_signed(_float(bu_ref, "paired_bootstrap_upper"))}; $d_z$={_fmt_signed(_float(bu_ref, "standardized_paired_effect_dz"))}; raw p={_fmt_p(_float(bu_ref, "permutation_p_value"))}; Holm p={_fmt_p(_float(bu_ref, "holm_adjusted_p_value"))}). U-Net+RES exceeded standard U-Net by {_fmt_signed(_float(res_ref, "paired_mean_difference"))} (95% CI {_fmt_signed(_float(res_ref, "paired_bootstrap_lower"))} to {_fmt_signed(_float(res_ref, "paired_bootstrap_upper"))}; $d_z$={_fmt_signed(_float(res_ref, "standardized_paired_effect_dz"))}; raw p={_fmt_p(_float(res_ref, "permutation_p_value"))}; Holm p={_fmt_p(_float(res_ref, "holm_adjusted_p_value"))}).

BU-Net minus U-Net+RES was {_fmt_signed(_float(bu_res, "paired_mean_difference"))} (95% CI {_fmt_signed(_float(bu_res, "paired_bootstrap_lower"))} to {_fmt_signed(_float(bu_res, "paired_bootstrap_upper"))}; $d_z$={_fmt_signed(_float(bu_res, "standardized_paired_effect_dz"))}; Holm p={_fmt_p(_float(bu_res, "holm_adjusted_p_value"))}). The estimate was small and close to the multiplicity threshold; it should not be read as a broad ranking beyond this protocol.

**Table 5. Frozen paired comparisons for the primary outcome.**

{_comparison_table(comparison_rows)}

{_figure("figures/final/fig04_paired_region_differences.png", "Figure 4. Patient-level paired regional Dice differences.", "fig:paired")}

{_figure("figures/final/fig05_primary_effects.png", "Figure 5. Frozen primary paired effects with 95% bootstrap confidence intervals.", "fig:effects")}

### 4.3 Regional, boundary, and lesion estimates

The largest regional overlap differences relative to standard U-Net occurred for TC. ET overlap changed less. Surface Dice means favored U-Net+RES numerically across all three regions. In contrast, finite HD95 means were lower for standard U-Net than for either component-based model; several TC and ET observations were infinite because one mask was empty. Thus, overlap improvements did not imply uniformly better boundary outlier behavior.

**Table 6. Selected secondary outcomes. HD95 means use finite observations; infinity counts are shown in parentheses.**

{_secondary_table(metric_rows)}

ET lesion recall was approximately 0.47 for all candidates, and ET lesion-wise Dice remained below region-level ET Dice. These estimates expose a lesion-detection limitation that a voxel-overlap summary alone would obscure.

### 4.4 Resource demand

BU-Net had {resource_means["bunet"]["parameter_count"] / resource_means["unet_reference"]["parameter_count"]:.2f} times the parameters and {resource_means["bunet"]["macs_per_slice"] / resource_means["unet_reference"]["macs_per_slice"]:.2f} times the MACs of standard U-Net. U-Net+RES achieved the highest primary mean with {resource_means["unet_res"]["parameter_count"] / 1e6:.3f} million parameters, compared with {resource_means["bunet"]["parameter_count"] / 1e6:.3f} million for BU-Net. Its mean p50 latency was {resource_means["unet_res"]["latency_p50_seconds"]:.3f} s/volume versus {resource_means["bunet"]["latency_p50_seconds"]:.3f} s/volume, and its mean peak allocated memory was {resource_means["unet_res"]["development_peak_allocated_vram_bytes"] / 1e6:.1f} MB versus {resource_means["bunet"]["development_peak_allocated_vram_bytes"] / 1e6:.1f} MB. Development accelerator-hours per run were similar and do not include preprocessing or study-level engineering time.

**Table 7. Architecture and measured resource profile on one host.**

{_resource_table(resource_rows, metric_rows)}

{_figure("figures/final/fig06_performance_resource_tradeoff.png", "Figure 6. Mean regional Dice in relation to model and runtime resource measures.", "fig:resources")}

### 4.5 Exploratory subgroups and qualitative review

Exploratory subgroup estimates are reported without confirmatory p values. HGG patients had higher mean scores than LGG patients for all models. The ET-absent subset contained five patients and is descriptive only. Training-set WT tertiles defined small (at most 64,267 mm3), medium (64,267 to 123,163.3 mm3), and large burden groups.

**Table 8. Exploratory mean regional Dice by frozen subgroup.**

{_subgroup_table(subgroup_rows)}

The qualitative panel includes preselected BU-Net success, hard, and failure cases: {qualitative["success"]["patient_id"]} (ET Dice {_fmt(qualitative["success"]["bunet_et_dice"])}), {qualitative["hard"]["patient_id"]} (ET Dice {_fmt(qualitative["hard"]["bunet_et_dice"])}), and {qualitative["failure"]["patient_id"]} (ET Dice {_fmt(qualitative["failure"]["bunet_et_dice"])}). The failure case demonstrates that a favorable cohort mean can coexist with near-zero ET overlap in an individual patient. Modality, prediction, false-positive, and false-negative panels were generated from retained test artifacts rather than selected after manuscript inspection.

{_figure("figures/final/fig07_modalities_and_predictions.png", "Figure 7. Multimodal MRI, reference labels, and frozen candidate predictions for an internal-test case.", "fig:modalities")}

{_figure("figures/final/fig08_false_positive_false_negative_overlay.png", "Figure 8. False-positive and false-negative error overlays.", "fig:errors")}

{_figure("figures/final/fig09_success_hard_failure_et_cases.png", "Figure 9. Prespecified BU-Net ET success, hard, and failure cases.", "fig:cases")}

{_figure("figures/final/figS01_split_balance.png", "Supplementary Figure S1. Patient-level balance across frozen partitions.", "fig:balance")}

## 5. Discussion

### 5.1 Principal findings

This controlled study supports three bounded conclusions. First, both BU-Net and U-Net+RES produced small patient-level improvements in mean regional Dice over standard U-Net after multiplicity correction. Second, U-Net+RES slightly exceeded the full BU-Net on the frozen primary endpoint. Third, BU-Net required substantially more parameters, MACs, latency, and allocated memory than U-Net+RES. Under this implementation and run budget, WC therefore did not provide an observable advantage over RES alone.

The effect sizes were small. BU-Net's paired standardized effect versus standard U-Net was {_fmt_signed(_float(bu_ref, "standardized_paired_effect_dz"))}, and the BU-Net versus U-Net+RES contrast was {_fmt_signed(_float(bu_res, "standardized_paired_effect_dz"))}. The latter p value was near 0.05 and its confidence interval lay close to zero. The evidence should be interpreted as a component-specific result under one bounded protocol, not a universal claim that WC is ineffective.

### 5.2 What the secondary outcomes add

Dice and surface Dice generally favored the component-based models, but finite HD95 means did not. Standard U-Net had lower finite HD95 means in WT, TC, and ET. This discordance can arise because Dice summarizes overlap while HD95 emphasizes distant boundary errors and becomes infinite when only one mask is empty. Reporting both prevents a one-dimensional account of segmentation quality.

Lesion-level estimates also temper the voxel-level results. ET lesion recall and lesion-wise Dice remained modest for every candidate. A model can achieve a high regional Dice on a large focus while missing small enhancing foci or generating disconnected false-positive components. The qualitative failure case reinforces this limitation.

### 5.3 Development ranking and test behavior

BU-Net ranked first during five-seed development, but U-Net+RES ranked first on the internal test. The development confidence interval between the finalists included zero, so freezing both was important. Selecting only the numerically leading validation model would have hidden this uncertainty. The result illustrates why seed replication, frozen finalist rules, and complete candidate reporting matter.

### 5.4 Scope relative to broader segmentation systems

nnU-Net, TransBTS, nnFormer, and other 3D systems are relevant modern comparators [9,14,15]. We did not train them. Nor did we perform cross-dataset or multi-institutional external validation. Literature scores cannot substitute for those experiments because acquisition, cohort, preprocessing, compute, and evaluation rules differ. Our results should therefore be read as a reproducible internal ablation of published BU-Net components, not as a leaderboard or state-of-the-art comparison.

### 5.5 Limitations

This study has several limitations. It used one public dataset and an internal split; external generalization is unknown. The networks were 2D and cannot use through-plane context as a 3D model can. Training was deliberately bounded at 2,000 optimizer steps or 0.5 accelerator-hours, so the results do not describe fully converged scaling behavior. The architecture widths and output encoding are reimplementation choices rather than a bit-exact reproduction of BU-Net. Resource measurements came from one Apple M1 Max host and do not establish deployment performance on clinical hardware. The 1 mm surface tolerance was predeclared but not calibrated against expert interobserver variability. HGG/LGG labels were inherited from the challenge release and were not independently adjudicated. Exploratory subgroups were small, especially the five-patient ET-absent subset. Finally, no prospective workflow, reader study, clinical endpoint, or regulatory assessment was performed.

### 5.6 Reproducibility and auditability

The released repository contains configuration files, split hashes, model and loss code, evaluator tests, analysis scripts, machine-readable outputs, figure and table generators, resource profiles, a claim ledger, and a tracked-artifact manifest. A clean-clone reproduction report records the exact commit and verifies deterministic output generation. Raw BraTS data and model checkpoints are excluded because redistribution is not authorized; users must obtain the dataset separately and configure local paths.

## 6. Conclusion

In a leakage-safe, patient-level, multi-seed evaluation of three matched 2D U-Net-family candidates, the published BU-Net RES component improved mean regional Dice relative to standard U-Net. Adding the published WC component in the full BU-Net did not improve the primary endpoint over RES alone and increased computational demand. Boundary and lesion-level results were more mixed than overlap scores. The work provides reproducible internal evidence about these components, while leaving 3D comparison, external validation, clinical calibration, and prospective utility as open questions.

## Data and code availability

Code, frozen manifests, derived reports, tables, figures, and reproduction instructions are available at https://github.com/TerekliTahaBerk/bratsarticle. BraTS images and labels are not redistributed. Access to the original dataset is governed by the BraTS data providers. The internal held-out test partition is a study-specific subset of the BraTS 2020 training cohort and is not an official challenge test set.

## Ethics statement

This computational study used a publicly distributed, de-identified challenge dataset and involved no direct participant contact or intervention. Authors should confirm the final journal-specific institutional review and exemption wording before submission.

## Declarations requiring author confirmation

Funding, competing interests, author contributions, and institutional review wording were not inferable from the repository and must be confirmed by the authors in the target journal's required format. No declaration has been invented in this manuscript package.

## References

1. Menze BH, et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*. 2015;34:1993-2024. doi:10.1109/TMI.2014.2377694.
2. Center for Biomedical Image Computing and Analytics. BraTS 2020 Tasks. University of Pennsylvania. https://www.med.upenn.edu/cbica/brats2020/tasks.html.
3. Bakas S, et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. *Scientific Data*. 2017;4:170117. doi:10.1038/sdata.2017.117.
4. Bakas S, et al. Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. arXiv:1811.02629. 2018.
5. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. 2015. doi:10.1007/978-3-319-24574-4_28.
6. Rehman ZU, et al. BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture. *Electronics*. 2020;9:2203. doi:10.3390/electronics9122203.
7. Salehi SSM, Erdogmus D, Gholipour A. Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. arXiv:1706.05721. 2017.
8. Abraham N, Khan NM. A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation. *IEEE ISBI*. 2019. doi:10.1109/ISBI.2019.8759329.
9. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18:203-211. doi:10.1038/s41592-020-01008-z.
10. Maier-Hein L, et al. Metrics Reloaded: recommendations for image analysis validation. *Nature Methods*. 2024;21:195-212. doi:10.1038/s41592-023-02151-z.
11. Nikolov S, et al. Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy: Deep Learning Algorithm Development and Validation Study. *Journal of Medical Internet Research*. 2021;23:e26151. doi:10.2196/26151.
12. Tejani AS, Klontzas ME, Gatti AA, Mongan JT, Moy L, Park SH, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. *Radiology: Artificial Intelligence*. 2024;6:e240300. doi:10.1148/ryai.240300.
13. Holm S. A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*. 1979;6:65-70. doi:10.2307/4615733.
14. Wang W, et al. TransBTS: Multimodal Brain Tumor Segmentation Using Transformer. arXiv:2103.04430. 2021.
15. Zhou H-Y, et al. nnFormer: Interleaved Transformer for Volumetric Segmentation. *IEEE Transactions on Image Processing*. 2023;32:4036-4049. doi:10.1109/TIP.2023.3293771.
"""


def _reviewer_response() -> str:
    metric_rows = _read_csv("reports/gate11_metric_summary.csv")
    comparison_rows = _read_csv("reports/gate11_comparisons.csv")
    bu_ref = _comparison(
        comparison_rows, "bunet_vs_unet_reference", "mean_regional_dice"
    )
    res_ref = _comparison(
        comparison_rows, "unet_res_vs_unet_reference", "mean_regional_dice"
    )
    return f"""# Response to the reviewer

**Manuscript:** *Leakage-Safe Multi-Seed Evaluation of Published BU-Net Components for Resource-Constrained 2D Glioma Segmentation*

We thank the reviewer for identifying problems that could not be corrected by editorial changes alone. We rebuilt the study around a canonical cohort, patient-level partitions, matched training, guarded test access, reproducible artifacts, and appropriately limited claims. The old cross-architecture ranking and clinical or state-of-the-art language were removed.

## 1. Attribution and novelty

**Concern:** RES and WC appeared to be presented as new contributions although they were introduced in BU-Net.

**Response:** Corrected throughout. RES, WC, and their combination are explicitly attributed to Rehman et al. (2020). The title, abstract, architecture figure, Methods, Discussion, and Conclusion describe this as an evaluation and reimplementation study. No architectural novelty is claimed. A dedicated implementation note records deliberate deviations from the source paper.

## 2. BraTS 2019/2020 overlap and leakage

**Concern:** Pooling BraTS editions could duplicate patients and contaminate splits.

**Response:** We did not pool the editions. BraTS 2020 training is the only canonical labeled cohort. BraTS 2019 is used only for content-based identity auditing. All 335 BraTS 2019 cases map to BraTS 2020; 34 cases are new in 2020. One mapped segmentation contains a two-voxel annotation revision. The final 369 patients were split at patient level into 258/37/74 training/validation/internal-test cases, with zero identifier or content-signature overlap.

## 3. Slice-level leakage

**Concern:** A slice-level split would allow images from one patient in several subsets.

**Response:** All partitions and statistical analyses now use the patient as the unit. Slices are sampled only within an already assigned training patient. The split manifests and hashes are frozen and tested automatically.

## 4. Unfair or inconsistent protocols

**Concern:** Architectures appeared to receive different losses, schedules, or training opportunities.

**Response:** Every final candidate now uses the same inputs, normalization, augmentation, BCE+FTL objective, AdamW optimizer, learning rate, weight decay, batch size, scheduler, device, and 2,000-step/0.5-hour cap. The protocol is intentionally bounded and this limitation is stated. Development screens, confirmation runs, and test evaluation have distinct roles and machine-readable registries.

## 5. Unsupported transformer or Swin conclusions

**Concern:** Undertrained transformer comparisons cannot support architectural conclusions.

**Response:** Removed. No Swin, transformer, attention, or literature score appears as an experimental comparator. nnU-Net, TransBTS, and nnFormer are cited only as important untested context. The manuscript explicitly states that no 3D, transformer, self-configuring, or external-validation comparison was performed.

## 6. Missing ablation

**Concern:** The contribution of individual components was unclear.

**Response:** The rebuilt design screens standard U-Net, U-Net+RES, U-Net+WC, BU-Net (RES+WC), residual-block U-Net, and residual-block U-Net+WC under one protocol. The final frozen comparison includes standard U-Net, U-Net+RES, and BU-Net. On the internal test, mean regional Dice is {_fmt(_float(_metric(metric_rows, "unet_reference", "mean_regional_dice"), "mean_finite"))}, {_fmt(_float(_metric(metric_rows, "unet_res", "mean_regional_dice"), "mean_finite"))}, and {_fmt(_float(_metric(metric_rows, "bunet", "mean_regional_dice"), "mean_finite"))}, respectively. The paired U-Net+RES versus standard U-Net difference is {_fmt_signed(_float(res_ref, "paired_mean_difference"))} (Holm p={_fmt_p(_float(res_ref, "holm_adjusted_p_value"))}); the BU-Net versus standard U-Net difference is {_fmt_signed(_float(bu_ref, "paired_mean_difference"))} (Holm p={_fmt_p(_float(bu_ref, "holm_adjusted_p_value"))}). We interpret this as evidence for RES under the present protocol, not architectural novelty.

## 7. Loss definition and selection

**Concern:** The loss formulation, class handling, and selection process were insufficiently specified.

**Response:** The Methods now provide the complete BCE+FTL formula and its parameters: equal term weights, alpha 0.3, beta 0.7, gamma 0.75, smoothing 1e-5, foreground-only FTL, and no class weights. Seven loss arms were screened once on the validation subset. Their full results are reported as development-only evidence.

## 8. Inconsistent metrics and aggregation

**Concern:** Metric definitions, empty-mask rules, and aggregation could change model rankings.

**Response:** The evaluator now fixes WT/TC/ET label mapping, Dice, HD95, surface Dice at 1 mm, lesion metrics, 26-connectivity, maximum-total-IoU lesion matching, and explicit empty-mask behavior. Seeds are averaged within patient before inference; patients are never multiplied into slice- or seed-level pseudo-samples. Infinite and undefined secondary values are retained and counted.

## 9. Statistics and uncertainty

**Concern:** Point estimates without patient-level uncertainty or multiplicity control were inadequate.

**Response:** The primary outcome and three contrasts were frozen before test access. We use 10,000 patient-level bootstrap resamples, 100,000 two-sided paired sign-flip permutations, paired effect size dz, and Holm correction. Regional and secondary endpoints are identified as estimation-only.

## 10. Resource and efficiency claims

**Concern:** “Lightweight,” “efficient,” or clinical deployment language lacked measurement.

**Response:** We removed those unqualified claims. The paper reports parameters, checkpoint size, MACs, FLOPs, p50/p95 volume latency, throughput, allocated/reserved memory, and development accelerator-hours from the implemented candidates on one host. BU-Net is explicitly shown to require more resources than standard U-Net and U-Net+RES. No clinical deployment claim remains.

## 11. Qualitative analysis and failure cases

**Concern:** The paper lacked image-level evidence and failure analysis.

**Response:** Artifact-derived figures now show modalities, ground truth, all frozen predictions, false-positive/false-negative overlays, and preselected success, hard, and failure cases. The failure case is discussed alongside modest ET lesion-recall and lesion-wise Dice estimates.

## 12. Reproducibility and code

**Concern:** The reported results could not be reproduced.

**Response:** The repository now contains configurations, model and loss implementations, evaluator tests, patient manifests and hashes, run registries, statistical scripts, derived tables and figures, environment lock, claim ledger, and reproduction instructions. A clean-clone audit verified 230 tracked-artifact hashes, static checks, the test suite, two byte-identical manuscript-input generations, and a clean worktree.

## 13. Requested 3D, nnU-Net, transformer, and external validation

**Concern:** Strong conclusions would require modern 3D/self-configuring baselines and external validation.

**Response:** We agree with the evidentiary requirement and narrowed the paper instead of fabricating or importing incomparable results. The revised manuscript does not claim superiority over these systems, state of the art, external generalization, or clinical applicability. These experiments remain future work and are prominent limitations.

## 14. Manuscript-wide claim correction

**Concern:** The original framing was broader than the evidence.

**Response:** The manuscript was rewritten. The conclusion is limited to a leakage-safe internal comparison: RES improved the primary overlap endpoint over standard U-Net under this bounded 2D protocol; adding WC in the full BU-Net did not improve that endpoint over RES alone and increased resource demand. Boundary and lesion-level findings are reported as mixed.
"""


def _claim_checklist() -> str:
    return """# CLAIM 2024-oriented reporting checklist

This is a repository audit aid, not an official CLAIM form. Item wording is summarized rather than copied.

| Reporting domain | Location or status |
|---|---|
| Title identifies AI/segmentation study | Title |
| Structured summary states design, data, outcome, and limits | Abstract |
| Clinical/scientific background | Introduction |
| Intended use is bounded to methodological evaluation | Introduction; Discussion |
| Data source and edition | Methods 3.2 |
| Canonical cohort and duplicate policy | Methods 3.2 |
| Inclusion/completeness accounting | Methods 3.2; Table 1 |
| Patient-level partitioning | Methods 3.2 |
| Split sizes and balance | Table 1; Figures 1, 2, S1 |
| Independence of partitions | Methods 3.2 |
| Reference-label definitions | Methods 3.8 |
| Image modalities and preprocessing | Methods 3.3 |
| Sampling and augmentation | Methods 3.3 |
| Model inputs and outputs | Methods 3.4 |
| Architecture provenance | Methods 3.4; Figure 3 |
| Model implementation deviations | Methods 3.4 |
| Loss formula and parameters | Methods 3.5 |
| Training optimizer and schedule | Methods 3.6 |
| Run budget and stopping rule | Methods 3.6 |
| Hyperparameter/selection data restricted to development | Methods 3.5-3.7 |
| Test-access guard | Methods 3.7 |
| Primary outcome prespecified | Methods 3.8-3.9 |
| Statistical unit is the patient | Methods 3.9 |
| Confidence intervals | Methods 3.9; Results |
| Multiplicity handling | Methods 3.9 |
| Missing, undefined, and infinite metric handling | Methods 3.8 |
| Subgroup status and sample sizes | Methods 3.9; Table 8 |
| Full candidate reporting | Results 4.2 |
| Error and failure analysis | Results 4.3, 4.5; Figures 8, 9 |
| Computational resource reporting | Methods 3.10; Table 7 |
| Code and derived artifact availability | Data and code availability |
| External validation status | Abstract; Discussion |
| Clinical validation status | Abstract; Discussion |
| Limitations | Discussion 5.5 |
| Funding | Author confirmation required |
| Competing interests | Author confirmation required |
| Author contributions | Author confirmation required |
| Ethics/IRB wording | Author and journal confirmation required |
"""


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manuscript/final_manuscript.md": _manuscript(),
        "manuscript/response_to_reviewer.md": _reviewer_response(),
        "manuscript/claim_2024_checklist.md": _claim_checklist(),
    }
    for relative_path, content in outputs.items():
        (ROOT / relative_path).write_text(content.rstrip() + "\n", encoding="utf-8")

    source_paths = [
        "reports/data_audit_summary.json",
        "splits/frozen/split_metadata.json",
        "reports/gate8_arm_summary.csv",
        "reports/gate9_confirmation_summary.csv",
        "reports/gate9_final_summary.csv",
        "reports/gate10_analysis_freeze.json",
        "reports/gate11_analysis.json",
        "reports/gate11_metric_summary.csv",
        "reports/gate11_comparisons.csv",
        "reports/gate11_resource_summary.csv",
        "reports/gate11_subgroups.csv",
        "literature/verified_sources.yaml",
    ]
    report = {
        "gate": 14,
        "status": "generated",
        "policy": "All scientific result values are rendered from tracked machine-readable artifacts.",
        "sources": {path: _sha256(path) for path in source_paths},
        "outputs": {path: _sha256(path) for path in outputs},
    }
    (REPORTS_DIR / "gate14_generation_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
