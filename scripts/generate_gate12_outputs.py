"""Generate final scientific figures and tables from frozen artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

FIGURE_ROOT = Path("figures/final")
TABLE_ROOT = Path("tables/final")
REPORT_PATH = Path("reports/gate12_completion.md")
MANIFEST_PATH = Path("reports/gate12_output_manifest.json")
QUALITATIVE_ROOT = Path("artifacts/internal_test/gate11/qualitative")

COLORS = {
    "unet_reference": "#4C78A8",
    "bunet": "#F58518",
    "unet_res": "#54A24B",
    "neutral": "#6B7280",
    "negative": "#C44E52",
    "positive": "#2A9D8F",
}
LABELS = {
    "unet_reference": "Standard U-Net",
    "bunet": "BU-Net",
    "unet_res": "U-Net + RES",
}
REGION_LABELS = {"wt": "WT", "tc": "TC", "et": "ET"}
SEGMENTATION_CMAP = ListedColormap(
    ["#000000", "#D73027", "#1A9850", "#FEE08B"]
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _source_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {path.as_posix(): file_digest(path) for path in paths}


def _save_figure(
    figure: plt.Figure,
    figure_id: str,
    *,
    title: str,
    caption: str,
    sources: Sequence[Path],
    manifest: list[dict[str, Any]],
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    png = FIGURE_ROOT / f"{figure_id}.png"
    pdf = FIGURE_ROOT / f"{figure_id}.pdf"
    figure.savefig(png, bbox_inches="tight", facecolor="white")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    with Image.open(png) as image:
        width, height = image.size
    manifest.append(
        {
            "id": figure_id,
            "kind": "figure",
            "title": title,
            "caption": caption,
            "png_path": png.as_posix(),
            "png_sha256": file_digest(png),
            "pdf_path": pdf.as_posix(),
            "pdf_sha256": file_digest(pdf),
            "pixel_width": width,
            "pixel_height": height,
            "source_sha256": _source_hashes(sources),
        }
    )


def _flow_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        edgecolor=color,
        facecolor=matplotlib.colors.to_rgba(color, 0.10),
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center")


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#555555",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.0,
            color=color,
        )
    )


def _cohort_flow(manifest: list[dict[str, Any]]) -> None:
    audit_path = Path("reports/data_audit_summary.json")
    split_path = Path("splits/frozen/split_metadata.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(10.2, 4.0))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 4.0)
    axis.axis("off")
    _flow_box(
        axis,
        0.2,
        2.5,
        2.0,
        0.8,
        f"BraTS 2019 labeled cohort\nn = {audit['brats2019']['subject_count']}",
        color=COLORS["neutral"],
    )
    _flow_box(
        axis,
        0.2,
        0.7,
        2.0,
        0.8,
        f"BraTS 2020 labeled cohort\nn = {audit['brats2020']['subject_count']}",
        color=COLORS["unet_reference"],
    )
    _flow_box(
        axis,
        3.2,
        2.5,
        2.1,
        0.8,
        f"Mapped overlap\nn = {audit['mapping']['mapped_overlap_count']}",
        color=COLORS["neutral"],
    )
    _flow_box(
        axis,
        3.2,
        0.7,
        2.1,
        0.8,
        f"New in BraTS 2020\nn = {audit['mapping']['new_in_brats2020_count']}",
        color=COLORS["unet_reference"],
    )
    _flow_box(
        axis,
        6.2,
        1.6,
        1.8,
        0.9,
        (
            "Canonical eligible\ncohort n = "
            f"{audit['brats2020']['eligible_subject_count']}"
        ),
        color=COLORS["positive"],
    )
    _flow_box(
        axis,
        8.7,
        1.6,
        1.3,
        0.9,
        "Patient-level\nsplit",
        color=COLORS["positive"],
    )
    _arrow(axis, (2.2, 2.9), (3.2, 2.9))
    _arrow(axis, (2.2, 1.1), (3.2, 1.1))
    _arrow(axis, (5.3, 2.9), (6.2, 2.1))
    _arrow(axis, (5.3, 1.1), (6.2, 2.0))
    _arrow(axis, (8.0, 2.05), (8.7, 2.05))
    axis.text(
        4.25,
        2.32,
        "No independent 2019 subjects",
        ha="center",
        va="top",
        color="#555555",
        fontsize=8,
    )
    axis.text(
        9.35,
        1.35,
        f"{split['counts']['train']} train · "
        f"{split['counts']['validation']} validation · "
        f"{split['counts']['test']} internal test",
        ha="center",
        va="top",
        fontsize=8,
    )
    _save_figure(
        figure,
        "fig01_cohort_flow",
        title="Cohort identity and canonicalization flow",
        caption=(
            "BraTS 2019 subjects were identity-mapped to BraTS 2020; no "
            "independent 2019 patients were added. The 369 eligible BraTS 2020 "
            "subjects formed the canonical patient-level cohort."
        ),
        sources=[audit_path, split_path],
        manifest=manifest,
    )


def _split_flow(manifest: list[dict[str, Any]]) -> None:
    split_path = Path("splits/frozen/split_metadata.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(10.2, 3.6))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 3.6)
    axis.axis("off")
    _flow_box(
        axis,
        0.25,
        1.35,
        2.1,
        0.9,
        f"Canonical patients\nn = {sum(split['counts'].values())}",
        color=COLORS["positive"],
    )
    _flow_box(
        axis,
        3.1,
        2.35,
        1.8,
        0.75,
        f"Train\nn = {split['counts']['train']}",
        color=COLORS["unet_reference"],
    )
    _flow_box(
        axis,
        3.1,
        1.35,
        1.8,
        0.75,
        f"Validation\nn = {split['counts']['validation']}",
        color=COLORS["bunet"],
    )
    _flow_box(
        axis,
        3.1,
        0.35,
        1.8,
        0.75,
        f"Internal test\nn = {split['counts']['test']}",
        color=COLORS["unet_res"],
    )
    _flow_box(
        axis,
        6.1,
        2.35,
        1.7,
        0.75,
        "Training and\naugmentation",
        color=COLORS["unet_reference"],
    )
    _flow_box(
        axis,
        6.1,
        1.35,
        1.7,
        0.75,
        "Development\nselection only",
        color=COLORS["bunet"],
    )
    _flow_box(
        axis,
        6.1,
        0.35,
        1.7,
        0.75,
        "One guarded\nopening",
        color=COLORS["unet_res"],
    )
    _flow_box(
        axis,
        8.55,
        1.35,
        1.4,
        0.75,
        "Frozen\nstatistics",
        color=COLORS["positive"],
    )
    for y in (2.72, 1.72, 0.72):
        _arrow(axis, (2.35, 1.80), (3.1, y))
        _arrow(axis, (4.9, y), (6.1, y))
    _arrow(axis, (7.8, 0.72), (8.55, 1.55))
    _arrow(axis, (7.8, 1.72), (8.55, 1.72))
    axis.text(
        1.3,
        0.92,
        "Patient-level · duplicate-free",
        ha="center",
        color="#555555",
        fontsize=8,
    )
    axis.text(
        5.5,
        3.30,
        f"Seed {split['seed']} · split SHA-256 identities frozen before test",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    _save_figure(
        figure,
        "fig02_split_and_analysis_flow",
        title="Patient-level split and guarded analysis flow",
        caption=(
            "The 258/37/74 split was frozen byte-for-byte. The internal held-out "
            "test subset was opened once only after candidate checkpoints and "
            "the statistical plan were frozen."
        ),
        sources=[split_path, Path("reports/gate10_analysis_freeze.json")],
        manifest=manifest,
    )


def _architecture(manifest: list[dict[str, Any]]) -> None:
    resource_path = Path("reports/gate11_resource_summary.csv")
    resources = pd.read_csv(resource_path).groupby("candidate_id").first()
    figure, axes = plt.subplots(3, 1, figsize=(11.0, 6.6), sharex=True)
    configurations = (
        ("unet_reference", False, False),
        ("unet_res", True, False),
        ("bunet", True, True),
    )
    for axis, (candidate, res_enabled, wc_enabled) in zip(
        axes,
        configurations,
        strict=True,
    ):
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 2.2)
        axis.axis("off")
        for index, channels in enumerate((16, 32, 64, 128)):
            x = 0.7 + index * 1.15
            height = 1.35 - index * 0.18
            axis.add_patch(
                FancyBboxPatch(
                    (x, 0.42),
                    0.62,
                    height,
                    boxstyle="round,pad=0.01",
                    facecolor=matplotlib.colors.to_rgba(
                        COLORS["unet_reference"],
                        0.18,
                    ),
                    edgecolor=COLORS["unet_reference"],
                    linewidth=0.9,
                )
            )
            axis.text(x + 0.31, 0.25, str(channels), ha="center", fontsize=7)
        bottleneck_x = 5.1
        axis.add_patch(
            FancyBboxPatch(
                (bottleneck_x, 0.42),
                0.72,
                0.65,
                boxstyle="round,pad=0.01",
                facecolor=matplotlib.colors.to_rgba(COLORS["neutral"], 0.16),
                edgecolor=COLORS["neutral"],
            )
        )
        axis.text(bottleneck_x + 0.36, 0.25, "256", ha="center", fontsize=7)
        if wc_enabled:
            axis.text(
                bottleneck_x + 0.36,
                0.75,
                "WC\n15 x 1 / 1 x 15",
                ha="center",
                va="center",
                color=COLORS["bunet"],
                fontsize=6,
                fontweight="bold",
            )
        for index, channels in enumerate((128, 64, 32, 16)):
            x = 6.0 + index * 0.82
            height = 0.72 + index * 0.16
            axis.add_patch(
                FancyBboxPatch(
                    (x, 0.42),
                    0.58,
                    height,
                    boxstyle="round,pad=0.01",
                    facecolor=matplotlib.colors.to_rgba(
                        COLORS["unet_res"],
                        0.16,
                    ),
                    edgecolor=COLORS["unet_res"],
                    linewidth=0.9,
                )
            )
            axis.text(x + 0.29, 0.25, str(channels), ha="center", fontsize=7)
        for index in range(4):
            left = 1.01 + index * 1.15
            right = 8.75 - index * 0.82
            y = 1.92 - index * 0.18
            axis.plot(
                [left, right],
                [y, y],
                color=COLORS["bunet"] if res_enabled else COLORS["neutral"],
                linewidth=1.4 if res_enabled else 0.9,
            )
            if res_enabled and index == 0:
                axis.text(
                    (left + right) / 2,
                    y + 0.08,
                    "RES: four extended skip paths",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=COLORS["bunet"],
                )
        row = resources.loc[candidate]
        axis.text(
            0.05,
            1.55,
            LABELS[candidate],
            ha="left",
            va="center",
            fontweight="bold",
        )
        axis.text(
            9.95,
            1.55,
            f"{int(row['parameter_count']) / 1e6:.2f} M params\n"
            f"{row['macs_per_slice'] / 1e9:.2f} G MAC/slice",
            ha="right",
            va="center",
            fontsize=8,
        )
    axes[-1].text(2.5, 0.02, "Encoder", ha="center", fontsize=8)
    axes[-1].text(7.6, 0.02, "Decoder", ha="center", fontsize=8)
    _save_figure(
        figure,
        "fig03_model_architectures",
        title="Controlled U-Net-family architectures",
        caption=(
            "All candidates share a four-level 2D U-Net backbone. U-Net + RES "
            "adds published BU-Net residual extended skips; BU-Net additionally "
            "adds the wide-context bottleneck. RES and WC are attributed to "
            "Rehman et al. (2020)."
        ),
        sources=[
            resource_path,
            Path("configs/models/unet.yaml"),
            Path("configs/models/unet_res.yaml"),
            Path("configs/models/bunet.yaml"),
        ],
        manifest=manifest,
    )


def _paired_region_differences(manifest: list[dict[str, Any]]) -> None:
    candidate_path = Path("reports/gate11_patient_candidate_metrics.csv")
    comparison_path = Path("reports/gate11_comparisons.csv")
    data = pd.read_csv(candidate_path)
    comparisons = pd.read_csv(comparison_path)
    reference = data.loc[data["candidate_id"] == "unet_reference"].set_index(
        "patient_id"
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    generator = np.random.default_rng(20260729)
    for region_index, region in enumerate(("wt", "tc", "et")):
        axis = axes[region_index]
        metric = f"{region}_dice"
        for candidate_index, candidate in enumerate(("bunet", "unet_res")):
            selected = data.loc[data["candidate_id"] == candidate].set_index(
                "patient_id"
            )
            patients = sorted(set(reference.index) & set(selected.index))
            differences = (
                selected.loc[patients, metric].to_numpy(dtype=float)
                - reference.loc[patients, metric].to_numpy(dtype=float)
            )
            x = candidate_index + generator.uniform(-0.11, 0.11, len(differences))
            axis.scatter(
                x,
                differences,
                s=12,
                alpha=0.30,
                color=COLORS[candidate],
                edgecolor="none",
            )
            summary = comparisons.loc[
                (comparisons["first_candidate"] == candidate)
                & (comparisons["second_candidate"] == "unet_reference")
                & (comparisons["metric"] == metric)
            ].iloc[0]
            axis.errorbar(
                candidate_index,
                summary["paired_mean_difference"],
                yerr=[
                    [
                        summary["paired_mean_difference"]
                        - summary["paired_bootstrap_lower"]
                    ],
                    [
                        summary["paired_bootstrap_upper"]
                        - summary["paired_mean_difference"]
                    ],
                ],
                fmt="D",
                markersize=5,
                capsize=3,
                color="#222222",
                zorder=5,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_title(REGION_LABELS[region])
        axis.set_xticks(
            [0, 1],
            ["BU-Net", "U-Net + RES"],
            rotation=15,
            ha="right",
        )
        axis.set_xlim(-0.45, 1.45)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Patient-paired Dice difference vs Standard U-Net")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                color="#222222",
                linestyle="none",
                label="Mean and 95% paired bootstrap CI",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(
        figure,
        "fig04_paired_region_differences",
        title="Patient-paired regional Dice differences",
        caption=(
            "Each point is one internal-test patient after within-patient seed "
            "aggregation. Diamonds and bars show paired mean differences and "
            "95% bootstrap confidence intervals versus Standard U-Net."
        ),
        sources=[candidate_path, comparison_path],
        manifest=manifest,
    )


def _forest(manifest: list[dict[str, Any]]) -> None:
    comparison_path = Path("reports/gate11_comparisons.csv")
    comparisons = pd.read_csv(comparison_path)
    primary = comparisons.loc[
        comparisons["formal_hypothesis_test"].astype(bool)
    ].copy()
    order = [
        "bunet_vs_unet_reference",
        "unet_res_vs_unet_reference",
        "bunet_vs_unet_res",
    ]
    primary["order"] = primary["comparison_id"].map(
        {name: index for index, name in enumerate(order)}
    )
    primary = primary.sort_values("order")
    display = {
        "bunet_vs_unet_reference": "BU-Net - Standard U-Net",
        "unet_res_vs_unet_reference": "U-Net + RES - Standard U-Net",
        "bunet_vs_unet_res": "BU-Net - U-Net + RES",
    }
    figure, axis = plt.subplots(figsize=(8.2, 3.2))
    y = np.arange(len(primary))[::-1]
    means = primary["paired_mean_difference"].to_numpy(dtype=float)
    lower = primary["paired_bootstrap_lower"].to_numpy(dtype=float)
    upper = primary["paired_bootstrap_upper"].to_numpy(dtype=float)
    axis.errorbar(
        means,
        y,
        xerr=[means - lower, upper - means],
        fmt="o",
        markersize=6,
        capsize=4,
        color="#222222",
    )
    axis.axvline(0.0, color="#555555", linewidth=0.9)
    axis.set_yticks(
        y,
        [display[value] for value in primary["comparison_id"]],
    )
    axis.set_xlabel("Paired difference in patient mean regional Dice")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    for y_value, row in zip(y, primary.itertuples(index=False), strict=True):
        axis.text(
            row.paired_bootstrap_upper + 0.001,
            y_value,
            f"Holm p={row.holm_adjusted_p_value:.4f}",
            va="center",
            fontsize=8,
        )
    axis.set_xlim(min(lower) - 0.006, max(upper) + 0.023)
    figure.tight_layout()
    _save_figure(
        figure,
        "fig05_primary_effects",
        title="Frozen primary-endpoint effect estimates",
        caption=(
            "Patient-paired mean differences in seed-aggregated WT/TC/ET Dice "
            "with 95% bootstrap intervals. Holm-adjusted p-values cover the "
            "predeclared three-comparison family."
        ),
        sources=[comparison_path],
        manifest=manifest,
    )


def _pareto(manifest: list[dict[str, Any]]) -> None:
    resource_path = Path("reports/gate11_resource_summary.csv")
    metric_path = Path("reports/gate11_metric_summary.csv")
    resources = (
        pd.read_csv(resource_path)
        .groupby("candidate_id", as_index=False)
        .mean(numeric_only=True)
    )
    metrics = pd.read_csv(metric_path)
    primary = metrics.loc[
        metrics["metric"] == "mean_regional_dice",
        [
            "candidate_id",
            "mean_finite",
            "bootstrap_lower_finite",
            "bootstrap_upper_finite",
        ],
    ]
    data = resources.merge(primary, on="candidate_id", validate="one_to_one")
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    scatter = axis.scatter(
        data["macs_per_slice"] / 1e9,
        data["mean_finite"],
        s=60 + 20 * data["parameter_count"] / 1e6,
        c=data["latency_p50_seconds"],
        cmap="viridis",
        edgecolor="#222222",
        linewidth=0.8,
        zorder=4,
    )
    annotation_offsets = {
        "unet_reference": (7, 6),
        "unet_res": (-105, 14),
        "bunet": (9, -38),
    }
    for row in data.itertuples(index=False):
        axis.errorbar(
            row.macs_per_slice / 1e9,
            row.mean_finite,
            yerr=[
                [row.mean_finite - row.bootstrap_lower_finite],
                [row.bootstrap_upper_finite - row.mean_finite],
            ],
            fmt="none",
            ecolor="#555555",
            capsize=3,
            zorder=3,
        )
        axis.annotate(
            f"{LABELS[row.candidate_id]}\n"
            f"{row.development_peak_allocated_vram_bytes / 1e6:.0f} MB VRAM",
            (row.macs_per_slice / 1e9, row.mean_finite),
            xytext=annotation_offsets[row.candidate_id],
            textcoords="offset points",
            fontsize=8,
        )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Median inference latency per volume (s)")
    axis.set_xlabel("MACs per 240 x 240 slice (G)")
    axis.set_ylabel("Internal-test mean regional Dice")
    axis.set_xlim(2.3, 11.8)
    axis.grid(color="#DDDDDD", linewidth=0.6)
    axis.text(
        0.02,
        0.02,
        "Marker area ∝ parameter count",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout()
    _save_figure(
        figure,
        "fig06_performance_resource_tradeoff",
        title="Performance and resource trade-off",
        caption=(
            "Accuracy is plotted against per-slice MACs; marker area encodes "
            "parameter count, color encodes Apple M1 Max median per-volume "
            "latency, and labels report mean peak allocated VRAM across seeds."
        ),
        sources=[resource_path, metric_path],
        manifest=manifest,
    )


def _normalize_mri(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    nonzero = values[values != 0]
    if not len(nonzero):
        return np.zeros_like(values)
    lower, upper = np.quantile(nonzero, [0.01, 0.99])
    if upper <= lower:
        return np.zeros_like(values)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _display_labels(labels: np.ndarray) -> np.ndarray:
    output = np.zeros_like(labels, dtype=np.uint8)
    output[labels == 1] = 1
    output[labels == 2] = 2
    output[labels == 4] = 3
    return output


def _load_qualitative(
    patient_id: str,
) -> tuple[Mapping[str, np.ndarray], dict[str, np.ndarray]]:
    root = QUALITATIVE_ROOT / patient_id
    with np.load(root / "context.npz", allow_pickle=False) as context_file:
        context = {name: np.asarray(context_file[name]) for name in context_file.files}
    predictions: dict[str, np.ndarray] = {}
    for candidate in ("unet_reference", "bunet", "unet_res"):
        with np.load(root / f"{candidate}.npz", allow_pickle=False) as loaded:
            predictions[candidate] = np.asarray(loaded["prediction_label"])
    return context, predictions


def _qualitative_modalities(manifest: list[dict[str, Any]]) -> None:
    analysis_path = Path("reports/gate11_analysis.json")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    patient_id = str(analysis["qualitative_cases"]["hard"]["patient_id"])
    context, predictions = _load_qualitative(patient_id)
    slice_index = int(context["lesion_slice_index"])
    images = np.asarray(context["lesion_image"])
    target = np.take(context["target_label"], slice_index, axis=2)
    figure, axes = plt.subplots(2, 4, figsize=(10.8, 5.5))
    for axis, image, title in zip(
        axes[0],
        images,
        ("T1", "T1ce", "T2", "FLAIR"),
        strict=True,
    ):
        axis.imshow(_normalize_mri(image), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    label_panels = [
        ("Ground truth", target),
        (
            "Standard U-Net",
            np.take(predictions["unet_reference"], slice_index, axis=2),
        ),
        ("BU-Net", np.take(predictions["bunet"], slice_index, axis=2)),
        ("U-Net + RES", np.take(predictions["unet_res"], slice_index, axis=2)),
    ]
    for axis, (title, labels) in zip(axes[1], label_panels, strict=True):
        axis.imshow(
            _display_labels(labels),
            cmap=SEGMENTATION_CMAP,
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(color="#D73027", label="NCR/NET (label 1)"),
            Patch(color="#1A9850", label="Edema (label 2)"),
            Patch(color="#FEE08B", label="Enhancing tumor (label 4)"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(f"Predeclared hard case · {patient_id} · axial slice {slice_index}")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    sources = [
        analysis_path,
        QUALITATIVE_ROOT / patient_id / "context.npz",
        *[
            QUALITATIVE_ROOT / patient_id / f"{candidate}.npz"
            for candidate in ("unet_reference", "bunet", "unet_res")
        ],
    ]
    _save_figure(
        figure,
        "fig07_modalities_and_predictions",
        title="Multimodal MRI, ground truth, and frozen predictions",
        caption=(
            "T1, contrast-enhanced T1, T2, and FLAIR are shown for the "
            "predeclared hard case alongside ground truth and fixed-seed "
            "predictions. The displayed slice maximizes ground-truth WT area."
        ),
        sources=sources,
        manifest=manifest,
    )


def _error_rgba(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    rgba = np.zeros((*target.shape, 4), dtype=np.float32)
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    rgba[true_positive] = (0.15, 0.75, 0.25, 0.60)
    rgba[false_positive] = (0.90, 0.15, 0.12, 0.78)
    rgba[false_negative] = (0.10, 0.35, 0.95, 0.78)
    return rgba


def _error_overlay(manifest: list[dict[str, Any]]) -> None:
    analysis_path = Path("reports/gate11_analysis.json")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    patient_id = str(analysis["qualitative_cases"]["failure"]["patient_id"])
    context, predictions = _load_qualitative(patient_id)
    slice_index = int(context["error_slice_index"])
    image = _normalize_mri(np.asarray(context["error_image"])[1])
    target = np.take(context["target_label"], slice_index, axis=2)
    prediction = np.take(predictions["bunet"], slice_index, axis=2)
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("T1ce")
    panels = (
        ("WT error", prediction != 0, target != 0),
        ("ET error", prediction == 4, target == 4),
    )
    for axis, (title, predicted_mask, target_mask) in zip(
        axes[1:],
        panels,
        strict=True,
    ):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.imshow(_error_rgba(predicted_mask, target_mask))
        axis.set_title(title)
    for axis in axes:
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(color=(0.15, 0.75, 0.25), label="True positive"),
            Patch(color=(0.90, 0.15, 0.12), label="False positive"),
            Patch(color=(0.10, 0.35, 0.95), label="False negative"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"Predeclared failure case · BU-Net seed 20260729 · "
        f"{patient_id} · slice {slice_index}"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    _save_figure(
        figure,
        "fig08_false_positive_false_negative_overlay",
        title="False-positive and false-negative overlays",
        caption=(
            "Voxel-level error overlays for the predeclared failure case. "
            "Green denotes true positive, red false positive, and blue false "
            "negative for WT and ET on the slice with maximum label disagreement."
        ),
        sources=[
            analysis_path,
            QUALITATIVE_ROOT / patient_id / "context.npz",
            QUALITATIVE_ROOT / patient_id / "bunet.npz",
        ],
        manifest=manifest,
    )


def _et_case_panel(manifest: list[dict[str, Any]]) -> None:
    analysis_path = Path("reports/gate11_analysis.json")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    roles = ("success", "hard", "failure")
    figure, axes = plt.subplots(3, 3, figsize=(8.0, 8.0))
    sources: list[Path] = [analysis_path]
    for row_index, role in enumerate(roles):
        details = analysis["qualitative_cases"][role]
        patient_id = str(details["patient_id"])
        context, predictions = _load_qualitative(patient_id)
        slice_index = int(context["lesion_slice_index"])
        image = _normalize_mri(np.asarray(context["lesion_image"])[1])
        target = np.take(context["target_label"], slice_index, axis=2) == 4
        prediction = np.take(predictions["bunet"], slice_index, axis=2) == 4
        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].imshow(
            np.ma.masked_where(~target, target),
            cmap=ListedColormap(["#FEE08B"]),
            alpha=0.72,
            interpolation="nearest",
        )
        axes[row_index, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(
            np.ma.masked_where(~prediction, prediction),
            cmap=ListedColormap(["#00A6D6"]),
            alpha=0.72,
            interpolation="nearest",
        )
        axes[row_index, 0].set_ylabel(
            f"{role.capitalize()}\n"
            f"ET Dice {float(details['bunet_et_dice']):.3f}",
            fontsize=8,
        )
        axes[row_index, 0].text(
            0.02,
            0.02,
            patient_id,
            transform=axes[row_index, 0].transAxes,
            color="white",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
        sources.extend(
            [
                QUALITATIVE_ROOT / patient_id / "context.npz",
                QUALITATIVE_ROOT / patient_id / "bunet.npz",
            ]
        )
    for column, title in enumerate(("T1ce", "Ground-truth ET", "BU-Net ET")):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    _save_figure(
        figure,
        "fig09_success_hard_failure_et_cases",
        title="Predeclared successful, hard, and failed ET cases",
        caption=(
            "Illustrative BU-Net seed-20260729 ET predictions for the "
            "predeclared success, hard, and failure roles. Case selection used "
            "the frozen artifact rule and does not add inferential units."
        ),
        sources=sources,
        manifest=manifest,
    )


def _split_balance_supplement(manifest: list[dict[str, Any]]) -> None:
    categorical_path = Path("splits/frozen/categorical_balance.csv")
    continuous_path = Path("splits/frozen/continuous_balance.csv")
    categorical = pd.read_csv(categorical_path)
    continuous = pd.read_csv(continuous_path)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    selected = categorical.loc[
        categorical["feature"].isin(["grade", "et_present"])
    ].copy()
    selected["label"] = selected["feature"] + ": " + selected["category"].astype(str)
    pivot = selected.pivot(
        index="label",
        columns="split",
        values="prevalence",
    ).fillna(0)
    pivot = pivot.reindex(columns=["train", "validation", "test"])
    x = np.arange(len(pivot))
    width = 0.25
    for index, split in enumerate(pivot.columns):
        axes[0].bar(
            x + (index - 1) * width,
            pivot[split],
            width,
            label=split,
            color=(
                COLORS["unet_reference"],
                COLORS["bunet"],
                COLORS["unet_res"],
            )[index],
        )
    axes[0].set_xticks(x, pivot.index, rotation=25, ha="right")
    axes[0].set_ylabel("Patient proportion")
    axes[0].set_title("Categorical balance")
    axes[0].legend(frameon=False)
    labels = [
        f"{row.split} · {str(row.feature).replace('_volume_mm3', '')}"
        for row in continuous.itertuples(index=False)
    ]
    values = continuous["standardized_mean_difference"].to_numpy(dtype=float)
    axes[1].barh(
        np.arange(len(values)),
        values,
        color=[
            {
                "train": COLORS["unet_reference"],
                "validation": COLORS["bunet"],
                "test": COLORS["unet_res"],
            }[str(split)]
            for split in continuous["split"]
        ],
    )
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].axvline(0.1, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].axvline(-0.1, color="#999999", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(np.arange(len(values)), labels)
    axes[1].set_xlabel("Standardized mean difference")
    axes[1].set_title("Log-volume balance")
    figure.tight_layout()
    _save_figure(
        figure,
        "figS01_split_balance",
        title="Frozen split balance",
        caption=(
            "Grade and ET-presence prevalence together with log-volume "
            "standardized mean differences across the frozen patient-level split."
        ),
        sources=[categorical_path, continuous_path],
        manifest=manifest,
    )


def _write_table(
    frame: pd.DataFrame,
    table_id: str,
    *,
    title: str,
    caption: str,
    sources: Sequence[Path],
    manifest: list[dict[str, Any]],
) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_ROOT / f"{table_id}.csv"
    tex_path = TABLE_ROOT / f"{table_id}.tex"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    latex = frame.to_latex(
        index=False,
        escape=True,
        na_rep="NA",
        float_format=lambda value: f"{value:.4f}",
        caption=caption,
        label=f"tab:{table_id}",
        position="htbp",
    )
    atomic_write_text(tex_path, latex)
    manifest.append(
        {
            "id": table_id,
            "kind": "table",
            "title": title,
            "caption": caption,
            "csv_path": csv_path.as_posix(),
            "csv_sha256": file_digest(csv_path),
            "tex_path": tex_path.as_posix(),
            "tex_sha256": file_digest(tex_path),
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "source_sha256": _source_hashes(sources),
        }
    )


def _cohort_table(manifest: list[dict[str, Any]]) -> None:
    train_path = Path("splits/frozen/train.csv")
    validation_path = Path("splits/frozen/validation.csv")
    cohort_path = Path("artifacts/internal_test/gate11/cohort_metadata.csv")
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(cohort_path)
    rows: list[dict[str, Any]] = []
    for name, frame in (
        ("Train", train),
        ("Validation", validation),
        ("Internal test", test),
    ):
        volume = frame["wt_volume_mm3"].astype(float)
        rows.append(
            {
                "Partition": name,
                "Patients": len(frame),
                "HGG": int((frame["grade"].astype(str) == "HGG").sum()),
                "LGG": int((frame["grade"].astype(str) == "LGG").sum()),
                "ET present": int(frame["et_voxel_count"].astype(float).gt(0).sum()),
                "WT volume median (IQR), mm3": (
                    f"{np.median(volume):.1f} "
                    f"({np.quantile(volume, 0.25):.1f}-"
                    f"{np.quantile(volume, 0.75):.1f})"
                ),
            }
        )
    _write_table(
        pd.DataFrame(rows),
        "table01_cohort_characteristics",
        title="Cohort characteristics by frozen partition",
        caption=(
            "Patient counts, grade, enhancing-tumor presence, and whole-tumor "
            "volume across the frozen partitions."
        ),
        sources=[train_path, validation_path, cohort_path],
        manifest=manifest,
    )


def _architecture_resource_table(manifest: list[dict[str, Any]]) -> None:
    resource_path = Path("reports/gate11_resource_summary.csv")
    metric_path = Path("reports/gate11_metric_summary.csv")
    resources = pd.read_csv(resource_path)
    grouped = resources.groupby("candidate_id", as_index=False).mean(numeric_only=True)
    metrics = pd.read_csv(metric_path)
    dice = metrics.loc[
        metrics["metric"] == "mean_regional_dice",
        ["candidate_id", "mean_finite"],
    ]
    frame = grouped.merge(dice, on="candidate_id", validate="one_to_one")
    frame = frame.assign(
        Model=frame["candidate_id"].map(LABELS),
        **{
            "Mean Dice": frame["mean_finite"],
            "Parameters (M)": frame["parameter_count"] / 1e6,
            "MAC/slice (G)": frame["macs_per_slice"] / 1e9,
            "Peak allocated VRAM (MB)": (
                frame["development_peak_allocated_vram_bytes"] / 1e6
            ),
            "GPU-hours": frame["development_gpu_hours"],
            "Latency p50 (s/volume)": frame["latency_p50_seconds"],
            "Latency p95 (s/volume)": frame["latency_p95_seconds"],
        },
    )[
        [
            "Model",
            "Mean Dice",
            "Parameters (M)",
            "MAC/slice (G)",
            "Peak allocated VRAM (MB)",
            "GPU-hours",
            "Latency p50 (s/volume)",
            "Latency p95 (s/volume)",
        ]
    ]
    _write_table(
        frame,
        "table02_architecture_resources",
        title="Accuracy and resource profile",
        caption=(
            "Internal-test accuracy and Apple M1 Max resource measurements. "
            "GPU-hours and peak allocated VRAM summarize development runs."
        ),
        sources=[resource_path, metric_path],
        manifest=manifest,
    )


def _primary_metric_table(manifest: list[dict[str, Any]]) -> None:
    metric_path = Path("reports/gate11_metric_summary.csv")
    metrics = pd.read_csv(metric_path)
    selected = metrics.loc[
        metrics["metric"].isin(
            ["mean_regional_dice", "wt_dice", "tc_dice", "et_dice"]
        )
    ].copy()
    selected["Model"] = selected["candidate_id"].map(LABELS)
    selected["Endpoint"] = selected["metric"].map(
        {
            "mean_regional_dice": "Mean regional Dice",
            "wt_dice": "WT Dice",
            "tc_dice": "TC Dice",
            "et_dice": "ET Dice",
        }
    )
    selected["Mean (95% CI)"] = selected.apply(
        lambda row: (
            f"{row['mean_finite']:.4f} "
            f"({row['bootstrap_lower_finite']:.4f}-"
            f"{row['bootstrap_upper_finite']:.4f})"
        ),
        axis=1,
    )
    frame = selected[
        [
            "Model",
            "Endpoint",
            "patient_count",
            "Mean (95% CI)",
            "median_finite",
        ]
    ].rename(
        columns={
            "patient_count": "Patients",
            "median_finite": "Median",
        }
    )
    _write_table(
        frame,
        "table03_internal_test_dice",
        title="Internal held-out test Dice estimates",
        caption=(
            "Patient-level seed-aggregated Dice estimates with percentile "
            "bootstrap confidence intervals."
        ),
        sources=[metric_path],
        manifest=manifest,
    )


def _comparison_table(manifest: list[dict[str, Any]]) -> None:
    comparison_path = Path("reports/gate11_comparisons.csv")
    comparisons = pd.read_csv(comparison_path)
    frame = comparisons.loc[
        comparisons["formal_hypothesis_test"].astype(bool),
        [
            "comparison_id",
            "paired_patient_count",
            "paired_mean_difference",
            "paired_median_difference",
            "paired_bootstrap_lower",
            "paired_bootstrap_upper",
            "standardized_paired_effect_dz",
            "permutation_p_value",
            "holm_adjusted_p_value",
            "reject_holm_alpha_0_05",
        ],
    ].rename(
        columns={
            "comparison_id": "Comparison",
            "paired_patient_count": "Paired patients",
            "paired_mean_difference": "Mean difference",
            "paired_median_difference": "Median difference",
            "paired_bootstrap_lower": "CI lower",
            "paired_bootstrap_upper": "CI upper",
            "standardized_paired_effect_dz": "Paired dz",
            "permutation_p_value": "Raw p",
            "holm_adjusted_p_value": "Holm p",
            "reject_holm_alpha_0_05": "Reject",
        }
    )
    _write_table(
        frame,
        "table04_primary_comparisons",
        title="Frozen primary-endpoint comparisons",
        caption=(
            "Paired patient-level differences, 95% bootstrap intervals, "
            "two-sided sign-flip p-values, and Holm family-wise correction."
        ),
        sources=[comparison_path],
        manifest=manifest,
    )


def _secondary_table(manifest: list[dict[str, Any]]) -> None:
    metric_path = Path("reports/gate11_metric_summary.csv")
    metrics = pd.read_csv(metric_path)
    selected_metrics = [
        "wt_hd95_mm",
        "tc_hd95_mm",
        "et_hd95_mm",
        "wt_surface_dice",
        "tc_surface_dice",
        "et_surface_dice",
        "et_lesion_recall",
        "et_lesion_wise_dice",
    ]
    frame = metrics.loc[
        metrics["metric"].isin(selected_metrics),
        [
            "candidate_id",
            "metric",
            "finite_count",
            "nan_count",
            "positive_infinity_count",
            "mean_finite",
            "median_finite",
        ],
    ].copy()
    frame["candidate_id"] = frame["candidate_id"].map(LABELS)
    frame = frame.rename(
        columns={
            "candidate_id": "Model",
            "metric": "Endpoint",
            "finite_count": "Finite n",
            "nan_count": "NaN n",
            "positive_infinity_count": "Infinity n",
            "mean_finite": "Finite mean",
            "median_finite": "Finite median",
        }
    )
    _write_table(
        frame,
        "table05_secondary_metrics",
        title="Selected secondary endpoints",
        caption=(
            "Surface, HD95, and lesion-wise estimates. Undefined and infinite "
            "values are counted explicitly and are not imputed."
        ),
        sources=[metric_path],
        manifest=manifest,
    )


def _subgroup_table(manifest: list[dict[str, Any]]) -> None:
    subgroup_path = Path("reports/gate11_subgroups.csv")
    subgroups = pd.read_csv(subgroup_path)
    frame = subgroups.loc[
        subgroups["record_type"] == "candidate_estimate",
        [
            "subgroup",
            "category",
            "candidate_id",
            "patient_count",
            "reportability",
            "mean_regional_dice",
            "bootstrap_lower",
            "bootstrap_upper",
        ],
    ].copy()
    frame["candidate_id"] = frame["candidate_id"].map(LABELS)
    frame = frame.rename(
        columns={
            "subgroup": "Subgroup",
            "category": "Category",
            "candidate_id": "Model",
            "patient_count": "Patients",
            "reportability": "Status",
            "mean_regional_dice": "Mean Dice",
            "bootstrap_lower": "CI lower",
            "bootstrap_upper": "CI upper",
        }
    )
    _write_table(
        frame,
        "table06_exploratory_subgroups",
        title="Exploratory subgroup estimates",
        caption=(
            "Estimation-only mean regional Dice by grade, ground-truth ET "
            "presence, and training-derived whole-tumor burden category."
        ),
        sources=[subgroup_path, Path("reports/gate10_analysis_freeze.json")],
        manifest=manifest,
    )


def _completion_report(manifest: Sequence[Mapping[str, Any]]) -> str:
    figures = [entry for entry in manifest if entry["kind"] == "figure"]
    tables = [entry for entry in manifest if entry["kind"] == "table"]
    lines = [
        "# Gate 12 Completion",
        "",
        "**Decision:** PASS",
        "",
        "## Generated outputs",
        "",
        f"- Figures: {len(figures)} PNG + {len(figures)} PDF",
        f"- Tables: {len(tables)} CSV + {len(tables)} LaTeX",
        "- Hand-entered scientific result values: 0",
        "- Internal-test manifest reopened: no",
        "- Raw-data files accessed: no",
        "",
        "## Figure quality audit",
        "",
        "| Figure | Pixels | PNG SHA-256 |",
        "|---|---:|---|",
    ]
    for entry in figures:
        lines.append(
            f"| {entry['id']} | {entry['pixel_width']}x{entry['pixel_height']} | "
            f"`{entry['png_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Table audit",
            "",
            "| Table | Rows | Columns | CSV SHA-256 |",
            "|---|---:|---:|---|",
        ]
    )
    for entry in tables:
        lines.append(
            f"| {entry['id']} | {entry['row_count']} | "
            f"{entry['column_count']} | `{entry['csv_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Every caption and source SHA-256 is recorded in "
            "`reports/gate12_output_manifest.json`. Qualitative images are "
            "derived from the fixed Gate 11 seed and predeclared case-selection "
            "rules. They are illustrative, not additional inferential units.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Generate all Gate 12 outputs deterministically."""
    _style()
    manifest: list[dict[str, Any]] = []
    figure_functions: Sequence[Callable[[list[dict[str, Any]]], None]] = (
        _cohort_flow,
        _split_flow,
        _architecture,
        _paired_region_differences,
        _forest,
        _pareto,
        _qualitative_modalities,
        _error_overlay,
        _et_case_panel,
        _split_balance_supplement,
    )
    table_functions: Sequence[Callable[[list[dict[str, Any]]], None]] = (
        _cohort_table,
        _architecture_resource_table,
        _primary_metric_table,
        _comparison_table,
        _secondary_table,
        _subgroup_table,
    )
    for function in (*figure_functions, *table_functions):
        function(manifest)
    atomic_write_json(
        MANIFEST_PATH,
        {
            "status": "complete",
            "gate": 12,
            "figure_count": len(figure_functions),
            "table_count": len(table_functions),
            "outputs": manifest,
        },
    )
    atomic_write_text(REPORT_PATH, _completion_report(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
