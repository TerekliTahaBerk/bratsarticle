"""Artifact-derived Q1/Q2 figures with no manually entered result values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _model_names(path: Path) -> dict[str, str]:
    matrix = _load_yaml(path)
    return {
        str(entry["id"]): str(entry["display_name"])
        for entry in cast(list[dict[str, Any]], matrix["main_models"])
    }


def _save_figure(
    figure: Figure,
    *,
    stem: Path,
    dpi: int,
) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
    }
    figure.savefig(outputs["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        path.as_posix(): file_digest(path)
        for path in outputs.values()
    }


def build_study_design_figure(
    *,
    config_path: Path = Path("configs/q1q2_v2/figure_execution.yaml"),
) -> dict[str, str]:
    """Render the outcome-independent study design from frozen counts."""
    config = _load_yaml(config_path)
    design = cast(dict[str, Any], config["design"])
    output = cast(dict[str, Any], config["outputs"])
    render = cast(dict[str, Any], config["render"])
    figure, axis = plt.subplots(figsize=(12, 4.5))
    axis.axis("off")
    boxes = [
        (
            0.025,
            "Development cohort",
            f"{int(design['development_patients'])} patients\n"
            f"{int(design['folds'])} patient-level folds x "
            f"{int(design['seeds'])} seeds",
        ),
        (
            0.275,
            "Frozen model matrix",
            f"{int(design['models'])} main models\n"
            f"{int(design['development_runs'])} total development runs",
        ),
        (
            0.525,
            "Gate G",
            "Checkpoint and analysis freeze\nResource and claim freeze",
        ),
        (
            0.775,
            "Single Gate H session",
            f"{int(design['external_main_checkpoints'])} checkpoints\n"
            f"{int(design['external_confirmatory_patients'])} confirmatory; "
            f"{int(design['external_supportive_patients'])} supportive",
        ),
    ]
    for index, (x, title, body) in enumerate(boxes):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.30),
                0.20,
                0.47,
                transform=axis.transAxes,
                boxstyle="round,pad=0.012",
                facecolor="#F4F7FB",
                edgecolor="#244A73",
                linewidth=1.5,
            )
        )
        axis.text(
            x + 0.012,
            0.68,
            title,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        axis.text(
            x + 0.012,
            0.47,
            body,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
        )
        if index < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(x + 0.245, 0.535),
                xytext=(x + 0.205, 0.535),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#244A73", "lw": 1.5},
            )
    axis.text(
        0.5,
        0.08,
        (
            "External outcomes cannot alter models, checkpoints, thresholds, "
            "or postprocessing."
        ),
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color="#7A1F1F",
    )
    return _save_figure(
        figure,
        stem=Path(str(output["directory"])) / str(output["study_design"]),
        dpi=int(render["dpi"]),
    )


def _confirmatory_performance(
    summary: pd.DataFrame,
    *,
    names: dict[str, str],
) -> Figure:
    data = summary.loc[
        summary["cohort"].eq("external_confirmatory")
        & summary["endpoint"].eq("mean_regional_dice")
    ].copy()
    if len(data) != len(names) or not np.isfinite(
        data[["mean_finite", "q1_finite", "q3_finite"]]
    ).all().all():
        raise RuntimeError("Confirmatory performance summary is incomplete")
    data = data.sort_values("mean_finite")
    y = np.arange(len(data))
    means = data["mean_finite"].to_numpy(float)
    lower = means - data["q1_finite"].to_numpy(float)
    upper = data["q3_finite"].to_numpy(float) - means
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#1D5D8F",
        ecolor="#8AA9C2",
        capsize=3,
    )
    axis.set_yticks(y, [names[str(value)] for value in data["model_id"]])
    axis.set_xlabel("Patient-level mean regional Dice (mean; IQR)")
    axis.set_title("External confirmatory performance")
    axis.grid(axis="x", alpha=0.25)
    return figure


def _contrast_forest(contrasts: pd.DataFrame) -> Figure:
    required = {
        "contrast_id",
        "mean_difference",
        "paired_bootstrap_lower_95",
        "paired_bootstrap_upper_95",
    }
    if required.difference(contrasts.columns) or not len(contrasts):
        raise RuntimeError("Primary contrast artifact is incomplete")
    data = contrasts.sort_values("mean_difference")
    y = np.arange(len(data))
    means = data["mean_difference"].to_numpy(float)
    lower = means - data["paired_bootstrap_lower_95"].to_numpy(float)
    upper = data["paired_bootstrap_upper_95"].to_numpy(float) - means
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.axvline(0.0, color="#555555", lw=1)
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#8C2D2D",
        ecolor="#C78D8D",
        capsize=3,
    )
    axis.set_yticks(y, data["contrast_id"])
    axis.set_xlabel("Paired mean difference in regional Dice (95% bootstrap CI)")
    axis.set_title("Prespecified confirmatory contrasts")
    axis.grid(axis="x", alpha=0.25)
    return figure


def _accuracy_cost(pareto: pd.DataFrame, names: dict[str, str]) -> Figure:
    required = {
        "model_id",
        "external_confirmatory_patient_mean_regional_dice",
        "inference_end_to_end_p50_seconds",
        "parameter_count",
        "pareto_accuracy_vs_inference_end_to_end_p50_seconds",
    }
    if required.difference(pareto.columns) or len(pareto) != len(names):
        raise RuntimeError("Accuracy-cost Pareto artifact is incomplete")
    figure, axis = plt.subplots(figsize=(8, 6))
    front = pareto[
        "pareto_accuracy_vs_inference_end_to_end_p50_seconds"
    ].astype(bool)
    sizes = 35 + 115 * (
        pareto["parameter_count"] / pareto["parameter_count"].max()
    )
    axis.scatter(
        pareto["inference_end_to_end_p50_seconds"],
        pareto["external_confirmatory_patient_mean_regional_dice"],
        s=sizes,
        c=np.where(front, "#187A5B", "#A8B2BC"),
        alpha=0.85,
        edgecolor="white",
    )
    for _, row in pareto.iterrows():
        axis.annotate(
            names[str(row["model_id"])],
            (
                float(cast(float, row["inference_end_to_end_p50_seconds"])),
                float(
                    cast(
                        float,
                        row[
                            "external_confirmatory_patient_mean_regional_dice"
                        ],
                    )
                ),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Median end-to-end latency (s/volume)")
    axis.set_ylabel("External confirmatory mean regional Dice")
    axis.set_title("Measured accuracy-latency Pareto view")
    axis.grid(alpha=0.25)
    return figure


def _resource_profiles(pareto: pd.DataFrame, names: dict[str, str]) -> Figure:
    data = pareto.sort_values("training_accelerator_hours_mean")
    labels = [names[str(value)] for value in data["model_id"]]
    y = np.arange(len(data))
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    axes[0].barh(y, data["training_accelerator_hours_mean"], color="#456A8B")
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Mean accelerator-hours per run")
    axes[1].barh(
        y,
        data["inference_end_to_end_p50_seconds"],
        color="#6D8F5D",
    )
    axes[1].set_xlabel("Median end-to-end seconds per volume")
    figure.suptitle("Measured training and inference resources")
    figure.tight_layout()
    return figure


def _subgroup_forest(subgroups: pd.DataFrame) -> Figure:
    required = {
        "contrast_id",
        "dimension",
        "level",
        "patient_count",
        "reporting_role",
        "mean_difference",
        "bootstrap_lower_95",
        "bootstrap_upper_95",
    }
    if required.difference(subgroups.columns) or not len(subgroups):
        raise RuntimeError("Exploratory subgroup contrast artifact is incomplete")
    data = subgroups.loc[
        subgroups["reporting_role"].eq("exploratory_estimation")
    ].copy()
    if not len(data):
        data = subgroups.copy()
    data = data.sort_values(["contrast_id", "dimension", "level"]).head(40)
    labels = [
        (
            f"{row['dimension']}: {row['level']} "
            f"(n={int(cast(int, row['patient_count']))})"
        )
        for _, row in data.iterrows()
    ]
    y = np.arange(len(data))
    mean = data["mean_difference"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(9, max(6, 0.26 * len(data))))
    axis.axvline(0.0, color="#555555", lw=1)
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack(
            [
                mean - data["bootstrap_lower_95"].to_numpy(float),
                data["bootstrap_upper_95"].to_numpy(float) - mean,
            ]
        ),
        fmt="o",
        color="#6B3F8C",
        ecolor="#B7A0C8",
        capsize=2,
    )
    axis.set_yticks(y, labels, fontsize=7)
    axis.set_xlabel("Exploratory paired difference (95% bootstrap CI)")
    axis.set_title("Prespecified exploratory subgroup estimates")
    axis.grid(axis="x", alpha=0.25)
    return figure


def build_q1q2_result_figures(
    *,
    config_path: Path = Path("configs/q1q2_v2/figure_execution.yaml"),
) -> dict[str, Any]:
    """Render all numerical figures only from completed, hash-bound analyses."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_external_results":
        raise PermissionError("Figure execution protocol is not frozen")
    statistics = _load_json(Path(str(config["statistical_completion"])))
    resources = _load_json(Path(str(config["resource_completion"])))
    subgroups = _load_json(Path(str(config["subgroup_completion"])))
    if (
        statistics.get("status") != "complete"
        or resources.get("status") != "complete"
        or subgroups.get("status") != "complete"
    ):
        raise PermissionError(
            "Completed statistics, resources, and subgroups are required"
        )
    statistical_outputs = cast(dict[str, dict[str, str]], statistics["outputs"])
    resource_outputs = cast(dict[str, str], resources["artifacts"])
    subgroup_outputs = cast(dict[str, dict[str, str]], subgroups["outputs"])

    def verified(path: Path, expected_sha256: str) -> Path:
        if not path.is_file() or file_digest(path) != expected_sha256:
            raise RuntimeError(f"Figure source hash differs: {path}")
        return path

    metric_path = verified(
        Path(statistical_outputs["model_metric_summary"]["path"]),
        statistical_outputs["model_metric_summary"]["sha256"],
    )
    contrast_path = verified(
        Path(statistical_outputs["primary_contrasts"]["path"]),
        statistical_outputs["primary_contrasts"]["sha256"],
    )
    resource_execution = _load_yaml(Path(str(config["resource_execution"])))
    pareto_path = Path(
        str(cast(dict[str, Any], resource_execution["outputs"])["accuracy_cost_pareto"])
    )
    verified(pareto_path, resource_outputs[pareto_path.as_posix()])
    subgroup_path = verified(
        Path(subgroup_outputs["contrast_subgroup_summary"]["path"]),
        subgroup_outputs["contrast_subgroup_summary"]["sha256"],
    )
    names = _model_names(Path(str(config["model_matrix"])))
    output = cast(dict[str, Any], config["outputs"])
    render = cast(dict[str, Any], config["render"])
    directory = Path(str(output["directory"]))
    figures = {
        "study_design": build_study_design_figure(config_path=config_path),
        "confirmatory_performance": _save_figure(
            _confirmatory_performance(pd.read_csv(metric_path), names=names),
            stem=directory / str(output["confirmatory_performance"]),
            dpi=int(render["dpi"]),
        ),
        "primary_contrasts": _save_figure(
            _contrast_forest(pd.read_csv(contrast_path)),
            stem=directory / str(output["primary_contrasts"]),
            dpi=int(render["dpi"]),
        ),
        "accuracy_cost": _save_figure(
            _accuracy_cost(pd.read_csv(pareto_path), names),
            stem=directory / str(output["accuracy_cost"]),
            dpi=int(render["dpi"]),
        ),
        "resource_profiles": _save_figure(
            _resource_profiles(pd.read_csv(pareto_path), names),
            stem=directory / str(output["resource_profiles"]),
            dpi=int(render["dpi"]),
        ),
        "exploratory_subgroups": _save_figure(
            _subgroup_forest(pd.read_csv(subgroup_path)),
            stem=directory / str(output["exploratory_subgroups"]),
            dpi=int(render["dpi"]),
        ),
    }
    completion = {
        "schema_version": 1,
        "status": "complete",
        "manual_result_values_used": False,
        "figures": figures,
    }
    atomic_write_json(Path(str(output["completion"])), completion)
    return completion


__all__ = ["build_q1q2_result_figures", "build_study_design_figure"]
