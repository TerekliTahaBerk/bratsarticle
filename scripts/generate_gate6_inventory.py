"""Generate the Gate 6 architecture and loss inventory from versioned configs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bratsarticle.models import find_closest_parameter_match
from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    model_from_config,
    trace_tensor_shapes,
)
from bratsarticle.training.loss_catalog import LOSS_FORMULAS, load_loss_catalog
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text


def _architecture_inventory(config_dir: Path) -> list[dict[str, Any]]:
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No model configs found in {config_dir}")
    configurations = {path.stem: load_model_config(path) for path in config_paths}
    if "unet" not in configurations:
        raise ValueError("The controlled U-Net config is required as parameter target")
    reference_parameters = count_trainable_parameters(
        model_from_config(configurations["unet"])
    )
    rows: list[dict[str, Any]] = []
    for path in config_paths:
        config = configurations[path.stem]
        model = model_from_config(config)
        parameter_count = count_trainable_parameters(model)
        match = find_closest_parameter_match(
            config,
            target_parameters=reference_parameters,
            maximum_base_channels=64,
        )
        rows.append(
            {
                "key": path.stem,
                "name": config.name,
                "config_path": path.as_posix(),
                "config_sha256": file_digest(path),
                "input_channels": config.input_channels,
                "output_channels": config.output_channels,
                "base_channels": config.base_channels,
                "depth": config.depth,
                "batch_normalization": config.batch_normalization,
                "dropout_probability": config.dropout_probability,
                "features": {
                    "residual_blocks": config.residual_blocks,
                    "residual_extended_skips": config.residual_extended_skips,
                    "wide_context": config.wide_context,
                },
                "res_kernel_sizes": list(config.res_kernel_sizes),
                "wc_kernel_size": config.wc_kernel_size,
                "parameter_count": parameter_count,
                "parameter_difference_from_equal_width_unet": (
                    parameter_count - reference_parameters
                ),
                "parameter_ratio_to_equal_width_unet": (
                    parameter_count / reference_parameters
                ),
                "closest_integer_width_match": {
                    "base_channels": match.base_channels,
                    "parameter_count": match.parameter_count,
                    "target_parameters": match.target_parameters,
                    "absolute_difference": match.absolute_difference,
                    "relative_difference": match.relative_difference,
                    "declared_tolerance_fraction": (
                        config.parameter_tolerance_fraction
                    ),
                    "within_declared_tolerance": match.within_tolerance,
                },
                "shape_trace_input": [1, 4, 64, 64],
                "shape_trace": trace_tensor_shapes(
                    model,
                    input_shape=(1, 4, 64, 64),
                ),
            }
        )
    return rows


def _loss_inventory(catalog_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "name": config.name,
            "formula": LOSS_FORMULAS[config.name],
            "alpha": config.alpha,
            "beta": config.beta,
            "gamma": config.gamma,
            "smoothing": config.smoothing,
            "class_weights": config.class_weights,
            "include_background": config.include_background,
            "reduction": config.reduction,
            "expects_logits": config.expects_logits,
        }
        for config in load_loss_catalog(catalog_path)
    ]


def _model_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Gate 6 Model Inventory",
        "",
        "Generated from the versioned model configurations. Parameter matching uses",
        "the controlled 16-channel U-Net as its target and searches integer base",
        "widths from 1 through 64. A failed tolerance is retained as a failure.",
        "",
        "| Model | RB | RES | WC | Parameters | Delta | Match width | "
        "Match parameters | Gap | Within 5% |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["models"]:
        features = row["features"]
        match = row["closest_integer_width_match"]
        lines.append(
            "| {name} | {rb} | {res} | {wc} | {parameters:,} | {difference:+,} | "
            "{width} | {matched:,} | {gap:.2%} | {within} |".format(
                name=row["key"],
                rb="✓" if features["residual_blocks"] else "—",
                res="✓" if features["residual_extended_skips"] else "—",
                wc="✓" if features["wide_context"] else "—",
                parameters=row["parameter_count"],
                difference=row["parameter_difference_from_equal_width_unet"],
                width=match["base_channels"],
                matched=match["parameter_count"],
                gap=match["relative_difference"],
                within="PASS" if match["within_declared_tolerance"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The equal-width matrix isolates feature additions while exposing their",
            "parameter cost. The closest-width results are a sensitivity design, not",
            "substitutes for the primary matrix. BU-Net and U-Net+RES cannot meet the",
            "declared 5% target with a single integer base-width multiplier; this",
            "limitation is therefore explicit.",
            "",
            "Complete per-module tensor traces and configuration hashes are stored in",
            "`reports/gate6_inventory.json`.",
        ]
    )
    return "\n".join(lines)


def _loss_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Gate 6 Loss Methods Table",
        "",
        "All entries consume raw four-channel logits. Targets are the mutually",
        "exclusive BraTS classes `{0, 1, 2, 4}`, mapped internally to contiguous",
        "indices. Class weights are disabled (`null`) in the declared catalog.",
        "",
        "| Loss | Formula | alpha | beta | gamma | Smooth | Background | Reduction |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in payload["losses"]:
        lines.append(
            "| {name} | `{formula}` | {alpha:g} | {beta:g} | {gamma:g} | "
            "{smoothing:g} | {background} | {reduction} |".format(
                name=row["name"],
                formula=row["formula"],
                alpha=row["alpha"],
                beta=row["beta"],
                gamma=row["gamma"],
                smoothing=row["smoothing"],
                background="yes" if row["include_background"] else "no",
                reduction=row["reduction"],
            )
        )
    lines.extend(
        [
            "",
            "`CE` uses softmax cross-entropy. `BCE` uses channel-wise sigmoid BCE",
            "against a one-hot target. Soft Dice and focal Tversky use softmax",
            "probabilities; combined objectives have equal 0.5/0.5 term weights.",
            "These are optimization candidates only. The central evaluator and its",
            "empty-mask rules are unchanged.",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    config_dir: Path,
    loss_catalog_path: Path,
    json_output: Path,
    model_markdown_output: Path,
    loss_markdown_output: Path,
) -> dict[str, Any]:
    """Generate machine-readable and human-readable Gate 6 inventories."""
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "reference_model": "unet",
        "parameter_match_search": {
            "minimum_base_channels": 1,
            "maximum_base_channels": 64,
            "declared_tolerance_fraction": 0.05,
        },
        "models": _architecture_inventory(config_dir),
        "loss_catalog_path": loss_catalog_path.as_posix(),
        "loss_catalog_sha256": file_digest(loss_catalog_path),
        "losses": _loss_inventory(loss_catalog_path),
    }
    atomic_write_json(json_output, payload)
    atomic_write_text(model_markdown_output, _model_markdown(payload))
    atomic_write_text(loss_markdown_output, _loss_markdown(payload))
    return payload


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/models"),
    )
    parser.add_argument(
        "--loss-catalog",
        type=Path,
        default=Path("configs/losses/catalog.yaml"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gate6_inventory.json"),
    )
    parser.add_argument(
        "--model-markdown-output",
        type=Path,
        default=Path("reports/gate6_model_summary.md"),
    )
    parser.add_argument(
        "--loss-markdown-output",
        type=Path,
        default=Path("reports/gate6_loss_methods.md"),
    )
    arguments = parser.parse_args()
    run(
        config_dir=arguments.config_dir,
        loss_catalog_path=arguments.loss_catalog,
        json_output=arguments.json_output,
        model_markdown_output=arguments.model_markdown_output,
        loss_markdown_output=arguments.loss_markdown_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
