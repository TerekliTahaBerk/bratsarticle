"""Machine-readable Methods metadata for the v2 loss comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bratsarticle.training.loss_catalog import (
    LOSS_FORMULAS,
    LossConfig,
    LossName,
    load_loss_catalog,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

MANDATORY_Q1Q2_LOSSES: tuple[LossName, ...] = (
    "cross_entropy_plus_soft_dice",
    "binary_cross_entropy_plus_focal_tversky",
    "cross_entropy_plus_focal_tversky",
)


def _row(config: LossConfig) -> dict[str, Any]:
    bce_background = (
        config.include_background
        if config.bce_include_background is None
        else config.bce_include_background
    )
    overlap_background = (
        config.include_background
        if config.overlap_include_background is None
        else config.overlap_include_background
    )
    return {
        "name": config.name,
        "formula": LOSS_FORMULAS[config.name],
        "logit_expectation": "raw logits",
        "ce_probability_transform": (
            "softmax"
            if "cross_entropy" in config.name
            and "binary_cross_entropy" not in config.name
            else "not_applicable"
        ),
        "bce_probability_transform": (
            "independent sigmoid"
            if "binary_cross_entropy" in config.name
            else "not_applicable"
        ),
        "overlap_probability_transform": (
            "softmax"
            if "dice" in config.name or "tversky" in config.name
            else "not_applicable"
        ),
        "ce_background_included": "cross_entropy" in config.name
        and "binary_cross_entropy" not in config.name,
        "bce_background_included": (
            bce_background if "binary_cross_entropy" in config.name else None
        ),
        "overlap_background_included": (
            overlap_background
            if "dice" in config.name or "tversky" in config.name
            else None
        ),
        "reduction_axes": (
            "batch and every spatial axis per class "
            "(batch,height,width in 2D; batch,depth,height,width in 3D)"
        ),
        "batch_aggregation": "joint numerator and denominator across batch",
        "class_aggregation": config.reduction,
        "alpha": config.alpha,
        "beta": config.beta,
        "gamma": config.gamma,
        "smoothing": config.smoothing,
        "class_weights": (
            "none"
            if config.class_weights is None
            else ",".join(str(value) for value in config.class_weights)
        ),
        "empty_class_behavior": (
            "smoothing yields 1 for jointly empty prediction/reference overlap; "
            "softmax usually makes predicted mass nonzero"
        ),
        "inference": "four-class softmax argmax; class index 3 maps to BraTS label 4",
    }


def write_loss_methods(
    catalog_path: Path,
    json_output: Path,
    csv_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    """Write exact loss equations and term-level transform/background behavior."""
    catalog = {config.name: config for config in load_loss_catalog(catalog_path)}
    missing = set(MANDATORY_Q1Q2_LOSSES) - set(catalog)
    if missing:
        raise ValueError(f"Mandatory q1q2 losses are missing: {sorted(missing)}")
    rows = [_row(catalog[name]) for name in MANDATORY_Q1Q2_LOSSES]
    atomic_write_csv(csv_output, rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "catalog_path": catalog_path.as_posix(),
        "catalog_sha256": file_digest(catalog_path),
        "mandatory_loss_count": len(rows),
        "losses": rows,
        "methods_table_must_match_catalog_hash": True,
    }
    atomic_write_json(json_output, payload)
    lines = [
        "# Loss definitions and code-Methods parity",
        "",
        f"Catalog SHA-256: `{payload['catalog_sha256']}`",
        "",
        (
            "All losses consume the same four raw logits. Multiclass CE and "
            "overlap terms use softmax. BCE uses independent sigmoid values for "
            "its term only; this is a loss construction and does not change the "
            "mutually exclusive four-class inference rule."
        ),
        "",
        (
            "| Loss | CE transform | BCE transform | Overlap transform | "
            "BCE bg | Overlap bg |"
        ),
        "|---|---|---|---|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['ce_probability_transform']} | "
            f"{row['bce_probability_transform']} | "
            f"{row['overlap_probability_transform']} | "
            f"{row['bce_background_included']} | "
            f"{row['overlap_background_included']} |"
        )
    lines.extend(
        [
            "",
            (
                "Overlap sums use the batch and every spatial axis, producing "
                "one value per selected class before the declared mean reduction. "
                "CE uses PyTorch mean reduction over batch and spatial elements. "
                "BCE uses elementwise logits loss, optional foreground channel "
                "selection, then the declared global mean."
            ),
            "",
            (
                "The architecture-attribution loss remains pending until the "
                "three mandatory candidates complete development-only five-fold "
                "selection. Neither the legacy 74-patient subset nor external "
                "labels may influence this selection."
            ),
        ]
    )
    atomic_write_text(markdown_output, "\n".join(lines) + "\n")
    return payload
