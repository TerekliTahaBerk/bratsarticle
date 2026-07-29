"""Validation contracts for the v2 equal-seed model matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


class ProtocolMatrixError(RuntimeError):
    """Raised when the v2 model matrix violates a frozen design invariant."""


@dataclass(frozen=True)
class MatrixValidation:
    """Validated model-matrix identity and run counts."""

    model_ids: tuple[str, ...]
    main_seeds: tuple[int, ...]
    fold_count: int
    convergence_run_count: int
    core_compute_matched_run_count: int
    config_sha256: str
    seeds_sha256: str


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolMatrixError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], payload)


def validate_model_matrix(
    matrix_path: Path,
    seeds_path: Path,
) -> MatrixValidation:
    """Require all mandatory models to use the same five training seeds."""
    matrix = _load_yaml(matrix_path)
    seed_config = _load_yaml(seeds_path)
    seeds = tuple(int(seed) for seed in seed_config["main_training"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ProtocolMatrixError("At least five unique main seeds are required")
    models = cast(list[dict[str, Any]], matrix["main_models"])
    expected_ids = {
        "unet_small",
        "unet_parameter_matched_res",
        "unet_compute_matched_res",
        "unet_res",
        "unet_wc",
        "bunet",
        "resblock_unet",
        "resblock_unet_wc",
        "nnunetv2_2d",
        "nnunetv2_3d_fullres",
        "unet_2p5d_k5",
        "swin_unetr",
    }
    expected_primary_sources = {
        "unet_res": "10.3390/electronics9122203",
        "unet_wc": "10.3390/electronics9122203",
        "bunet": "10.3390/electronics9122203",
        "resblock_unet_wc": "10.3390/electronics9122203",
        "nnunetv2_2d": "10.1038/s41592-020-01008-z",
        "nnunetv2_3d_fullres": "10.1038/s41592-020-01008-z",
        "unet_2p5d_k5": "10.3390/bioengineering10020181",
        "swin_unetr": "10.1007/978-3-031-08999-2_22",
    }
    model_ids = tuple(str(model["id"]) for model in models)
    if len(model_ids) != len(set(model_ids)):
        raise ProtocolMatrixError("Model IDs are not unique")
    if set(model_ids) != expected_ids:
        raise ProtocolMatrixError(
            f"Mandatory model mismatch: {sorted(expected_ids - set(model_ids))}"
        )
    for model in models:
        model_seeds = tuple(int(seed) for seed in model["seeds"])
        if model_seeds != seeds:
            raise ProtocolMatrixError(
                f"{model['id']} does not use the common ordered seed list"
            )
        license_name = str(model.get("implementation_license", ""))
        if license_name != "Apache-2.0":
            raise ProtocolMatrixError(
                f"{model['id']} has an unresolved implementation license"
            )
        config = str(model["config"])
        if config.startswith("configs/") and not Path(config).is_file():
            raise ProtocolMatrixError(f"Missing model configuration: {config}")
        expected_source = expected_primary_sources.get(str(model["id"]))
        if (
            expected_source is not None
            and str(model.get("primary_source")) != expected_source
        ):
            raise ProtocolMatrixError(
                f"{model['id']} has the wrong primary-source attribution"
            )
    fold_count = 5
    convergence_runs = len(models) * len(seeds) * fold_count
    core_models = [model for model in models if model["family"] == "component_core"]
    core_compute_runs = len(core_models) * len(seeds) * fold_count
    return MatrixValidation(
        model_ids=model_ids,
        main_seeds=seeds,
        fold_count=fold_count,
        convergence_run_count=convergence_runs,
        core_compute_matched_run_count=core_compute_runs,
        config_sha256=file_digest(matrix_path),
        seeds_sha256=file_digest(seeds_path),
    )


def write_matrix_validation(
    matrix_path: Path,
    seeds_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Serialize the validated pretraining matrix contract."""
    validated = validate_model_matrix(matrix_path, seeds_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "validated_before_main_training",
        "model_ids": list(validated.model_ids),
        "model_count": len(validated.model_ids),
        "main_seeds": list(validated.main_seeds),
        "seed_count": len(validated.main_seeds),
        "fold_count": validated.fold_count,
        "convergence_matched_run_count": validated.convergence_run_count,
        "component_core_compute_matched_run_count": (
            validated.core_compute_matched_run_count
        ),
        "matrix_sha256": validated.config_sha256,
        "seeds_sha256": validated.seeds_sha256,
        "external_inference_permitted": False,
    }
    atomic_write_json(output_path, payload)
    return payload
