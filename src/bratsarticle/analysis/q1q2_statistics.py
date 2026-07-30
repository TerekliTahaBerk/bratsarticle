"""Frozen patient-level analysis for the Q1/Q2 development and external cohorts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json


@dataclass(frozen=True)
class Contrast:
    """One prespecified paired model contrast."""

    contrast_id: str
    first: str
    second: str


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for a paired patient mean."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired bootstrap requires finite one-dimensional values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    estimates = values[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail], method="linear")
    return float(lower), float(upper)


def sign_flip_permutation_p_value(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    """Return a deterministic two-sided Monte Carlo paired sign-flip p value."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Sign-flip testing requires finite paired differences")
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < resamples:
        current = min(10_000, resamples - completed)
        signs = generator.integers(
            0,
            2,
            size=(current, len(values)),
            dtype=np.int8,
        )
        permuted = ((signs * 2 - 1) * values).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed))
        completed += current
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply deterministic Holm family-wise error correction."""
    if not p_values or not all(np.isfinite(value) for value in p_values.values()):
        raise ValueError("Holm adjustment requires a non-empty finite p-value family")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        candidate = min(1.0, (count - index) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _paired_differences(
    model_patient_metrics: pd.DataFrame,
    *,
    contrast: Contrast,
    endpoint: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"model_id", "patient_id", endpoint}
    missing = required.difference(model_patient_metrics.columns)
    if missing:
        raise ValueError(f"Model-patient table is missing columns: {sorted(missing)}")
    first = model_patient_metrics.loc[
        model_patient_metrics["model_id"].eq(contrast.first),
        ["patient_id", endpoint],
    ].rename(columns={endpoint: "first"})
    second = model_patient_metrics.loc[
        model_patient_metrics["model_id"].eq(contrast.second),
        ["patient_id", endpoint],
    ].rename(columns={endpoint: "second"})
    if (
        first["patient_id"].duplicated().any()
        or second["patient_id"].duplicated().any()
    ):
        raise ValueError(f"{contrast.contrast_id} has duplicate model-patient rows")
    paired = first.merge(second, on="patient_id", how="inner", validate="one_to_one")
    if len(paired) != len(first) or len(paired) != len(second):
        raise ValueError(f"{contrast.contrast_id} does not have complete patient pairs")
    values = paired[["first", "second"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{contrast.contrast_id}:{endpoint} contains a non-finite pair"
        )
    differences = values[:, 0] - values[:, 1]
    paired["difference"] = differences
    return paired, differences


def paired_contrast_summary(
    model_patient_metrics: pd.DataFrame,
    *,
    contrast: Contrast,
    endpoint: str,
    bootstrap_resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
    permutation_resamples: int,
    permutation_seed: int,
    smallest_effect_size_of_interest: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate one model contrast using patient-level paired observations."""
    paired, differences = _paired_differences(
        model_patient_metrics,
        contrast=contrast,
        endpoint=endpoint,
    )
    lower, upper = paired_bootstrap_interval(
        differences,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=bootstrap_seed,
    )
    standard_deviation = (
        float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    )
    mean = float(np.mean(differences))
    summary = {
        **asdict(contrast),
        "endpoint": endpoint,
        "paired_patient_count": len(differences),
        "first_mean": float(paired["first"].mean()),
        "second_mean": float(paired["second"].mean()),
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "paired_bootstrap_lower_95": lower,
        "paired_bootstrap_upper_95": upper,
        "standardized_paired_effect_dz": (
            mean / standard_deviation
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else float("nan")
        ),
        "raw_p": sign_flip_permutation_p_value(
            differences,
            resamples=permutation_resamples,
            seed=permutation_seed,
        ),
        "probability_of_superiority": float(
            np.mean(differences > 0) + 0.5 * np.mean(differences == 0)
        ),
        "smallest_effect_size_of_interest": smallest_effect_size_of_interest,
        "mean_reaches_practical_threshold": (mean >= smallest_effect_size_of_interest),
    }
    paired.insert(0, "contrast_id", contrast.contrast_id)
    paired.insert(1, "first_model_id", contrast.first)
    paired.insert(2, "second_model_id", contrast.second)
    return summary, paired


def _difference_cube(
    checkpoint_metrics: pd.DataFrame,
    *,
    contrast: Contrast,
    endpoint: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    required = {
        "model_id",
        "patient_id",
        "training_seed",
        "training_fold",
        endpoint,
    }
    missing = required.difference(checkpoint_metrics.columns)
    if missing:
        raise ValueError(
            f"Checkpoint patient table is missing columns: {sorted(missing)}"
        )
    subset = checkpoint_metrics.loc[
        checkpoint_metrics["model_id"].isin([contrast.first, contrast.second]),
        [
            "model_id",
            "patient_id",
            "training_seed",
            "training_fold",
            endpoint,
        ],
    ].copy()
    duplicates = subset.duplicated(
        ["model_id", "patient_id", "training_seed", "training_fold"]
    )
    if duplicates.any():
        raise ValueError(f"{contrast.contrast_id} contains duplicate replicate rows")
    first = subset.loc[subset["model_id"].eq(contrast.first)].drop(columns="model_id")
    second = subset.loc[subset["model_id"].eq(contrast.second)].drop(columns="model_id")
    paired = first.merge(
        second,
        on=["patient_id", "training_seed", "training_fold"],
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    if len(paired) != len(first) or len(paired) != len(second):
        raise ValueError(f"{contrast.contrast_id} has incomplete replicate pairing")
    paired["difference"] = paired[f"{endpoint}_first"].astype(float) - paired[
        f"{endpoint}_second"
    ].astype(float)
    patients = tuple(sorted(str(value) for value in paired["patient_id"].unique()))
    seeds = tuple(sorted(int(value) for value in paired["training_seed"].unique()))
    folds = tuple(sorted(int(value) for value in paired["training_fold"].unique()))
    expected = len(patients) * len(seeds) * len(folds)
    if len(paired) != expected or not np.isfinite(paired["difference"]).all():
        raise ValueError(
            f"{contrast.contrast_id} is not a complete finite patient-seed-fold cube"
        )
    cube = (
        paired.pivot(
            index="patient_id",
            columns=["training_seed", "training_fold"],
            values="difference",
        )
        .reindex(
            index=list(patients),
            columns=pd.MultiIndex.from_product([seeds, folds]),
        )
        .to_numpy(dtype=np.float64)
        .reshape(len(patients), len(seeds), len(folds))
    )
    if not np.isfinite(cube).all():
        raise ValueError(f"{contrast.contrast_id} difference cube contains gaps")
    return cube, patients, seeds, folds


def hierarchical_bootstrap_intervals(
    checkpoint_metrics: pd.DataFrame,
    *,
    contrast: Contrast,
    endpoint: str,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Resample seeds then patients, with a separate fold-resampling sensitivity."""
    cube, patients, seeds, folds = _difference_cube(
        checkpoint_metrics,
        contrast=contrast,
        endpoint=endpoint,
    )
    generator = np.random.default_rng(seed)
    main_estimates = np.empty(resamples, dtype=np.float64)
    fold_estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_seed_indices = generator.integers(0, len(seeds), size=len(seeds))
        sampled_patient_indices = generator.integers(
            0,
            len(patients),
            size=len(patients),
        )
        selected = cube[sampled_patient_indices][:, sampled_seed_indices, :]
        main_estimates[index] = float(selected.mean())
        sampled_fold_indices = generator.integers(0, len(folds), size=len(folds))
        fold_estimates[index] = float(selected[:, :, sampled_fold_indices].mean())
    tail = (1.0 - confidence_level) / 2.0
    main_lower, main_upper = np.quantile(
        main_estimates,
        [tail, 1.0 - tail],
        method="linear",
    )
    fold_lower, fold_upper = np.quantile(
        fold_estimates,
        [tail, 1.0 - tail],
        method="linear",
    )
    return {
        **asdict(contrast),
        "endpoint": endpoint,
        "patient_count": len(patients),
        "training_seed_count": len(seeds),
        "fold_count": len(folds),
        "resamples": resamples,
        "resampling_order": "training_seed_then_patient",
        "fold_handling_primary": "average_all_folds",
        "hierarchical_mean_difference": float(main_estimates.mean()),
        "hierarchical_bootstrap_lower_95": float(main_lower),
        "hierarchical_bootstrap_upper_95": float(main_upper),
        "fold_resampling_sensitivity_mean_difference": float(fold_estimates.mean()),
        "fold_resampling_sensitivity_lower_95": float(fold_lower),
        "fold_resampling_sensitivity_upper_95": float(fold_upper),
    }


def _mixed_effects_sensitivity(
    checkpoint_metrics: pd.DataFrame,
    *,
    contrast: Contrast,
    endpoint: str,
) -> dict[str, Any]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        raise RuntimeError(
            "statsmodels is required for the frozen mixed-effects sensitivity"
        ) from error
    subset = checkpoint_metrics.loc[
        checkpoint_metrics["model_id"].isin([contrast.first, contrast.second]),
        [
            "model_id",
            "patient_id",
            "training_seed",
            "training_fold",
            endpoint,
        ],
    ].copy()
    if not np.isfinite(subset[endpoint].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{contrast.contrast_id} mixed model has non-finite outcomes")
    subset["model_indicator"] = subset["model_id"].eq(contrast.first).astype(int)
    subset["outcome"] = subset[endpoint].astype(float)
    subset["single_crossed_group"] = 1
    fitted = smf.mixedlm(
        "outcome ~ model_indicator + C(training_fold)",
        subset,
        groups=subset["single_crossed_group"],
        re_formula="0",
        vc_formula={
            "patient": "0 + C(patient_id)",
            "training_seed": "0 + C(training_seed)",
        },
    ).fit(reml=True, method="lbfgs", maxiter=2000, disp=False)
    if not bool(fitted.converged):
        raise RuntimeError(
            f"{contrast.contrast_id} mixed-effects sensitivity did not converge"
        )
    coefficient = float(fitted.params["model_indicator"])
    standard_error = float(fitted.bse["model_indicator"])
    return {
        **asdict(contrast),
        "endpoint": endpoint,
        "status": "converged",
        "observation_count": len(subset),
        "patient_random_intercept": True,
        "training_seed_random_effect": True,
        "fold_fixed_effect": True,
        "coefficient_first_minus_second": coefficient,
        "standard_error": standard_error,
        "wald_lower_95": coefficient - 1.959963984540054 * standard_error,
        "wald_upper_95": coefficient + 1.959963984540054 * standard_error,
        "raw_p": float(fitted.pvalues["model_indicator"]),
        "log_likelihood": float(fitted.llf),
    }


def _finite_metric_summary(
    frame: pd.DataFrame,
    *,
    cohort: str,
    model_id: str,
    endpoint: str,
) -> dict[str, Any]:
    values = frame.loc[frame["model_id"].eq(model_id), endpoint].to_numpy(
        dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    return {
        "cohort": cohort,
        "model_id": model_id,
        "endpoint": endpoint,
        "patient_count": len(values),
        "finite_count": len(finite),
        "nan_count": int(np.isnan(values).sum()),
        "positive_infinity_count": int(np.isposinf(values).sum()),
        "negative_infinity_count": int(np.isneginf(values).sum()),
        "mean_finite": float(finite.mean()) if len(finite) else float("nan"),
        "median_finite": float(np.median(finite)) if len(finite) else float("nan"),
        "q1_finite": (
            float(np.quantile(finite, 0.25, method="linear"))
            if len(finite)
            else float("nan")
        ),
        "q3_finite": (
            float(np.quantile(finite, 0.75, method="linear"))
            if len(finite)
            else float("nan")
        ),
    }


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    suffixes = (
        "_dice",
        "_hd95_mm",
        "_surface_dice",
        "_sensitivity",
        "_precision",
        "_specificity",
        "_relative_volume_error",
        "_lesion_recall",
        "_lesion_precision",
        "_lesion_wise_dice",
        "_lesion_wise_hd95_mm",
        "_false_positive_lesion_count",
    )
    selected = [
        column
        for column in frame.columns
        if column == "mean_regional_dice" or column.endswith(suffixes)
    ]
    return tuple(sorted(selected))


def _development_checkpoint_metrics(
    manifest: dict[str, Any],
    *,
    expected_models: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    runs = [
        cast(dict[str, Any], run)
        for run in cast(list[Any], manifest["runs"])
        if cast(dict[str, Any], run).get("stage") == "main_convergence"
    ]
    if len(runs) != 300:
        raise ValueError("Gate G must contain 300 main-convergence runs")
    for run in runs:
        path = Path(str(run["patient_metrics_path"]))
        if file_digest(path) != str(run["patient_metrics_sha256"]):
            raise ValueError(f"Development metric hash differs: {run['run_id']}")
        frame = pd.read_csv(path)
        frame = frame.loc[frame["evaluation_stage"].eq("raw")].copy()
        frame["training_fold"] = int(run["fold"])
        frame["training_seed"] = int(run["seed"])
        frame["model_id"] = str(run["model_id"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if set(str(value) for value in combined["model_id"].unique()) != expected_models:
        raise ValueError("Development metric model identities differ from the matrix")
    duplicates = combined.duplicated(["model_id", "patient_id", "training_seed"])
    if duplicates.any():
        raise ValueError("Development validation metrics contain duplicate identities")
    counts = combined.groupby(["model_id", "patient_id"]).size()
    if len(counts) != len(expected_models) * 369 or not counts.eq(5).all():
        raise ValueError(
            "Every development model-patient pair must contain five seed replicates"
        )
    if not combined.groupby("patient_id")["training_fold"].nunique().eq(1).all():
        raise ValueError("Development patient fold identity differs across runs")
    return combined


def _development_model_patient(checkpoints: pd.DataFrame) -> pd.DataFrame:
    numeric = list(_metric_columns(checkpoints))
    grouped = checkpoints.groupby(["model_id", "patient_id"], sort=True)
    output = grouped[numeric].mean().reset_index()
    output["valid_seed_count"] = grouped["training_seed"].nunique().to_numpy()
    return output


def _model_metadata(path: Path) -> dict[str, dict[str, Any]]:
    matrix = _load_yaml(path)
    entries = [
        cast(dict[str, Any], entry) for entry in cast(list[Any], matrix["main_models"])
    ]
    return {str(entry["id"]): entry for entry in entries}


def _best_development_2d_model(
    *,
    development_model_patient: pd.DataFrame,
    gate_g_manifest: dict[str, Any],
    model_metadata: dict[str, dict[str, Any]],
    endpoint: str,
) -> dict[str, Any]:
    eligible = {
        model_id
        for model_id, metadata in model_metadata.items()
        if str(metadata["dimensionality"]) == "2D"
    }
    performance = (
        development_model_patient.loc[
            development_model_patient["model_id"].isin(eligible)
        ]
        .groupby("model_id")[endpoint]
        .mean()
    )
    main_runs = [
        cast(dict[str, Any], run)
        for run in cast(list[Any], gate_g_manifest["runs"])
        if cast(dict[str, Any], run).get("stage") == "main_convergence"
        and str(cast(dict[str, Any], run)["model_id"]) in eligible
    ]
    cost = pd.DataFrame(main_runs).groupby("model_id")["accelerator_hours"].mean()
    ranking = pd.DataFrame(
        {
            "model_id": performance.index,
            "development_mean_regional_dice": performance.to_numpy(),
            "mean_accelerator_hours": cost.reindex(performance.index).to_numpy(),
        }
    ).sort_values(
        [
            "development_mean_regional_dice",
            "mean_accelerator_hours",
            "model_id",
        ],
        ascending=[False, True, True],
    )
    if (
        ranking.empty
        or not np.isfinite(
            ranking[
                ["development_mean_regional_dice", "mean_accelerator_hours"]
            ].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise ValueError("Development-only 2D model selection is incomplete")
    selected = ranking.iloc[0]
    return {
        "selection_rule": (
            "highest_development_cv_mean_regional_dice_then_lower_mean_"
            "accelerator_hours_then_lexicographic_model_id"
        ),
        "selected_model_id": str(selected["model_id"]),
        "ranking": _records(ranking),
    }


def _prespecified_contrasts(
    plan: dict[str, Any],
    *,
    best_development_2d_model: str,
) -> tuple[Contrast, ...]:
    primary = cast(dict[str, Any], plan["primary_contrast"])
    contrasts = [
        Contrast(
            contrast_id=str(primary["id"]),
            first=str(primary["first"]),
            second=str(primary["second"]),
        )
    ]
    for raw in cast(list[Any], plan["confirmatory_secondary_contrasts"]):
        entry = cast(dict[str, Any], raw)
        if "second" in entry:
            second = str(entry["second"])
        elif (
            entry.get("second_selection")
            == "highest_development_cv_mean_regional_dice_among_frozen_2d_models"
        ):
            second = best_development_2d_model
        else:
            raise ValueError(f"Unknown frozen second-model rule: {entry}")
        contrasts.append(
            Contrast(
                contrast_id=str(entry["id"]),
                first=str(entry["first"]),
                second=second,
            )
        )
    if len(contrasts) != 5 or len({item.contrast_id for item in contrasts}) != 5:
        raise ValueError("The confirmatory family must contain five unique contrasts")
    return tuple(contrasts)


def _validate_prerequisites(
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate_g_freeze_path = Path(str(execution["gate_g_analysis_freeze"]))
    gate_g_manifest_path = Path(str(execution["gate_g_checkpoint_manifest"]))
    gate_h_completion_path = Path(str(execution["gate_h_completion"]))
    gate_g_freeze = _load_json(gate_g_freeze_path)
    gate_g_manifest = _load_json(gate_g_manifest_path)
    gate_h = _load_json(gate_h_completion_path)
    if (
        gate_g_freeze.get("status") != "frozen_external_inference_permitted"
        or gate_g_manifest.get("status") != "frozen"
        or gate_g_freeze.get("checkpoint_manifest_sha256")
        != file_digest(gate_g_manifest_path)
        or gate_h.get("status") != "pass"
        or gate_h.get("gate_h_pass") is not True
    ):
        raise PermissionError("Frozen Gate G and passing Gate H are required")
    for key, path_key, hash_key in (
        (
            "checkpoint",
            "checkpoint_patient_metrics",
            "checkpoint_patient_metrics_sha256",
        ),
        ("model", "model_patient_metrics", "model_patient_metrics_sha256"),
    ):
        path = Path(str(gate_h[path_key]))
        if not path.is_file() or file_digest(path) != str(gate_h[hash_key]):
            raise ValueError(f"Gate H {key} patient metric hash differs")
    return gate_g_freeze, gate_g_manifest, gate_h


def analyze_q1q2_statistics(
    execution_path: Path = Path("configs/q1q2_v2/statistical_execution.yaml"),
) -> dict[str, Any]:
    """Run the frozen analysis only after Gate G and Gate H pass."""
    execution = _load_yaml(execution_path)
    if execution.get("status") != "frozen_before_external_results":
        raise PermissionError("Statistical execution contract is not frozen")
    gate_g_freeze, gate_g_manifest, gate_h = _validate_prerequisites(execution)
    plan_path = Path(str(execution["statistical_analysis_plan"]))
    plan = _load_yaml(plan_path)
    if (
        plan.get("status") != "prespecified_pending_training_and_checkpoint_freeze"
        or plan.get("external_inference_permitted") is not False
    ):
        raise PermissionError("The statistical analysis plan is not prespecified")
    metadata = _model_metadata(Path(str(execution["model_matrix"])))
    expected_models = set(metadata)
    development_checkpoints = _development_checkpoint_metrics(
        gate_g_manifest,
        expected_models=expected_models,
    )
    development_model_patient = _development_model_patient(development_checkpoints)
    selection = _best_development_2d_model(
        development_model_patient=development_model_patient,
        gate_g_manifest=gate_g_manifest,
        model_metadata=metadata,
        endpoint=str(cast(dict[str, Any], plan["primary_endpoint"])["name"]),
    )
    external_checkpoints = pd.read_csv(Path(str(gate_h["checkpoint_patient_metrics"])))
    external_models = pd.read_csv(Path(str(gate_h["model_patient_metrics"])))
    role = str(execution["confirmatory_cohort_role"])
    confirmatory_checkpoints = external_checkpoints.loc[
        external_checkpoints["cohort_role"].eq(role)
    ].copy()
    confirmatory_models = external_models.loc[
        external_models["cohort_role"].eq(role)
    ].copy()
    expected = cast(dict[str, Any], execution["expected"])
    if len(expected_models) != int(expected["models"]):
        raise ValueError("Model matrix count differs from the execution contract")
    confirmatory_checkpoint_duplicates = confirmatory_checkpoints.duplicated(
        ["model_id", "patient_id", "training_seed", "training_fold"]
    )
    confirmatory_model_duplicates = confirmatory_models.duplicated(
        ["model_id", "patient_id"]
    )
    confirmatory_checkpoint_counts = confirmatory_checkpoints.groupby(
        ["model_id", "patient_id"]
    ).size()
    confirmatory_model_counts = confirmatory_models.groupby("model_id").size()
    if (
        len(confirmatory_checkpoints)
        != int(expected["models"])
        * int(expected["confirmatory_patients"])
        * int(expected["replicates_per_external_model_patient"])
        or len(confirmatory_models)
        != int(expected["models"]) * int(expected["confirmatory_patients"])
        or set(str(value) for value in confirmatory_models["model_id"].unique())
        != expected_models
        or confirmatory_checkpoint_duplicates.any()
        or confirmatory_model_duplicates.any()
        or not confirmatory_checkpoint_counts.eq(
            int(expected["replicates_per_external_model_patient"])
        ).all()
        or not confirmatory_model_counts.eq(
            int(expected["confirmatory_patients"])
        ).all()
        or not confirmatory_models["valid_checkpoint_count"]
        .eq(int(expected["replicates_per_external_model_patient"]))
        .all()
    ):
        raise ValueError("External confirmatory analysis matrix is incomplete")
    primary_endpoint = str(cast(dict[str, Any], plan["primary_endpoint"])["name"])
    bootstrap = cast(
        dict[str, Any],
        cast(dict[str, Any], plan["estimation"])["paired_patient_bootstrap"],
    )
    hierarchical = cast(
        dict[str, Any],
        cast(dict[str, Any], plan["estimation"])["hierarchical_bootstrap"],
    )
    permutation = cast(
        dict[str, Any], cast(dict[str, Any], plan["estimation"])["paired_permutation"]
    )
    practical = cast(dict[str, Any], plan["practical_interpretation"])
    contrasts = _prespecified_contrasts(
        plan,
        best_development_2d_model=str(selection["selected_model_id"]),
    )
    contrast_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    hierarchical_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    for index, contrast in enumerate(contrasts):
        summary, paired = paired_contrast_summary(
            confirmatory_models,
            contrast=contrast,
            endpoint=primary_endpoint,
            bootstrap_resamples=int(bootstrap["resamples"]),
            confidence_level=float(bootstrap["confidence_level"]),
            bootstrap_seed=int(bootstrap["seed"]) + index,
            permutation_resamples=int(permutation["resamples"]),
            permutation_seed=int(permutation["seed"]) + index,
            smallest_effect_size_of_interest=float(
                practical["smallest_effect_size_of_interest"]
            ),
        )
        contrast_rows.append(summary)
        paired_frames.append(paired)
        hierarchical_rows.append(
            hierarchical_bootstrap_intervals(
                confirmatory_checkpoints,
                contrast=contrast,
                endpoint=primary_endpoint,
                resamples=int(hierarchical["resamples"]),
                confidence_level=float(bootstrap["confidence_level"]),
                seed=int(hierarchical["seed"]) + index,
            )
        )
        mixed_rows.append(
            _mixed_effects_sensitivity(
                confirmatory_checkpoints,
                contrast=contrast,
                endpoint=primary_endpoint,
            )
        )
    adjusted = holm_adjust(
        {row["contrast_id"]: float(row["raw_p"]) for row in contrast_rows}
    )
    alpha = float(cast(dict[str, Any], plan["multiplicity"])["alpha_two_sided"])
    for row in contrast_rows:
        row["holm_adjusted_p"] = adjusted[str(row["contrast_id"])]
        row["holm_reject_at_alpha"] = float(row["holm_adjusted_p"]) <= alpha
        row["claim_interpretation"] = (
            "positive_and_practically_relevant"
            if row["holm_reject_at_alpha"]
            and float(row["mean_difference"])
            >= float(row["smallest_effect_size_of_interest"])
            else (
                "positive_but_below_practical_threshold"
                if row["holm_reject_at_alpha"] and float(row["mean_difference"]) > 0
                else "no_confirmatory_superiority"
            )
        )
    metric_rows: list[dict[str, Any]] = []
    supportive_role = str(execution["supportive_cohort_role"])
    supportive_models = external_models.loc[
        external_models["cohort_role"].eq(supportive_role)
    ].copy()
    supportive_counts = supportive_models.groupby("model_id").size()
    if (
        len(supportive_models)
        != int(expected["models"]) * int(expected["supportive_patients"])
        or supportive_models.duplicated(["model_id", "patient_id"]).any()
        or not supportive_counts.eq(int(expected["supportive_patients"])).all()
    ):
        raise ValueError("Supportive external analysis matrix is incomplete")
    for cohort, frame in (
        ("development_cross_validation", development_model_patient),
        ("external_confirmatory", confirmatory_models),
        ("external_supportive_other_neoplasm", supportive_models),
    ):
        for model_id in sorted(expected_models):
            for endpoint in _metric_columns(frame):
                metric_rows.append(
                    _finite_metric_summary(
                        frame,
                        cohort=cohort,
                        model_id=model_id,
                        endpoint=endpoint,
                    )
                )
    outputs = cast(dict[str, Any], execution["outputs"])
    output_paths = {key: Path(str(value)) for key, value in outputs.items()}
    atomic_write_csv(
        output_paths["development_checkpoint_metrics"],
        _records(development_checkpoints),
    )
    atomic_write_csv(
        output_paths["development_model_patient_metrics"],
        _records(development_model_patient),
    )
    atomic_write_csv(
        output_paths["external_confirmatory_checkpoint_metrics"],
        _records(confirmatory_checkpoints),
    )
    atomic_write_csv(
        output_paths["external_confirmatory_model_patient_metrics"],
        _records(confirmatory_models),
    )
    atomic_write_csv(
        output_paths["external_supportive_model_patient_metrics"],
        _records(supportive_models),
    )
    atomic_write_csv(output_paths["model_metric_summary"], metric_rows)
    atomic_write_csv(output_paths["primary_contrasts"], contrast_rows)
    atomic_write_csv(
        output_paths["paired_patient_differences"],
        _records(pd.concat(paired_frames, ignore_index=True)),
    )
    atomic_write_csv(output_paths["hierarchical_bootstrap"], hierarchical_rows)
    atomic_write_csv(output_paths["mixed_effects"], mixed_rows)
    atomic_write_json(output_paths["development_2d_selection"], selection)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "gate": "post_H_statistical_analysis",
        "legacy_internal_test_accessed": False,
        "external_results_used_for_model_selection": False,
        "confirmatory_patient_count": int(expected["confirmatory_patients"]),
        "model_count": int(expected["models"]),
        "contrast_count": len(contrasts),
        "best_2d_model_selected_on_development_only": selection["selected_model_id"],
        "analysis_plan_sha256": file_digest(plan_path),
        "analysis_execution_sha256": file_digest(execution_path),
        "gate_g_analysis_freeze_sha256": file_digest(
            Path(str(execution["gate_g_analysis_freeze"]))
        ),
        "gate_h_completion_sha256": file_digest(
            Path(str(execution["gate_h_completion"]))
        ),
        "gate_g_frozen_input_hashes": gate_g_freeze["analysis_input_sha256"],
        "outputs": {
            key: {
                "path": path.as_posix(),
                "sha256": file_digest(path),
            }
            for key, path in output_paths.items()
            if key != "completion"
        },
    }
    atomic_write_json(output_paths["completion"], completion)
    return completion


__all__ = [
    "Contrast",
    "analyze_q1q2_statistics",
    "hierarchical_bootstrap_intervals",
    "holm_adjust",
    "paired_bootstrap_interval",
    "paired_contrast_summary",
    "sign_flip_permutation_p_value",
]
