"""Real-data MPS repeat-tolerance audit required before reportable Swin runs."""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from bratsarticle.data.dataset import build_cv_fold_dataset
from bratsarticle.data.preprocessing import load_preprocessing_config
from bratsarticle.experiments.q1q2_swin_runner import (
    SWIN_MODEL_ID,
    _load_selected_loss,
    _loss,
    _model,
    _sample_patch,
    load_swin_runner_config,
)
from bratsarticle.training.reproducibility import seed_everything
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return commit, bool(status.strip())


def _one_step(
    *,
    model_config: Path,
    catalog: Path,
    loss_name: str,
    image: torch.Tensor,
    label: torch.Tensor,
    seed: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    seed_everything(seed, deterministic=True, deterministic_warn_only=True)
    device = torch.device("mps")
    model, _ = _model(model_config)
    model.to(device).train()
    loss_function = _loss(catalog, loss_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = cast(torch.Tensor, model(image.to(device)))
    loss = loss_function(logits, label.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("Swin tolerance audit produced non-finite loss")
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    loss_value = float(loss.detach().cpu())
    logits_cpu = logits.detach().cpu()
    parameters = torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in model.parameters()]
    )
    del optimizer, loss_function, logits, loss, model
    torch.mps.empty_cache()
    gc.collect()
    return loss_value, logits_cpu, parameters


def _difference(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float | bool]:
    difference = torch.abs(first - second)
    reference = torch.maximum(torch.abs(first), torch.abs(second))
    allowed = absolute_tolerance + relative_tolerance * reference
    return {
        "maximum_absolute_difference": float(torch.max(difference)),
        "mean_absolute_difference": float(torch.mean(difference)),
        "maximum_allowed_difference": float(torch.max(allowed)),
        "within_elementwise_tolerance": bool(torch.all(difference <= allowed)),
    }


def run_swin_repeat_tolerance(
    *,
    runner_config_path: Path,
    selected_loss_path: Path,
    dataset_root: Path,
    output_path: Path,
    allow_training_diagnostics: bool,
) -> dict[str, Any]:
    """Repeat an identical real-data optimization step and compare tensors."""
    if not allow_training_diagnostics:
        raise PermissionError(
            "Swin repeat-tolerance audit requires explicit diagnostic authorization"
        )
    lock = Path("artifacts/q1q2_v2/queue_runtime/loss_screen.lock")
    if lock.exists():
        raise RuntimeError("Loss-screen queue is still active")
    config = load_swin_runner_config(runner_config_path)
    git_commit, dirty = _git_state()
    if dirty:
        raise RuntimeError("Swin repeat-tolerance audit requires a clean repository")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Swin repeat-tolerance audit requires available MPS")

    data = cast(dict[str, Any], config["data"])
    model_raw = cast(dict[str, Any], config["model"])
    loss_raw = cast(dict[str, Any], config["loss"])
    training = cast(dict[str, Any], config["training"])
    tolerance = cast(dict[str, Any], config["repeat_tolerance"])
    comparison = cast(dict[str, Any], tolerance["comparison"])
    loss_name, evidence_path = _load_selected_loss(selected_loss_path)
    fold = int(tolerance["fold"])
    seed = int(tolerance["seed"])
    fold_path = Path(str(data["fold_pattern"]).format(fold=fold))
    canonical_path = Path(str(data["canonical_manifest"]))
    preprocessing_path = Path(str(data["preprocessing"]))
    model_config = Path(str(model_raw["config"]))
    catalog = Path(str(loss_raw["catalog"]))
    preprocessing = load_preprocessing_config(preprocessing_path)
    volumes = build_cv_fold_dataset(
        fold_path,
        canonical_path,
        "train",
        dataset_root,
        preprocessing,
        seed=seed,
    )
    volume = volumes.subject_volume(0)
    patch_probe, patch_size = _model(model_config)
    del patch_probe
    gc.collect()
    image_array, label_array = _sample_patch(
        volume.image,
        volume.label,
        patch_size=patch_size,
        generator=np.random.default_rng(seed),
        tumor_probability=float(data["tumor_patch_probability"]),
        preprocessing=preprocessing,
    )
    image = torch.from_numpy(image_array[None])
    label = torch.from_numpy(label_array[None])
    first = _one_step(
        model_config=model_config,
        catalog=catalog,
        loss_name=loss_name,
        image=image,
        label=label,
        seed=seed,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    second = _one_step(
        model_config=model_config,
        catalog=catalog,
        loss_name=loss_name,
        image=image,
        label=label,
        seed=seed,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    absolute = float(comparison["absolute_tolerance"])
    relative = float(comparison["relative_tolerance"])
    logits_difference = _difference(
        first[1],
        second[1],
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )
    parameter_difference = _difference(
        first[2],
        second[2],
        absolute_tolerance=absolute,
        relative_tolerance=relative,
    )
    loss_difference = abs(first[0] - second[0])
    acceptance = {
        "loss": loss_difference <= float(comparison["loss_absolute_tolerance"]),
        "logits": bool(logits_difference["within_elementwise_tolerance"]),
        "parameters_after_step": bool(
            parameter_difference["within_elementwise_tolerance"]
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if all(acceptance.values()) else "fail",
        "model_id": SWIN_MODEL_ID,
        "scientific_role": "pre_main_mps_repeat_tolerance_diagnostic",
        "git_commit": git_commit,
        "runner_config": runner_config_path.as_posix(),
        "runner_config_sha256": file_digest(runner_config_path),
        "selected_loss_config_sha256": file_digest(selected_loss_path),
        "loss_selection_artifact_sha256": file_digest(evidence_path),
        "fold": fold,
        "fold_sha256": file_digest(fold_path),
        "seed": seed,
        "patient_id": str(volumes.manifest.iloc[0]["subject_id"]),
        "patch_size": list(patch_size),
        "loss_name": loss_name,
        "loss_values": [first[0], second[0]],
        "loss_absolute_difference": loss_difference,
        "logits_difference": logits_difference,
        "parameters_after_step_difference": parameter_difference,
        "tolerances": comparison,
        "acceptance": acceptance,
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
    }
    atomic_write_json(output_path, report)
    return report


__all__ = ["run_swin_repeat_tolerance"]
