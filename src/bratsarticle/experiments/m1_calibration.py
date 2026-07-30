"""Nonreportable M1 Max throughput calibration for the frozen v2 model matrix."""

from __future__ import annotations

import gc
import platform
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from monai.networks.nets.swin_unetr import SwinUNETR
from torch import nn

from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    model_from_config,
)
from bratsarticle.training.loss_catalog import (
    ConfiguredSegmentationLoss,
    build_loss,
    load_loss_catalog,
)
from bratsarticle.training.reproducibility import seed_everything
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text


@dataclass(frozen=True)
class CalibrationCase:
    """One synthetic optimization workload matching a frozen model input."""

    model_id: str
    adapter: str
    config_path: str
    input_shape: tuple[int, ...]
    microbatch_size: int
    gradient_accumulation_steps: int


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], payload)


def _loss(catalog_path: Path, name: str) -> ConfiguredSegmentationLoss:
    matches = [
        config for config in load_loss_catalog(catalog_path) if config.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one loss named {name}")
    return build_loss(matches[0])


def _cases(
    calibration: dict[str, Any],
    matrix: dict[str, Any],
) -> list[CalibrationCase]:
    benchmark = cast(dict[str, Any], calibration["benchmark"])
    native_2d = cast(dict[str, Any], benchmark["native_2d"])
    native_2p5d = cast(dict[str, Any], benchmark["native_2p5d"])
    swin = cast(dict[str, Any], benchmark["swin_unetr"])
    cases: list[CalibrationCase] = []
    for raw_model in cast(list[dict[str, Any]], matrix["main_models"]):
        adapter = str(raw_model["adapter"])
        model_id = str(raw_model["id"])
        if adapter == "official_nnunetv2":
            continue
        if adapter == "monai_swinunetr":
            patch = tuple(int(value) for value in swin["patch_size"])
            cases.append(
                CalibrationCase(
                    model_id=model_id,
                    adapter=adapter,
                    config_path=str(raw_model["config"]),
                    input_shape=(
                        int(swin["microbatch_size"]),
                        4,
                        *patch,
                    ),
                    microbatch_size=int(swin["microbatch_size"]),
                    gradient_accumulation_steps=int(
                        swin["gradient_accumulation_steps"]
                    ),
                )
            )
            continue
        workload = native_2p5d if model_id == "unet_2p5d_k5" else native_2d
        model_config = load_model_config(Path(str(raw_model["config"])))
        cases.append(
            CalibrationCase(
                model_id=model_id,
                adapter=adapter,
                config_path=str(raw_model["config"]),
                input_shape=(
                    int(workload["microbatch_size"]),
                    model_config.input_channels,
                    int(workload["input_height"]),
                    int(workload["input_width"]),
                ),
                microbatch_size=int(workload["microbatch_size"]),
                gradient_accumulation_steps=int(
                    workload["gradient_accumulation_steps"]
                ),
            )
        )
    return cases


def _model(case: CalibrationCase) -> nn.Module:
    if case.adapter == "native_configurable_unet":
        return model_from_config(load_model_config(Path(case.config_path)))
    if case.adapter == "monai_swinunetr":
        raw = _load_yaml(Path(case.config_path))["model"]
        return SwinUNETR(
            in_channels=int(raw["input_channels"]),
            out_channels=int(raw["output_channels"]),
            feature_size=int(raw["feature_size"]),
            use_checkpoint=bool(raw["use_checkpoint"]),
            spatial_dims=int(raw["spatial_dims"]),
        )
    raise ValueError(f"Unsupported calibration adapter: {case.adapter}")


def _random_brats_labels(
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    labels = torch.randint(0, 4, shape, device=device)
    return torch.where(labels == 3, torch.full_like(labels, 4), labels)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_sample(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "mps":
        return None, None
    return (
        int(torch.mps.current_allocated_memory()),
        int(torch.mps.driver_allocated_memory()),
    )


def _optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: ConfiguredSegmentationLoss,
    image: torch.Tensor,
    label: torch.Tensor,
    accumulation_steps: int,
    device: torch.device,
) -> tuple[float, int | None, int | None]:
    optimizer.zero_grad(set_to_none=True)
    last_loss = float("nan")
    allocated_samples: list[int] = []
    driver_samples: list[int] = []
    for _ in range(accumulation_steps):
        logits = cast(torch.Tensor, model(image))
        loss = loss_function(logits, label) / accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError("Calibration produced a non-finite loss")
        loss.backward()
        _synchronize(device)
        allocated, driver = _memory_sample(device)
        if allocated is not None:
            allocated_samples.append(allocated)
        if driver is not None:
            driver_samples.append(driver)
        last_loss = float(loss.detach().cpu()) * accumulation_steps
    optimizer.step()
    _synchronize(device)
    return (
        last_loss,
        max(allocated_samples, default=None),
        max(driver_samples, default=None),
    )


def _benchmark_case(
    case: CalibrationCase,
    *,
    loss_function: ConfiguredSegmentationLoss,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    if case.gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation must be positive")
    model = _model(case).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )
    image = torch.randn(case.input_shape, device=device)
    label = _random_brats_labels(
        (case.input_shape[0], *case.input_shape[2:]),
        device=device,
    )
    durations: list[float] = []
    losses: list[float] = []
    allocated_samples: list[int] = []
    driver_samples: list[int] = []
    for step in range(warmup_steps + measured_steps):
        started = time.perf_counter()
        loss, allocated, driver = _optimizer_step(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            image=image,
            label=label,
            accumulation_steps=case.gradient_accumulation_steps,
            device=device,
        )
        elapsed = time.perf_counter() - started
        if step >= warmup_steps:
            durations.append(elapsed)
            losses.append(loss)
            if allocated is not None:
                allocated_samples.append(allocated)
            if driver is not None:
                driver_samples.append(driver)
    return {
        "status": "pass",
        "model_id": case.model_id,
        "adapter": case.adapter,
        "config": case.config_path,
        "config_sha256": file_digest(Path(case.config_path)),
        "input_shape": list(case.input_shape),
        "microbatch_size": case.microbatch_size,
        "gradient_accumulation_steps": case.gradient_accumulation_steps,
        "effective_batch_size": (
            case.microbatch_size * case.gradient_accumulation_steps
        ),
        "parameter_count": count_trainable_parameters(model),
        "measured_optimizer_steps": measured_steps,
        "optimizer_step_seconds": durations,
        "median_optimizer_step_seconds": statistics.median(durations),
        "minimum_optimizer_step_seconds": min(durations),
        "maximum_optimizer_step_seconds": max(durations),
        "terminal_loss": losses[-1],
        "maximum_sampled_mps_allocated_unified_memory_bytes": max(
            allocated_samples,
            default=None,
        ),
        "maximum_sampled_mps_driver_allocated_unified_memory_bytes": max(
            driver_samples,
            default=None,
        ),
        "memory_caution": (
            "Samples are synchronized post-backward observations, not an "
            "allocator-reported exact peak."
        ),
    }


def estimate_serial_budget(
    rows: list[dict[str, Any]],
    *,
    runs_per_model: int = 25,
    maximum_optimizer_steps: int = 50_000,
) -> dict[str, Any]:
    """Estimate a transparent upper-bound proxy from measured optimizer steps."""
    successful = [row for row in rows if row.get("status") == "pass"]
    model_hours = {
        str(row["model_id"]): (
            float(row["median_optimizer_step_seconds"])
            * maximum_optimizer_steps
            / 3600.0
        )
        for row in successful
    }
    total_hours = sum(model_hours.values()) * runs_per_model
    return {
        "assumption": (
            "constant measured synthetic optimizer-step time through the "
            "50,000-step ceiling; excludes validation, checkpoint I/O, "
            "preprocessing, nnU-Net, compute-matched repeats, and reproduction"
        ),
        "maximum_optimizer_steps": maximum_optimizer_steps,
        "runs_per_model": runs_per_model,
        "model_upper_proxy_hours_per_run": model_hours,
        "measured_model_count": len(model_hours),
        "serial_upper_proxy_hours_for_measured_models": total_hours,
        "serial_upper_proxy_days_for_measured_models": total_hours / 24.0,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Apple M1 Max calibration",
        "",
        "Status: **NONREPORTABLE HARDWARE CALIBRATION**",
        "",
        (
            "Synthetic tensors were used. No raw, legacy internal-test, or "
            "external data were opened, and these losses are not scientific results."
        ),
        "",
        "| Model | Input | Effective batch | Median optimizer step | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in cast(list[dict[str, Any]], payload["models"]):
        seconds = row.get("median_optimizer_step_seconds")
        formatted = "n/a" if seconds is None else f"{float(seconds):.3f} s"
        lines.append(
            f"| {row['model_id']} | {row.get('input_shape', 'n/a')} | "
            f"{row.get('effective_batch_size', 'n/a')} | {formatted} | "
            f"{row['status']} |"
        )
    budget = cast(dict[str, Any], payload["serial_budget_proxy"])
    lines.extend(
        [
            "",
            (
                "Measured-model 50,000-step upper proxy: "
                f"{float(budget['serial_upper_proxy_hours_for_measured_models']):,.1f} "
                "serial hours "
                f"({float(budget['serial_upper_proxy_days_for_measured_models']):,.1f} "
                "days), before validation and the explicitly excluded work."
            ),
            "",
            (
                "This report decides feasibility only. It cannot justify reducing "
                "the five folds, the common five seeds, or the convergence rule."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_calibration(
    *,
    calibration_path: Path,
    matrix_path: Path,
    loss_catalog_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Run every eligible synthetic workload and serialize feasibility evidence."""
    calibration = _load_yaml(calibration_path)
    matrix = _load_yaml(matrix_path)
    benchmark = cast(dict[str, Any], calibration["benchmark"])
    if str(calibration["status"]) != "frozen_nonreportable_hardware_calibration":
        raise PermissionError("Calibration config is not frozen")
    guards = cast(dict[str, Any], calibration["guards"])
    if any(bool(value) for value in guards.values()):
        raise PermissionError("Every calibration data/result guard must be false")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("The reportable M1 calibration requires available MPS")
    device = torch.device("mps")
    determinism = cast(dict[str, Any], benchmark["deterministic_algorithms"])
    seed_everything(
        int(benchmark["seed"]),
        deterministic=bool(determinism["enabled"]),
        deterministic_warn_only=bool(
            determinism["warn_only_for_missing_mps_kernel"]
        ),
    )
    loss_function = _loss(loss_catalog_path, str(benchmark["loss"])).to(device)
    rows: list[dict[str, Any]] = []
    for case in _cases(calibration, matrix):
        torch.mps.empty_cache()
        gc.collect()
        try:
            rows.append(
                _benchmark_case(
                    case,
                    loss_function=loss_function,
                    device=device,
                    warmup_steps=int(benchmark["warmup_optimizer_steps"]),
                    measured_steps=int(benchmark["measured_optimizer_steps"]),
                )
            )
        except Exception:
            rows.append(
                {
                    "status": "fail",
                    "model_id": case.model_id,
                    "adapter": case.adapter,
                    "config": case.config_path,
                    "input_shape": list(case.input_shape),
                    "microbatch_size": case.microbatch_size,
                    "gradient_accumulation_steps": (
                        case.gradient_accumulation_steps
                    ),
                    "effective_batch_size": (
                        case.microbatch_size * case.gradient_accumulation_steps
                    ),
                    "error": traceback.format_exc(),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "pass" if all(row["status"] == "pass" for row in rows) else "fail"
        ),
        "scientific_use": "prohibited",
        "hardware": {
            "chip": "Apple M1 Max",
            "gpu_cores": 32,
            "unified_memory_gb": 32,
            "backend": "mps",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": calibration_path.as_posix(),
        "config_sha256": file_digest(calibration_path),
        "model_matrix": matrix_path.as_posix(),
        "model_matrix_sha256": file_digest(matrix_path),
        "raw_data_accessed": False,
        "legacy_internal_test_accessed": False,
        "external_data_accessed": False,
        "deterministic_algorithms": {
            "enabled": bool(determinism["enabled"]),
            "warn_only_for_missing_mps_kernel": bool(
                determinism["warn_only_for_missing_mps_kernel"]
            ),
            "rationale": str(determinism["rationale"]),
        },
        "models": rows,
        "unbenchmarked_models": ["nnunetv2_2d", "nnunetv2_3d_fullres"],
        "serial_budget_proxy": estimate_serial_budget(rows),
    }
    atomic_write_json(output_json, payload)
    atomic_write_text(output_markdown, _markdown(payload))
    return payload


__all__ = ["CalibrationCase", "estimate_serial_budget", "run_calibration"]
