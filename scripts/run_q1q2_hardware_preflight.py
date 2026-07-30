#!/usr/bin/env python3
"""Measure the v2 host and estimate the mandatory experiment budget."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, cast

import torch
from monai.networks.nets.swin_unetr import SwinUNETR

from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text


def _legacy_mps_seconds_per_step() -> float:
    rates: list[float] = []
    for path in sorted(Path("artifacts/runs").glob("*/resource_profile.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("device") != "mps":
            continue
        steps = int(payload.get("completed_optimizer_steps", 0))
        elapsed = float(payload.get("elapsed_seconds", 0.0))
        if steps > 0 and elapsed > 0:
            rates.append(elapsed / steps)
    if not rates:
        raise RuntimeError("No legacy MPS step-time artifacts were found")
    return float(statistics.median(rates))


def _swin_smoke() -> dict[str, Any]:
    torch.manual_seed(20260730)
    device = torch.device("mps")
    torch.mps.empty_cache()
    model = SwinUNETR(
        in_channels=4,
        out_channels=4,
        feature_size=24,
        use_checkpoint=False,
        spatial_dims=3,
    ).to(device)
    image = torch.randn((1, 4, 64, 64, 64), device=device)
    target = torch.randint(0, 4, (1, 64, 64, 64), device=device)
    started = time.perf_counter()
    logits = cast(torch.Tensor, model(image))
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()  # type: ignore[no-untyped-call]
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "status": "pass",
        "input_shape": [1, 4, 64, 64, 64],
        "output_shape": list(logits.shape),
        "forward_backward_seconds": elapsed,
        "loss": float(loss.detach().cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "mps_framework_reported_allocated_unified_memory_bytes": (
            torch.mps.current_allocated_memory()
        ),
        "mps_driver_allocated_unified_memory_bytes": (
            torch.mps.driver_allocated_memory()
        ),
        "warning": (
            "64-cubed is an operator smoke, not the frozen 96-cubed training "
            "patch or a throughput benchmark"
        ),
    }


def run_preflight(output_json: Path, budget_json: Path, budget_report: Path) -> None:
    """Run a real MPS smoke and serialize feasibility evidence."""
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS must be available for the reportable preflight")
    test = torch.ones((1024, 1024), device="mps")
    result = test @ test
    torch.mps.synchronize()
    disk = shutil.disk_usage(Path.cwd())
    preflight: dict[str, Any] = {
        "schema_version": 1,
        "hardware": {
            "model": "MacBook Pro MacBookPro18,2",
            "chip": "Apple M1 Max",
            "cpu_cores": 10,
            "gpu_cores": 32,
            "unified_memory_gb": 32,
            "memory_terminology": (
                "MPS framework-reported allocated unified memory"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "monai": importlib.metadata.version("monai"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
        },
        "accelerators": {
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_matrix_smoke_value": float(result[0, 0].cpu()),
        },
        "disk": {
            "free_bytes": disk.free,
            "free_gib": disk.free / (1024**3),
        },
        "swin_unetr_mps_smoke": _swin_smoke(),
        "external_results_accessed": False,
        "internal_legacy_test_accessed": False,
    }
    atomic_write_json(output_json, preflight)

    seconds_per_step = _legacy_mps_seconds_per_step()
    native_max_hours = seconds_per_step * 50_000 / 3600
    native_convergence_runs = 315
    core_compute_runs = 200
    swin_main_runs = 25
    swin_seconds = float(
        preflight["swin_unetr_mps_smoke"]["forward_backward_seconds"]
    )
    swin_smoke_extrapolated_hours_per_run = swin_seconds * 50_000 / 3600
    known_upper_hours = (
        native_convergence_runs * native_max_hours
        + core_compute_runs * 4.0
        + swin_main_runs * swin_smoke_extrapolated_hours_per_run
    )
    budget: dict[str, Any] = {
        "schema_version": 1,
        "mandatory_runs": {
            "convergence_matched_all_models": 300,
            "compute_matched_component_core": 200,
            "development_loss_selection": 15,
            "architecture_loss_interaction_additional": 100,
            "total": 615,
        },
        "known_native_2d_convergence_like_runs": native_convergence_runs,
        "legacy_median_seconds_per_optimizer_step": seconds_per_step,
        "native_2d_maximum_hours_per_50000_step_run": native_max_hours,
        "core_compute_matched_hours": 800.0,
        "swin_unetr": {
            "main_runs": swin_main_runs,
            "64_cubed_smoke_seconds_per_step": swin_seconds,
            "extrapolated_hours_per_50000_steps_at_smoke_shape": (
                swin_smoke_extrapolated_hours_per_run
            ),
            "caution": (
                "The frozen 96-cubed patch will not be faster; this extrapolation "
                "is a feasibility lower-bound proxy, not a training-time estimate."
            ),
        },
        "not_yet_benchmarked_runs": {
            "nnunetv2_2d": 25,
            "nnunetv2_3d_fullres": 25,
        },
        "known_upper_bound_proxy_accelerator_hours_excluding_nnunet": (
            known_upper_hours
        ),
        "known_upper_bound_proxy_serial_days_excluding_nnunet": (
            known_upper_hours / 24
        ),
        "available_parallel_accelerator_count": 1,
        "full_matrix_feasible_on_current_host_in_bounded_execution": False,
        "required_resolution": (
            "A declared CUDA scheduler/cluster allocation with enough parallel "
            "GPU-hours and storage, or an explicitly revised protocol. The "
            "mandatory five-fold/equal-five-seed design cannot be silently reduced."
        ),
    }
    atomic_write_json(budget_json, budget)

    lines = [
        "# Compute and storage feasibility",
        "",
        "Decision: **FULL MATRIX BLOCKED ON THE CURRENT SINGLE-MPS HOST**",
        "",
        (
            f"The mandatory plan contains {budget['mandatory_runs']['total']} "
            "reportable training runs before reproduction reruns. The 300-model "
            "convergence matrix and 200 core compute-matched runs retain all five "
            "folds and the common five-seed list."
        ),
        "",
        (
            f"Legacy artifact timing gives a median {seconds_per_step:.3f} seconds "
            f"per 2D optimizer step, or {native_max_hours:.2f} hours at the "
            "50,000-step ceiling. The known native-2D and fixed compute budgets "
            "already require thousands of serial accelerator-hours."
        ),
        "",
        (
            f"Swin UNETR completed a real 64-cubed MPS forward/backward smoke in "
            f"{swin_seconds:.2f} seconds. Even extrapolating that smaller-than-"
            "frozen patch yields "
            f"{swin_smoke_extrapolated_hours_per_run:.1f} hours per 50,000 steps. "
            "This is a lower-bound feasibility proxy, not a final duration claim."
        ),
        "",
        (
            f"The combined known upper-bound proxy is {known_upper_hours:,.0f} "
            f"accelerator-hours ({known_upper_hours / 24:,.0f} serial days) and "
            "still excludes 25 nnU-Net 2D and 25 nnU-Net 3D runs."
        ),
        "",
        (
            "MPS itself is available and the 3D transformer operators execute, "
            "so this is not a device-detection failure. It is a scheduling and "
            "total-compute blocker. Full training must not start without a "
            "credible cluster allocation or a protocol revision that is documented "
            "before results."
        ),
    ]
    atomic_write_text(budget_report, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-mps-smoke", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_mps_smoke:
        raise PermissionError("Preflight requires --allow-mps-smoke")
    run_preflight(
        Path("reports/q1q2_v2/hardware_preflight.json"),
        Path("reports/q1q2_v2/compute_budget.json"),
        Path("reports/q1q2_v2/compute_budget.md"),
    )


if __name__ == "__main__":
    main()
