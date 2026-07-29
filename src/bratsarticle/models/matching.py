"""Deterministic parameter- and compute-matching for configurable 2D U-Nets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bratsarticle.models.configurable_unet import ConfigurableUNet2D, ModelConfig
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)


@dataclass(frozen=True)
class ModelProfile:
    """Static model properties at one declared tensor shape."""

    name: str
    base_channels: int
    depth: int
    parameter_count: int
    macs_per_slice: int
    flops_per_slice: int
    receptive_field_proxy_pixels: int
    largest_single_activation_bytes: int
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]


@dataclass(frozen=True)
class MatchChoice:
    """Closest architecture to a declared static-resource target."""

    criterion: str
    profile: ModelProfile
    target_value: int
    absolute_difference: int
    relative_difference: float
    within_tolerance: bool


def _receptive_field_proxy(config: ModelConfig) -> int:
    """Return a transparent longest-path axial receptive-field proxy."""
    receptive_field = 1
    jump = 1
    for _ in range(config.depth):
        receptive_field += 4 * jump
        receptive_field += jump
        jump *= 2
    receptive_field += 4 * jump
    if config.wide_context:
        receptive_field += 2 * (config.wc_kernel_size - 1) * jump
    for _ in range(config.depth):
        jump //= 2
        receptive_field += jump
        receptive_field += 4 * jump
    if config.residual_extended_skips:
        widest_res_kernel = max(config.res_kernel_sizes)
        receptive_field += 2 * (widest_res_kernel - 1)
    return int(receptive_field)


def profile_model(
    config: ModelConfig,
    *,
    input_shape: tuple[int, int, int, int] = (1, 4, 240, 240),
) -> ModelProfile:
    """Profile parameters and convolutional MACs without allocating real images."""
    if input_shape[1] != config.input_channels:
        raise ValueError("Input shape channels do not match the model configuration")
    macs = 0
    largest_activation_bytes = 0

    with torch.device("meta"):
        model = ConfigurableUNet2D(config).eval()

    def convolution_hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal macs, largest_activation_bytes
        largest_activation_bytes = max(
            largest_activation_bytes,
            output.numel() * output.element_size(),
        )
        if isinstance(module, nn.Conv2d):
            batch, output_channels, height, width = output.shape
            kernel_height, kernel_width = module.kernel_size
            operations = (
                batch
                * output_channels
                * height
                * width
                * (module.in_channels // module.groups)
                * kernel_height
                * kernel_width
            )
            macs += int(operations)
        elif isinstance(module, nn.ConvTranspose2d):
            batch, input_channels, height, width = inputs[0].shape
            kernel_height, kernel_width = module.kernel_size
            operations = (
                batch
                * input_channels
                * height
                * width
                * (module.out_channels // module.groups)
                * kernel_height
                * kernel_width
            )
            macs += int(operations)

    handles = [
        module.register_forward_hook(convolution_hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    with torch.no_grad():
        output = model(torch.zeros(input_shape, device="meta"))
    for handle in handles:
        handle.remove()
    output_shape = tuple(int(value) for value in output.shape)
    expected_output = (
        input_shape[0],
        config.output_channels,
        input_shape[2],
        input_shape[3],
    )
    if output_shape != expected_output:
        raise RuntimeError(f"Unexpected output shape: {output_shape}")
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelProfile(
        name=config.name,
        base_channels=config.base_channels,
        depth=config.depth,
        parameter_count=parameter_count,
        macs_per_slice=macs,
        flops_per_slice=2 * macs,
        receptive_field_proxy_pixels=_receptive_field_proxy(config),
        largest_single_activation_bytes=largest_activation_bytes,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def search_plain_unet_controls(
    target_config: ModelConfig,
    *,
    widths: range = range(4, 65),
    depths: range = range(2, 7),
    tolerance_fraction: float = 0.02,
) -> tuple[list[ModelProfile], MatchChoice, MatchChoice]:
    """Search width/depth controls closest to RES parameters and MACs."""
    if tolerance_fraction < 0:
        raise ValueError("tolerance_fraction cannot be negative")
    target_profile = profile_model(target_config)
    candidates: list[ModelProfile] = []
    for depth in depths:
        for width in widths:
            plain = replace(
                target_config,
                name=f"plain_unet_w{width}_d{depth}",
                base_channels=width,
                depth=depth,
                residual_blocks=False,
                residual_extended_skips=False,
                wide_context=False,
                parameter_target=None,
                parameter_tolerance_fraction=tolerance_fraction,
            )
            candidates.append(profile_model(plain))

    same_depth_parameter_candidates = [
        profile
        for profile in candidates
        if profile.depth == target_profile.depth
        and abs(profile.parameter_count - target_profile.parameter_count)
        / target_profile.parameter_count
        <= tolerance_fraction
    ]
    parameter_pool = same_depth_parameter_candidates or candidates
    parameter_profile = min(
        parameter_pool,
        key=lambda profile: (
            abs(profile.parameter_count - target_profile.parameter_count),
            abs(profile.macs_per_slice - target_profile.macs_per_slice),
            profile.depth,
            profile.base_channels,
        ),
    )
    compute_profile = min(
        candidates,
        key=lambda profile: (
            abs(profile.macs_per_slice - target_profile.macs_per_slice),
            abs(profile.parameter_count - target_profile.parameter_count),
            profile.depth,
            profile.base_channels,
        ),
    )

    parameter_difference = (
        parameter_profile.parameter_count - target_profile.parameter_count
    )
    compute_difference = compute_profile.macs_per_slice - target_profile.macs_per_slice
    parameter_choice = MatchChoice(
        criterion="parameter_count",
        profile=parameter_profile,
        target_value=target_profile.parameter_count,
        absolute_difference=parameter_difference,
        relative_difference=abs(parameter_difference) / target_profile.parameter_count,
        within_tolerance=(
            abs(parameter_difference) / target_profile.parameter_count
            <= tolerance_fraction
        ),
    )
    compute_choice = MatchChoice(
        criterion="macs_per_slice",
        profile=compute_profile,
        target_value=target_profile.macs_per_slice,
        absolute_difference=compute_difference,
        relative_difference=abs(compute_difference)
        / target_profile.macs_per_slice,
        within_tolerance=(
            abs(compute_difference) / target_profile.macs_per_slice
            <= tolerance_fraction
        ),
    )
    return candidates, parameter_choice, compute_choice


def write_matching_report(
    *,
    target_config: ModelConfig,
    search_output: Path,
    summary_output: Path,
    report_output: Path,
    tolerance_fraction: float = 0.02,
) -> dict[str, Any]:
    """Run the search and serialize all candidate and selection evidence."""
    target = profile_model(target_config)
    candidates, parameter_match, compute_match = search_plain_unet_controls(
        target_config,
        tolerance_fraction=tolerance_fraction,
    )
    rows: list[dict[str, Any]] = []
    for profile in candidates:
        row = asdict(profile)
        row["input_shape"] = "x".join(str(value) for value in profile.input_shape)
        row["output_shape"] = "x".join(str(value) for value in profile.output_shape)
        row["parameter_relative_difference_to_res"] = (
            abs(profile.parameter_count - target.parameter_count)
            / target.parameter_count
        )
        row["mac_relative_difference_to_res"] = (
            abs(profile.macs_per_slice - target.macs_per_slice)
            / target.macs_per_slice
        )
        rows.append(row)
    atomic_write_csv(search_output, rows)

    def choice_payload(choice: MatchChoice) -> dict[str, Any]:
        return {
            "criterion": choice.criterion,
            "profile": asdict(choice.profile),
            "target_value": choice.target_value,
            "absolute_difference": choice.absolute_difference,
            "relative_difference": choice.relative_difference,
            "within_tolerance": choice.within_tolerance,
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "selected_before_main_training",
        "search_space": {
            "base_channels": [4, 64],
            "depth": [2, 6],
            "candidate_count": len(candidates),
            "plain_unet_only": True,
            "input_shape": list(target.input_shape),
            "tolerance_fraction": tolerance_fraction,
        },
        "parameter_selection_policy": (
            "preserve target depth when at least one same-depth candidate is "
            "within tolerance; otherwise select the globally closest width/depth"
        ),
        "target_unet_plus_res": asdict(target),
        "parameter_match": choice_payload(parameter_match),
        "compute_match": choice_payload(compute_match),
        "mac_definition": (
            "convolution and transposed-convolution multiply-accumulate pairs; "
            "FLOP=2*MAC"
        ),
        "memory_field_definition": (
            "largest single float32 activation tensor from a meta-device shape "
            "trace; not measured peak training memory"
        ),
        "receptive_field_field_definition": (
            "transparent longest-path axial theoretical proxy in pixels"
        ),
    }
    atomic_write_json(summary_output, summary)

    report_lines = [
        "# Capacity- and compute-matching search",
        "",
        "Status: **selected before main training**",
        "",
        (
            "A deterministic exhaustive search evaluated plain U-Nets with base "
            "widths 4-64 and depths 2-6 at input 1x4x240x240. No RES, WC, or "
            "residual block was allowed in either control."
        ),
        (
            "For parameter matching, target depth was preserved whenever a "
            "same-depth candidate met the 2% tolerance, reducing topology as a "
            "source of confounding. A global width/depth fallback was permitted "
            "only if no such candidate existed."
        ),
        "",
        "| Model | Width | Depth | Parameters | MAC/slice | Difference | Within 2% |",
        "|---|---:|---:|---:|---:|---:|:---:|",
        (
            f"| U-Net+RES target | {target.base_channels} | {target.depth} | "
            f"{target.parameter_count:,} | {target.macs_per_slice:,} | — | — |"
        ),
        (
            f"| Parameter-matched plain U-Net | "
            f"{parameter_match.profile.base_channels} | "
            f"{parameter_match.profile.depth} | "
            f"{parameter_match.profile.parameter_count:,} | "
            f"{parameter_match.profile.macs_per_slice:,} | "
            f"{parameter_match.relative_difference:.4%} parameters | "
            f"{'yes' if parameter_match.within_tolerance else 'no'} |"
        ),
        (
            f"| Compute-matched plain U-Net | {compute_match.profile.base_channels} | "
            f"{compute_match.profile.depth} | "
            f"{compute_match.profile.parameter_count:,} | "
            f"{compute_match.profile.macs_per_slice:,} | "
            f"{compute_match.relative_difference:.4%} MAC | "
            f"{'yes' if compute_match.within_tolerance else 'no'} |"
        ),
        "",
        (
            "The parameter-matched and compute-matched controls are distinct "
            "estimands. Their realized wall-clock budgets and measured peak "
            "unified memory remain training-time Gate D/F evidence."
        ),
    ]
    atomic_write_text(report_output, "\n".join(report_lines) + "\n")
    return summary
