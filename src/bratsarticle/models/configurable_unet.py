"""Feature-flagged U-Net family including published BU-Net components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as functional


class ConvolutionNormActivation(nn.Module):
    """Convolution followed by optional batch normalization and ReLU."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int | tuple[int, int],
        batch_normalization: bool,
    ) -> None:
        super().__init__()
        padding: int | tuple[int, int]
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=not batch_normalization,
        )
        self.normalization: nn.Module = (
            nn.BatchNorm2d(output_channels) if batch_normalization else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        return cast(
            torch.Tensor,
            self.activation(self.normalization(self.convolution(inputs))),
        )


class PlainBlock(nn.Module):
    """Two-convolution U-Net block."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        batch_normalization: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvolutionNormActivation(
                input_channels,
                output_channels,
                kernel_size=3,
                batch_normalization=batch_normalization,
            ),
            ConvolutionNormActivation(
                output_channels,
                output_channels,
                kernel_size=3,
                batch_normalization=batch_normalization,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the plain block."""
        return cast(torch.Tensor, self.layers(inputs))


class ResidualBlock(nn.Module):
    """Two-convolution residual block used by the Res U-Net controls."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        batch_normalization: bool,
    ) -> None:
        super().__init__()
        self.first = ConvolutionNormActivation(
            input_channels,
            output_channels,
            kernel_size=3,
            batch_normalization=batch_normalization,
        )
        self.second_convolution = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=not batch_normalization,
        )
        self.second_normalization: nn.Module = (
            nn.BatchNorm2d(output_channels) if batch_normalization else nn.Identity()
        )
        self.projection: nn.Module = (
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        residual = self.projection(inputs)
        output = self.second_normalization(self.second_convolution(self.first(inputs)))
        return cast(torch.Tensor, self.activation(output + residual))


class ResidualExtendedSkip(nn.Module):
    """BU-Net residual extended skip (RES) with four separable-kernel branches."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_sizes: tuple[int, ...] = (9, 11, 13, 15),
        batch_normalization: bool = True,
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(
            kernel < 1 or kernel % 2 == 0 for kernel in kernel_sizes
        ):
            raise ValueError("RES kernels must be positive odd values")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvolutionNormActivation(
                        channels,
                        channels,
                        kernel_size=(kernel, 1),
                        batch_normalization=batch_normalization,
                    ),
                    ConvolutionNormActivation(
                        channels,
                        channels,
                        kernel_size=(1, kernel),
                        batch_normalization=batch_normalization,
                    ),
                )
                for kernel in kernel_sizes
            ]
        )
        self.post = nn.Sequential(
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=3,
                batch_normalization=batch_normalization,
            ),
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=3,
                batch_normalization=batch_normalization,
            ),
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=1,
                batch_normalization=batch_normalization,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sum four contextual branches and identity, then refine."""
        combined = inputs
        for branch in self.branches:
            combined = combined + branch(inputs)
        return cast(torch.Tensor, self.post(combined))


class WideContext(nn.Module):
    """BU-Net wide context (WC) using two ordered separable-kernel branches."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 15,
        batch_normalization: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("WC kernel must be a positive odd value")
        self.vertical_then_horizontal = nn.Sequential(
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                batch_normalization=batch_normalization,
            ),
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                batch_normalization=batch_normalization,
            ),
        )
        self.horizontal_then_vertical = nn.Sequential(
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                batch_normalization=batch_normalization,
            ),
            ConvolutionNormActivation(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                batch_normalization=batch_normalization,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sum the two ordered wide-context branches."""
        return cast(
            torch.Tensor,
            self.vertical_then_horizontal(inputs)
            + self.horizontal_then_vertical(inputs),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Architecture flags shared across the ablation matrix."""

    name: str
    input_channels: int = 4
    output_channels: int = 4
    base_channels: int = 16
    depth: int = 4
    batch_normalization: bool = True
    dropout_probability: float = 0.3
    residual_blocks: bool = False
    residual_extended_skips: bool = False
    wide_context: bool = False
    res_kernel_sizes: tuple[int, ...] = (9, 11, 13, 15)
    wc_kernel_size: int = 15
    parameter_target: int | None = None
    parameter_tolerance_fraction: float = 0.05

    def __post_init__(self) -> None:
        """Validate architecture and parameter-matching controls."""
        if self.input_channels < 1 or self.output_channels < 2:
            raise ValueError("Invalid model channel count")
        if self.base_channels < 1 or self.depth < 1:
            raise ValueError("base_channels and depth must be positive")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        if self.parameter_target is not None and self.parameter_target < 1:
            raise ValueError("parameter_target must be positive")
        if self.parameter_tolerance_fraction < 0:
            raise ValueError("parameter tolerance cannot be negative")


@dataclass(frozen=True)
class ParameterMatchResult:
    """Closest integer base width to a declared parameter target."""

    base_channels: int
    parameter_count: int
    target_parameters: int
    absolute_difference: int
    relative_difference: float
    within_tolerance: bool


class ConfigurableUNet2D(nn.Module):
    """U-Net family with independent residual-block, RES-skip, and WC flags."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        block_type = ResidualBlock if config.residual_blocks else PlainBlock
        encoder_channels = [
            config.base_channels * (2**index) for index in range(config.depth)
        ]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.skip_transforms = nn.ModuleList()
        previous = config.input_channels
        for channels in encoder_channels:
            self.encoders.append(
                block_type(
                    previous,
                    channels,
                    batch_normalization=config.batch_normalization,
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            self.skip_transforms.append(
                ResidualExtendedSkip(
                    channels,
                    kernel_sizes=config.res_kernel_sizes,
                    batch_normalization=config.batch_normalization,
                )
                if config.residual_extended_skips
                else nn.Identity()
            )
            previous = channels
        bottleneck_channels = config.base_channels * (2**config.depth)
        self.bottleneck = block_type(
            encoder_channels[-1],
            bottleneck_channels,
            batch_normalization=config.batch_normalization,
        )
        self.context: nn.Module = (
            WideContext(
                bottleneck_channels,
                kernel_size=config.wc_kernel_size,
                batch_normalization=config.batch_normalization,
            )
            if config.wide_context
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(config.dropout_probability)
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.upsamplers.append(
                nn.ConvTranspose2d(
                    decoder_channels,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(
                block_type(
                    skip_channels * 2,
                    skip_channels,
                    batch_normalization=config.batch_normalization,
                )
            )
            decoder_channels = skip_channels
        self.classifier = nn.Conv2d(
            config.base_channels,
            config.output_channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return same-size logits under the selected feature flags."""
        skips: list[torch.Tensor] = []
        output = inputs
        for encoder, pool, skip_transform in zip(
            self.encoders,
            self.pools,
            self.skip_transforms,
            strict=True,
        ):
            output = encoder(output)
            skips.append(skip_transform(output))
            output = pool(self.dropout(output))
        output = self.context(self.bottleneck(output))
        for upsampler, decoder, skip in zip(
            self.upsamplers,
            self.decoders,
            reversed(skips),
            strict=True,
        ):
            output = upsampler(output)
            if output.shape[-2:] != skip.shape[-2:]:
                output = functional.interpolate(
                    output,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            output = decoder(self.dropout(torch.cat((skip, output), dim=1)))
        return cast(torch.Tensor, self.classifier(output))


def load_model_config(path: Path) -> ModelConfig:
    """Load one model architecture configuration."""
    root = cast(DictConfig, OmegaConf.load(path))
    OmegaConf.resolve(root)
    config = root.model
    target = config.parameter_matching.target_parameters
    return ModelConfig(
        name=str(config.name),
        input_channels=int(config.input_channels),
        output_channels=int(config.output_channels),
        base_channels=int(config.base_channels),
        depth=int(config.depth),
        batch_normalization=bool(config.batch_normalization),
        dropout_probability=float(config.dropout_probability),
        residual_blocks=bool(config.features.residual_blocks),
        residual_extended_skips=bool(config.features.residual_extended_skips),
        wide_context=bool(config.features.wide_context),
        res_kernel_sizes=tuple(int(value) for value in config.res.kernel_sizes),
        wc_kernel_size=int(config.wc.kernel_size),
        parameter_target=None if target is None else int(target),
        parameter_tolerance_fraction=float(
            config.parameter_matching.tolerance_fraction
        ),
    )


def model_from_config(config: ModelConfig) -> ConfigurableUNet2D:
    """Build a configured U-Net-family model."""
    return ConfigurableUNet2D(config)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def find_closest_parameter_match(
    config: ModelConfig,
    *,
    target_parameters: int,
    minimum_base_channels: int = 1,
    maximum_base_channels: int = 128,
    tolerance_fraction: float | None = None,
) -> ParameterMatchResult:
    """Find the closest integer base width without hiding a failed tolerance."""
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive")
    if minimum_base_channels < 1 or maximum_base_channels < minimum_base_channels:
        raise ValueError("Invalid base-channel search interval")
    tolerance = (
        config.parameter_tolerance_fraction
        if tolerance_fraction is None
        else tolerance_fraction
    )
    if tolerance < 0:
        raise ValueError("tolerance_fraction cannot be negative")

    best_width = minimum_base_channels
    best_count = count_trainable_parameters(
        ConfigurableUNet2D(
            ModelConfig(
                **{
                    **config.__dict__,
                    "base_channels": minimum_base_channels,
                }
            )
        )
    )
    best_difference = abs(best_count - target_parameters)
    for width in range(minimum_base_channels + 1, maximum_base_channels + 1):
        candidate = ModelConfig(
            **{
                **config.__dict__,
                "base_channels": width,
            }
        )
        count = count_trainable_parameters(ConfigurableUNet2D(candidate))
        difference = abs(count - target_parameters)
        if difference < best_difference:
            best_width = width
            best_count = count
            best_difference = difference
    relative_difference = best_difference / target_parameters
    return ParameterMatchResult(
        base_channels=best_width,
        parameter_count=best_count,
        target_parameters=target_parameters,
        absolute_difference=best_count - target_parameters,
        relative_difference=relative_difference,
        within_tolerance=relative_difference <= tolerance,
    )


def trace_tensor_shapes(
    model: ConfigurableUNet2D,
    *,
    input_shape: tuple[int, int, int, int] = (1, 4, 64, 64),
) -> list[dict[str, Any]]:
    """Trace major module output shapes with one deterministic dummy forward."""
    trace: list[dict[str, Any]] = []
    handles: list[Any] = []
    selected: list[tuple[str, nn.Module]] = []
    selected.extend(
        (f"encoder_{index}", module) for index, module in enumerate(model.encoders)
    )
    selected.extend(
        (f"skip_{index}", module) for index, module in enumerate(model.skip_transforms)
    )
    selected.append(("bottleneck", model.bottleneck))
    selected.append(("context", model.context))
    selected.extend(
        (f"upsampler_{index}", module) for index, module in enumerate(model.upsamplers)
    )
    selected.extend(
        (f"decoder_{index}", module) for index, module in enumerate(model.decoders)
    )
    selected.append(("classifier", model.classifier))

    def hook(name: str) -> Any:
        def record(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            trace.append({"module": name, "shape": list(output.shape)})

        return record

    for name, module in selected:
        handles.append(module.register_forward_hook(hook(name)))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(input_shape, dtype=torch.float32))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    if tuple(output.shape) != (
        input_shape[0],
        model.config.output_channels,
        input_shape[2],
        input_shape[3],
    ):
        raise RuntimeError("Model shape trace ended with an unexpected output")
    return trace
