"""Standard same-padding 2D U-Net baseline."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional


class DoubleConvolution(nn.Module):
    """Two 3-by-3 convolutions with ReLU activations."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""
        return cast(torch.Tensor, self.layers(inputs))


class StandardUNet2D(nn.Module):
    """Contracting/expanding U-Net with skip concatenation and no added modules."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        output_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if input_channels < 1 or output_channels < 2:
            raise ValueError("Invalid U-Net input/output channel count")
        if base_channels < 1 or depth < 1:
            raise ValueError("U-Net base_channels and depth must be positive")
        encoder_channels = [base_channels * (2**index) for index in range(depth)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        previous_channels = input_channels
        for channels in encoder_channels:
            self.encoders.append(DoubleConvolution(previous_channels, channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            previous_channels = channels
        bottleneck_channels = base_channels * (2**depth)
        self.bottleneck = DoubleConvolution(
            encoder_channels[-1],
            bottleneck_channels,
        )
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_input = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.upsamplers.append(
                nn.ConvTranspose2d(
                    decoder_input,
                    skip_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(DoubleConvolution(skip_channels * 2, skip_channels))
            decoder_input = skip_channels
        self.classifier = nn.Conv2d(
            base_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return same-spatial-size class logits."""
        skips: list[torch.Tensor] = []
        output = inputs
        for encoder, pool in zip(self.encoders, self.pools, strict=True):
            output = encoder(output)
            skips.append(output)
            output = pool(output)
        output = self.bottleneck(output)
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
            output = decoder(torch.cat((skip, output), dim=1))
        return cast(torch.Tensor, self.classifier(output))
