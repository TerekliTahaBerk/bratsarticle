"""Outcome-independent static profiling for arbitrary PyTorch modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.flop_counter import FlopCounterMode


@dataclass(frozen=True)
class TorchStaticProfile:
    """Static shape, parameter, activation, and operation-count metadata."""

    parameter_count: int
    flops_per_input: int
    mac_equivalents_per_input: int
    largest_single_activation_bytes: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    flop_counter: str = "torch.utils.flop_counter.FlopCounterMode"
    mac_definition: str = "one_mac_equivalent_equals_two_counted_flops"


def profile_torch_module(
    model: nn.Module,
    *,
    input_shape: Sequence[int],
) -> dict[str, object]:
    """Profile one declared input using FakeTensorMode without real allocation."""
    shape = tuple(int(value) for value in input_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("Static resource profiling requires a positive input shape")
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count < 1:
        raise ValueError("Static resource profiling requires trainable parameters")
    largest_activation_bytes = 0

    def activation_hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: object,
    ) -> None:
        nonlocal largest_activation_bytes
        tensors: list[torch.Tensor] = []
        if isinstance(output, torch.Tensor):
            tensors = [output]
        elif isinstance(output, (tuple, list)):
            tensors = [value for value in output if isinstance(value, torch.Tensor)]
        for tensor in tensors:
            largest_activation_bytes = max(
                largest_activation_bytes,
                tensor.numel() * tensor.element_size(),
            )

    hooks = [
        module.register_forward_hook(activation_hook)
        for module in model.modules()
        if not any(module.children())
    ]
    was_training = model.training
    try:
        model.eval()
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_input = torch.empty(shape)
            with FlopCounterMode(display=False) as counter, torch.no_grad():
                output = model(fake_input)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Static resource profiling requires one tensor output")
        flops = int(counter.get_total_flops())
        if flops < 1:
            raise RuntimeError("Static FLOP counter returned no operations")
        profile = TorchStaticProfile(
            parameter_count=parameter_count,
            flops_per_input=flops,
            mac_equivalents_per_input=flops // 2,
            largest_single_activation_bytes=largest_activation_bytes,
            input_shape=shape,
            output_shape=tuple(int(value) for value in output.shape),
        )
        return asdict(profile)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)


__all__ = ["TorchStaticProfile", "profile_torch_module"]
