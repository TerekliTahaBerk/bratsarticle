from dataclasses import replace
from pathlib import Path

import pytest
import torch

from bratsarticle.models import (
    ConfigurableUNet2D,
    ModelConfig,
    ResidualExtendedSkip,
    WideContext,
    find_closest_parameter_match,
)
from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    trace_tensor_shapes,
)

MODEL_CONFIGS = sorted(Path("configs/models").glob("*.yaml"))


@pytest.mark.parametrize(
    ("module"),
    [
        ResidualExtendedSkip(
            3,
            kernel_sizes=(3, 5),
            batch_normalization=False,
        ),
        WideContext(3, kernel_size=5, batch_normalization=False),
    ],
)
def test_bunet_component_preserves_shape_and_gradient(module: torch.nn.Module) -> None:
    inputs = torch.randn((2, 3, 17, 19), requires_grad=True)

    output = module(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("config_path", MODEL_CONFIGS, ids=lambda path: path.stem)
def test_every_ablation_configuration_runs(config_path: Path) -> None:
    configured = load_model_config(config_path)
    small = replace(
        configured,
        base_channels=2,
        depth=2,
        batch_normalization=False,
        dropout_probability=0.0,
        res_kernel_sizes=(3, 5),
        wc_kernel_size=5,
    )
    model = ConfigurableUNet2D(small)
    inputs = torch.randn((1, 4, 31, 35), requires_grad=True)

    logits = model(inputs)
    logits.mean().backward()

    assert logits.shape == (1, 4, 31, 35)
    assert inputs.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_ablation_feature_flags_are_independent() -> None:
    observed: dict[str, tuple[bool, bool, bool]] = {}
    for config_path in MODEL_CONFIGS:
        config = load_model_config(config_path)
        observed[config_path.stem] = (
            config.residual_blocks,
            config.residual_extended_skips,
            config.wide_context,
        )

    assert observed == {
        "bunet": (False, True, True),
        "resunet": (True, False, False),
        "resunet_wc": (True, False, True),
        "unet": (False, False, False),
        "unet_res": (False, True, False),
        "unet_wc": (False, False, True),
    }


def test_shape_trace_covers_major_modules() -> None:
    config = ModelConfig(
        name="trace",
        base_channels=2,
        depth=2,
        batch_normalization=False,
        dropout_probability=0.0,
        residual_extended_skips=True,
        wide_context=True,
        res_kernel_sizes=(3,),
        wc_kernel_size=3,
    )

    trace = trace_tensor_shapes(
        ConfigurableUNet2D(config),
        input_shape=(1, 4, 32, 32),
    )

    assert trace[-1] == {"module": "classifier", "shape": [1, 4, 32, 32]}
    assert {entry["module"] for entry in trace} == {
        "encoder_0",
        "encoder_1",
        "skip_0",
        "skip_1",
        "bottleneck",
        "context",
        "upsampler_0",
        "upsampler_1",
        "decoder_0",
        "decoder_1",
        "classifier",
    }


def test_parameter_match_reports_success_and_failure_honestly() -> None:
    config = ModelConfig(
        name="small",
        base_channels=3,
        depth=1,
        batch_normalization=False,
    )
    exact_target = count_trainable_parameters(ConfigurableUNet2D(config))

    exact = find_closest_parameter_match(
        config,
        target_parameters=exact_target,
        maximum_base_channels=6,
    )
    impossible = find_closest_parameter_match(
        config,
        target_parameters=exact_target + 1,
        maximum_base_channels=6,
        tolerance_fraction=0.0,
    )

    assert exact.base_channels == 3
    assert exact.absolute_difference == 0
    assert exact.within_tolerance
    assert not impossible.within_tolerance
