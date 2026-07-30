import torch

from bratsarticle.models.resource_profile import profile_torch_module


def test_generic_static_profile_counts_conv_flops_without_real_allocation() -> None:
    model = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)

    profile = profile_torch_module(model, input_shape=(1, 4, 16, 16))

    assert profile["parameter_count"] == 4 * 8 * 3 * 3
    assert profile["input_shape"] == (1, 4, 16, 16)
    assert profile["output_shape"] == (1, 8, 16, 16)
    assert int(profile["flops_per_input"]) > 0
    assert profile["mac_equivalents_per_input"] == (
        int(profile["flops_per_input"]) // 2
    )
