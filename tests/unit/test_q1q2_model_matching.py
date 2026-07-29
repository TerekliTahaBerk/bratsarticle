from __future__ import annotations

from bratsarticle.models.configurable_unet import ModelConfig
from bratsarticle.models.matching import profile_model, search_plain_unet_controls


def test_static_profiler_matches_audited_standard_unet() -> None:
    profile = profile_model(ModelConfig(name="unet", base_channels=16, depth=4))

    assert profile.parameter_count == 1_942_772
    assert profile.macs_per_slice == 2_676_326_400
    assert profile.flops_per_slice == 5_352_652_800
    assert profile.output_shape == (1, 4, 240, 240)


def test_matching_search_never_adds_res_or_wc() -> None:
    target = ModelConfig(
        name="res",
        base_channels=4,
        depth=2,
        residual_extended_skips=True,
        res_kernel_sizes=(3, 5),
    )

    candidates, parameter_match, compute_match = search_plain_unet_controls(
        target,
        widths=range(2, 8),
        depths=range(1, 4),
        tolerance_fraction=0.25,
    )

    assert len(candidates) == 18
    assert parameter_match.profile.name.startswith("plain_unet_")
    assert compute_match.profile.name.startswith("plain_unet_")
    assert parameter_match.target_value == profile_model(target).parameter_count
    assert compute_match.target_value == profile_model(target).macs_per_slice
