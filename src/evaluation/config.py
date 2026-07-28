"""Typed configuration for the central evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig, OmegaConf

OutputMode = Literal["labels", "softmax", "nested_sigmoid"]
ConsistencyRule = Literal["outward_union"]
EvaluationStage = Literal["raw", "filtered"]
MatchingMethod = Literal["maximum_total_iou"]


@dataclass(frozen=True)
class EmptyMaskRules:
    """Explicit behavior for empty reference or prediction masks."""

    overlap_both_empty: float = 1.0
    overlap_one_empty: float = 0.0
    surface_dice_both_empty: float = 1.0
    surface_dice_one_empty: float = 0.0
    hd95_both_empty_mm: float = 0.0
    hd95_one_empty: Literal["infinity"] = "infinity"
    undefined_rate: Literal["nan"] = "nan"
    relative_volume_error_empty_reference: Literal["zero_or_infinity"] = (
        "zero_or_infinity"
    )


@dataclass(frozen=True)
class LesionEvaluationConfig:
    """Connected-component and lesion-matching policy."""

    connectivity: Literal[6, 18, 26] = 26
    minimum_voxels: int = 1
    minimum_volume_mm3: float = 0.0
    matching_method: MatchingMethod = "maximum_total_iou"
    minimum_match_iou: float = 0.0


@dataclass(frozen=True)
class PostprocessingConfig:
    """Visible evaluation stages and small-component filtering settings."""

    stages: tuple[EvaluationStage, ...] = ("raw",)
    minimum_prediction_voxels: int = 10
    minimum_prediction_volume_mm3: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Complete immutable evaluator policy."""

    output_mode: OutputMode = "labels"
    from_logits: bool = True
    wt_threshold: float = 0.5
    tc_threshold: float = 0.5
    et_threshold: float = 0.5
    enforce_nested_consistency: bool = False
    consistency_rule: ConsistencyRule = "outward_union"
    surface_tolerance_mm: float = 1.0
    hd_percentile: float = 95.0
    default_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    empty_masks: EmptyMaskRules = EmptyMaskRules()
    lesions: LesionEvaluationConfig = LesionEvaluationConfig()
    postprocessing: PostprocessingConfig = PostprocessingConfig()

    def __post_init__(self) -> None:
        """Reject ambiguous or invalid evaluation settings."""
        if self.output_mode not in {"labels", "softmax", "nested_sigmoid"}:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")
        if self.consistency_rule != "outward_union":
            raise ValueError(
                f"Unsupported nested consistency rule: {self.consistency_rule}"
            )
        for threshold in (self.wt_threshold, self.tc_threshold, self.et_threshold):
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Nested sigmoid thresholds must be in [0, 1]")
        if self.surface_tolerance_mm < 0:
            raise ValueError("Surface Dice tolerance cannot be negative")
        if not 0.0 < self.hd_percentile <= 100.0:
            raise ValueError("Hausdorff percentile must be in (0, 100]")
        if self.empty_masks.hd95_one_empty != "infinity":
            raise ValueError("One-empty-mask HD behavior must be infinity")
        if self.empty_masks.undefined_rate != "nan":
            raise ValueError("Undefined rate behavior must be nan")
        if self.empty_masks.relative_volume_error_empty_reference != "zero_or_infinity":
            raise ValueError(
                "Empty-reference relative volume behavior must be zero_or_infinity"
            )
        if len(self.default_spacing_mm) != 3 or any(
            value <= 0 for value in self.default_spacing_mm
        ):
            raise ValueError("Default spacing must contain three positive values")
        if self.lesions.minimum_voxels < 1:
            raise ValueError("Lesion minimum_voxels must be at least one")
        if self.lesions.connectivity not in {6, 18, 26}:
            raise ValueError("Lesion connectivity must be 6, 18, or 26")
        if self.lesions.minimum_volume_mm3 < 0:
            raise ValueError("Lesion minimum volume cannot be negative")
        if not 0.0 <= self.lesions.minimum_match_iou <= 1.0:
            raise ValueError("Minimum lesion-match IoU must be in [0, 1]")
        if self.lesions.matching_method != "maximum_total_iou":
            raise ValueError(
                f"Unsupported lesion matching: {self.lesions.matching_method}"
            )
        if not self.postprocessing.stages:
            raise ValueError("At least one evaluation stage is required")
        if len(set(self.postprocessing.stages)) != len(self.postprocessing.stages):
            raise ValueError("Evaluation stages must be unique")
        if any(
            stage not in {"raw", "filtered"} for stage in self.postprocessing.stages
        ):
            raise ValueError("Evaluation stages must be raw and/or filtered")
        if "filtered" in self.postprocessing.stages and (
            self.postprocessing.minimum_prediction_voxels < 1
        ):
            raise ValueError("Filtered evaluation requires a positive voxel threshold")
        if self.postprocessing.minimum_prediction_volume_mm3 < 0:
            raise ValueError("Postprocessing minimum volume cannot be negative")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load and validate an evaluator YAML configuration."""
    raw = OmegaConf.load(path)
    OmegaConf.resolve(raw)
    config = cast(DictConfig, raw).evaluation
    empty = config.empty_masks
    lesions = config.lesions
    postprocessing = config.postprocessing
    if list(config.nested_sigmoid.channel_order) != ["wt", "tc", "et"]:
        raise ValueError("Nested sigmoid channel order must be [wt, tc, et]")
    spacing = tuple(float(value) for value in config.default_spacing_mm)
    if len(spacing) != 3:
        raise ValueError("default_spacing_mm must contain exactly three values")
    stages = tuple(cast(EvaluationStage, str(value)) for value in postprocessing.stages)
    return EvaluationConfig(
        output_mode=cast(OutputMode, str(config.output_mode)),
        from_logits=bool(config.from_logits),
        wt_threshold=float(config.nested_sigmoid.thresholds.wt),
        tc_threshold=float(config.nested_sigmoid.thresholds.tc),
        et_threshold=float(config.nested_sigmoid.thresholds.et),
        enforce_nested_consistency=bool(config.nested_sigmoid.enforce_consistency),
        consistency_rule=cast(
            ConsistencyRule, str(config.nested_sigmoid.consistency_rule)
        ),
        surface_tolerance_mm=float(config.surface_dice_tolerance_mm),
        hd_percentile=float(config.hd_percentile),
        default_spacing_mm=spacing,
        empty_masks=EmptyMaskRules(
            overlap_both_empty=float(empty.overlap_both_empty),
            overlap_one_empty=float(empty.overlap_one_empty),
            surface_dice_both_empty=float(empty.surface_dice_both_empty),
            surface_dice_one_empty=float(empty.surface_dice_one_empty),
            hd95_both_empty_mm=float(empty.hd95_both_empty_mm),
            hd95_one_empty=cast(Literal["infinity"], str(empty.hd95_one_empty)),
            undefined_rate=cast(Literal["nan"], str(empty.undefined_rate)),
            relative_volume_error_empty_reference=cast(
                Literal["zero_or_infinity"],
                str(empty.relative_volume_error_empty_reference),
            ),
        ),
        lesions=LesionEvaluationConfig(
            connectivity=cast(Literal[6, 18, 26], int(lesions.connectivity)),
            minimum_voxels=int(lesions.minimum_voxels),
            minimum_volume_mm3=float(lesions.minimum_volume_mm3),
            matching_method=cast(MatchingMethod, str(lesions.matching_method)),
            minimum_match_iou=float(lesions.minimum_match_iou),
        ),
        postprocessing=PostprocessingConfig(
            stages=stages,
            minimum_prediction_voxels=int(postprocessing.minimum_prediction_voxels),
            minimum_prediction_volume_mm3=float(
                postprocessing.minimum_prediction_volume_mm3
            ),
        ),
    )
