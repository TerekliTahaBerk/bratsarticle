"""Timed official nnU-Net inference without changing its prediction path."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from nnunetv2.inference.export_prediction import (  # type: ignore[import-untyped]
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import (  # type: ignore[import-untyped]
    nnUNetPredictor,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


class Q1Q2TimedNNUNetPredictor(nnUNetPredictor):  # type: ignore[misc]
    """Expose synchronized preprocessing, forward, and postprocessing timings."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self._case_forward_seconds = 0.0

    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
    ) -> torch.Tensor:
        _synchronize(self.device)
        started = time.perf_counter()
        result = super().predict_sliding_window_return_logits(input_image)
        _synchronize(self.device)
        self._case_forward_seconds += time.perf_counter() - started
        return cast(torch.Tensor, result)

    def predict_case_timed(
        self,
        input_files: list[Path],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict one case with the official preprocessor and export transform."""
        if len(input_files) != 4 or not all(path.is_file() for path in input_files):
            raise ValueError("Official nnU-Net inference requires four input files")
        total_started = time.perf_counter()
        preprocessor = self.configuration_manager.preprocessor_class(
            verbose=self.verbose_preprocessing
        )
        preprocessing_started = time.perf_counter()
        data, _, properties = preprocessor.run_case(
            [path.as_posix() for path in input_files],
            None,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        self._case_forward_seconds = 0.0
        logits = self.predict_logits_from_preprocessed_data(
            torch.from_numpy(data)
        ).cpu()
        forward_seconds = self._case_forward_seconds

        postprocessing_started = time.perf_counter()
        prediction = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            self.plans_manager,
            self.configuration_manager,
            self.label_manager,
            properties,
            return_probabilities=False,
        )
        postprocessing_seconds = time.perf_counter() - postprocessing_started
        end_to_end_seconds = time.perf_counter() - total_started
        return np.asarray(prediction), {
            "preprocessing_seconds": preprocessing_seconds,
            "model_forward_seconds": forward_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "end_to_end_seconds": end_to_end_seconds,
        }


__all__ = ["Q1Q2TimedNNUNetPredictor"]
