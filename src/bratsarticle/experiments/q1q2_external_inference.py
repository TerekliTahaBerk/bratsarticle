"""Adapter-specific inference for the single frozen external session."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.inferers.utils import sliding_window_inference

from bratsarticle.adapters.nnunetv2 import nnunet_to_brats_labels
from bratsarticle.data.dataset import extract_context_slices
from bratsarticle.data.external_dataset import ExternalVolumeDataset
from bratsarticle.data.preprocessing import load_preprocessing_config
from bratsarticle.experiments.q1q2_swin_runner import _model as build_swin_model
from bratsarticle.models.configurable_unet import (
    load_model_config,
    model_from_config,
)
from bratsarticle.training.losses import class_indices_to_labels
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json
from evaluation import CentralEvaluator


def _checkpoint_state(
    path: Path,
    *,
    expected_sha256: str,
    expected_run_id: str,
    device: torch.device,
) -> dict[str, Any]:
    if file_digest(path) != expected_sha256:
        raise ValueError(f"Frozen checkpoint hash differs: {path}")
    payload = cast(
        dict[str, Any],
        torch.load(path, map_location=device, weights_only=False),
    )
    metadata = cast(dict[str, Any], payload.get("metadata", {}))
    if metadata.get("run_id") != expected_run_id:
        raise ValueError("Checkpoint run identity differs from Gate G")
    return cast(dict[str, Any], payload["model"])


def _identified_rows(
    rows: list[dict[str, Any]],
    *,
    run: Mapping[str, Any],
    checkpoint_sha256: str,
    volume_metadata: Mapping[str, Any],
    cohort_role: str,
) -> list[dict[str, Any]]:
    return [
        {
            "external_run_id": str(run["run_id"]),
            "model_id": str(run["model_id"]),
            "training_fold": int(run["fold"]),
            "training_seed": int(run["seed"]),
            "checkpoint_role": "best_development",
            "checkpoint_sha256": checkpoint_sha256,
            "cohort_role": cohort_role,
            **dict(volume_metadata),
            **row,
        }
        for row in rows
    ]


def predict_native_external_checkpoint(
    *,
    run: Mapping[str, Any],
    dataset: ExternalVolumeDataset,
    evaluator: CentralEvaluator,
    model_config_path: Path,
    preprocessing_config_path: Path,
    validation_batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Infer every external volume with one frozen native checkpoint."""
    if device.type != "mps":
        raise ValueError("Reportable native external inference requires MPS")
    model = model_from_config(load_model_config(model_config_path))
    checkpoint_path = Path(str(run["best_checkpoint_path"]))
    checkpoint_sha256 = str(run["best_checkpoint_sha256"])
    model.load_state_dict(
        _checkpoint_state(
            checkpoint_path,
            expected_sha256=checkpoint_sha256,
            expected_run_id=str(run["run_id"]),
            device=device,
        )
    )
    model.to(device)
    model.eval()
    preprocessing = load_preprocessing_config(preprocessing_config_path)
    slice_axis = preprocessing.slice_axis
    offsets = (-2, -1, 0, 1, 2) if str(run["model_id"]) == "unet_2p5d_k5" else (0,)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for patient_index in range(len(dataset)):
            volume = dataset.load(patient_index)
            slice_count = volume.label.shape[slice_axis]
            predicted_slices: list[np.ndarray] = []
            for start in range(0, slice_count, validation_batch_size):
                indices = range(
                    start,
                    min(start + validation_batch_size, slice_count),
                )
                images = np.stack(
                    [
                        extract_context_slices(
                            volume.image,
                            slice_index,
                            slice_axis=slice_axis,
                            context_offsets=offsets,
                        )
                        for slice_index in indices
                    ]
                )
                logits = model(torch.from_numpy(images).to(device, dtype=torch.float32))
                labels = class_indices_to_labels(torch.argmax(logits, dim=1))
                predicted_slices.extend(
                    np.asarray(item, dtype=np.int16)
                    for item in labels.detach().cpu().numpy()
                )
            prediction = np.stack(predicted_slices, axis=slice_axis)
            metric_rows = evaluator.evaluate_batch(
                prediction,
                volume.label,
                patient_ids=[volume.patient_id],
                spacings_mm=[volume.spacing_mm],
            )
            rows.extend(
                _identified_rows(
                    metric_rows,
                    run=run,
                    checkpoint_sha256=checkpoint_sha256,
                    volume_metadata=volume.metadata,
                    cohort_role=volume.cohort_role,
                )
            )
    return rows


def predict_swin_external_checkpoint(
    *,
    run: Mapping[str, Any],
    dataset: ExternalVolumeDataset,
    evaluator: CentralEvaluator,
    model_config_path: Path,
    overlap: float,
    mode: str,
    sliding_window_batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Infer every external volume with one frozen Swin checkpoint."""
    if device.type != "mps":
        raise ValueError("Reportable Swin external inference requires MPS")
    model, patch_size = build_swin_model(model_config_path)
    checkpoint_path = Path(str(run["best_checkpoint_path"]))
    checkpoint_sha256 = str(run["best_checkpoint_sha256"])
    model.load_state_dict(
        _checkpoint_state(
            checkpoint_path,
            expected_sha256=checkpoint_sha256,
            expected_run_id=str(run["run_id"]),
            device=device,
        )
    )
    model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for patient_index in range(len(dataset)):
            volume = dataset.load(patient_index)
            image = torch.from_numpy(
                np.ascontiguousarray(volume.image[None], dtype=np.float32)
            )
            logits = cast(
                torch.Tensor,
                sliding_window_inference(
                    image,
                    roi_size=patch_size,
                    sw_batch_size=sliding_window_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                    sw_device=device,
                    device=torch.device("cpu"),
                ),
            )
            prediction = class_indices_to_labels(torch.argmax(logits, dim=1))
            metric_rows = evaluator.evaluate_batch(
                prediction.numpy(),
                volume.label,
                patient_ids=[volume.patient_id],
                spacings_mm=[volume.spacing_mm],
            )
            rows.extend(
                _identified_rows(
                    metric_rows,
                    run=run,
                    checkpoint_sha256=checkpoint_sha256,
                    volume_metadata=volume.metadata,
                    cohort_role=volume.cohort_role,
                )
            )
    return rows


def prepare_nnunet_external_input(
    *,
    dataset: ExternalVolumeDataset,
    destination: Path,
) -> dict[str, Any]:
    """Create the official four-channel uncompressed NIfTI input once."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.parent / "derivation_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "complete"
            or existing.get("inventory_sha256") != dataset.inventory_sha256
            or int(existing.get("patient_count", -1)) != len(dataset)
        ):
            raise ValueError("Existing nnU-Net external derivation differs")
        stored_hashes = cast(
            dict[str, dict[str, str]],
            existing["input_sha256_by_patient_channel"],
        )
        for patient_id, channels in stored_hashes.items():
            for stored_channel, expected_hash in channels.items():
                path = destination / f"{patient_id}_{int(stored_channel):04d}.nii"
                if not path.is_file() or file_digest(path) != expected_hash:
                    raise ValueError("Derived nnU-Net external input hash differs")
        return cast(dict[str, Any], existing)
    hashes: dict[str, dict[str, str]] = {}
    for patient_index in range(len(dataset)):
        row = dataset.frame.iloc[patient_index]
        patient_id = str(row["patient_id"])
        patient_hashes: dict[str, str] = {}
        for channel_index, role in enumerate(("t1", "t1ce", "t2", "flair")):
            source = (dataset.data_root / str(row[f"{role}_path"])).resolve()
            if not source.is_relative_to(dataset.data_root) or not source.is_file():
                raise PermissionError(
                    f"nnU-Net external input escapes the data root: {source}"
                )
            path = destination / f"{patient_id}_{channel_index:04d}.nii"
            source_image = cast(
                nib.Nifti1Image,
                nib.load(str(source), mmap="r"),
            )
            if not path.is_file():
                nib.save(
                    nib.Nifti1Image(  # type: ignore[no-untyped-call]
                        np.asarray(source_image.dataobj),
                        source_image.affine,
                        header=source_image.header.copy(),  # type: ignore[no-untyped-call]
                    ),
                    str(path),
                )
            derived_image = cast(
                nib.Nifti1Image,
                nib.load(str(path), mmap="r"),
            )
            if not np.array_equal(
                np.asarray(derived_image.dataobj),
                np.asarray(source_image.dataobj),
            ) or not np.allclose(
                derived_image.affine,
                source_image.affine,
                rtol=0.0,
                atol=1e-5,
            ):
                raise ValueError(
                    f"Derived nnU-Net external input differs: {patient_id}:{role}"
                )
            patient_hashes[str(channel_index)] = file_digest(path)
        hashes[patient_id] = patient_hashes
    report = {
        "schema_version": 1,
        "status": "complete",
        "patient_count": len(hashes),
        "inventory_sha256": dataset.inventory_sha256,
        "input_directory": destination.as_posix(),
        "input_sha256_by_patient_channel": hashes,
    }
    atomic_write_json(manifest_path, report)
    return report


def _nnunet_job(queue_path: Path, run_id: str) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    matches = [
        cast(dict[str, Any], job)
        for job in cast(list[Any], queue["jobs"])
        if cast(dict[str, Any], job).get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one official nnU-Net job for {run_id}")
    return matches[0]


def predict_nnunet_external_checkpoint(
    *,
    run: Mapping[str, Any],
    dataset: ExternalVolumeDataset,
    evaluator: CentralEvaluator,
    input_directory: Path,
    queue_path: Path,
    environment: Mapping[str, str],
    log_directory: Path,
) -> list[dict[str, Any]]:
    """Run official best-checkpoint inference and evaluate before cleanup."""
    job = _nnunet_job(queue_path, str(run["run_id"]))
    checkpoint_path = Path(str(run["best_checkpoint_path"]))
    checkpoint_sha256 = str(run["best_checkpoint_sha256"])
    if file_digest(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Official nnU-Net external checkpoint hash differs")
    command = [
        "nnUNetv2_predict",
        "-i",
        input_directory.as_posix(),
        "-o",
        "{output}",
        "-d",
        "501",
        "-p",
        str(job["plans_identifier"]),
        "-tr",
        str(job["trainer"]),
        "-c",
        str(job["configuration"]),
        "-f",
        str(job["fold_nnunet_zero_indexed"]),
        "-chk",
        "checkpoint_best.pth",
        "-npp",
        "1",
        "-nps",
        "1",
        "-device",
        "mps",
        "--disable_progress_bar",
    ]
    rows: list[dict[str, Any]] = []
    log_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{run['run_id']}-",
    ) as raw_output:
        output = Path(raw_output)
        resolved_command = [
            output.as_posix() if value == "{output}" else value for value in command
        ]
        with (
            (log_directory / "stdout.log").open("a", encoding="utf-8") as stdout,
            (log_directory / "stderr.log").open("a", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                resolved_command,
                env=dict(environment),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Official nnU-Net external inference failed: {run['run_id']}"
            )
        expected = {
            f"{dataset.frame.iloc[index]['patient_id']}.nii"
            for index in range(len(dataset))
        }
        observed = {path.name for path in output.glob("*.nii")}
        if observed != expected:
            raise RuntimeError("Official nnU-Net external prediction set differs")
        for patient_index in range(len(dataset)):
            volume = dataset.load(patient_index)
            prediction_path = output / f"{volume.patient_id}.nii"
            prediction_image = cast(
                nib.Nifti1Image,
                nib.load(str(prediction_path), mmap="r"),
            )
            nnunet_prediction = np.asarray(prediction_image.dataobj)
            if nnunet_prediction.shape != volume.label.shape:
                raise ValueError(
                    f"nnU-Net external shape differs for {volume.patient_id}"
                )
            metric_rows = evaluator.evaluate_batch(
                nnunet_to_brats_labels(nnunet_prediction),
                volume.label,
                patient_ids=[volume.patient_id],
                spacings_mm=[volume.spacing_mm],
            )
            rows.extend(
                _identified_rows(
                    metric_rows,
                    run=run,
                    checkpoint_sha256=checkpoint_sha256,
                    volume_metadata=volume.metadata,
                    cohort_role=volume.cohort_role,
                )
            )
    return rows


def native_model_config(model_matrix_path: Path, model_id: str) -> Path:
    """Resolve one native model config from the frozen matrix."""
    matrix = yaml.safe_load(model_matrix_path.read_text(encoding="utf-8"))
    matches = [
        cast(dict[str, Any], entry)
        for entry in cast(list[Any], matrix["main_models"])
        if cast(dict[str, Any], entry).get("id") == model_id
    ]
    if len(matches) != 1 or matches[0].get("adapter") != "native_configurable_unet":
        raise ValueError(f"Native external model entry differs: {model_id}")
    return Path(str(matches[0]["config"]))


__all__ = [
    "native_model_config",
    "predict_native_external_checkpoint",
    "predict_nnunet_external_checkpoint",
    "predict_swin_external_checkpoint",
    "prepare_nnunet_external_input",
]
