"""Guarded single-opening Gate 11 internal-test inference runner."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from bratsarticle.data.dataset import BraTSSliceDataset, build_internal_test_dataset
from bratsarticle.data.preprocessing import load_preprocessing_config
from bratsarticle.experiments.gate10 import assert_clean_repository
from bratsarticle.experiments.hardware import (
    accelerator_available,
    accelerator_device,
    accelerator_device_names,
)
from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    model_from_config,
)
from bratsarticle.training.losses import class_indices_to_labels
from bratsarticle.training.reproducibility import collect_run_metadata, seed_everything
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json
from evaluation import CentralEvaluator, load_evaluation_config


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def load_gate11_plan(path: Path) -> dict[str, Any]:
    """Load and validate the pre-access Gate 11 execution contract."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(loaded, "configuration")
    plan = dict(_mapping(root.get("gate11"), "gate11"))
    if int(plan.get("gate", -1)) != 11:
        raise ValueError("Gate 11 plan must declare gate: 11")
    if plan.get("status") != "frozen_pre_access":
        raise ValueError("Gate 11 plan must be frozen before test access")
    inference = _mapping(plan["inference"], "inference")
    if bool(inference["mixed_precision"]):
        raise ValueError("Gate 11 must match the frozen full-precision protocol")
    if int(inference["expected_patients"]) != 74:
        raise ValueError("Gate 11 expects 74 internal-test patients")
    if int(inference["expected_checkpoints"]) != 13:
        raise ValueError("Gate 11 expects 13 frozen checkpoints")
    access = _mapping(plan["access"], "access")
    if int(access["maximum_manifest_open_events"]) != 1:
        raise ValueError("Gate 11 requires exactly one manifest opening")
    if not bool(access["refuse_if_prior_access_event_exists"]):
        raise ValueError("Gate 11 must reject prior test access")
    qualitative = _mapping(plan["qualitative"], "qualitative")
    if int(qualitative["frozen_seed"]) != 20260729:
        raise ValueError("Qualitative seed must be fixed before test access")
    return plan


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _prior_test_access_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event") == "internal_test_manifest_access"
    ]


def _gate10_paths(plan: Mapping[str, Any]) -> dict[str, Path]:
    gate10 = _mapping(plan["gate10"], "gate10")
    return {key: Path(str(value)) for key, value in gate10.items()}


def _validate_gate10_hashes(
    gate10_paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analysis = json.loads(
        gate10_paths["analysis_freeze"].read_text(encoding="utf-8")
    )
    checkpoints = json.loads(
        gate10_paths["checkpoint_manifest"].read_text(encoding="utf-8")
    )
    split_metadata_path = gate10_paths["frozen_split_dir"] / "split_metadata.json"
    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    if analysis["status"] != "frozen" or analysis["internal_test_accessed"]:
        raise RuntimeError("Gate 10 analysis freeze is not test-eligible")
    if file_digest(gate10_paths["plan"]) != analysis["plan_sha256"]:
        raise RuntimeError("Gate 10 statistical plan hash changed")
    if (
        file_digest(gate10_paths["checkpoint_manifest"])
        != analysis["checkpoint_manifest_sha256"]
    ):
        raise RuntimeError("Gate 10 checkpoint manifest hash changed")
    if file_digest(split_metadata_path) != analysis["frozen_split_metadata_sha256"]:
        raise RuntimeError("Frozen split metadata hash changed")
    if not split_metadata["frozen"] or split_metadata["status"] != "pass":
        raise RuntimeError("Frozen split is not valid")
    for split_name in ("train", "validation", "test"):
        manifest = gate10_paths["frozen_split_dir"] / f"{split_name}.csv"
        if file_digest(manifest) != analysis["manifest_sha256"][split_name]:
            raise RuntimeError(f"Frozen {split_name} manifest hash changed")
    if (
        file_digest(gate10_paths["evaluation_config"])
        != checkpoints["evaluation_config_sha256"]
    ):
        raise RuntimeError("Frozen evaluator config hash changed")
    if (
        file_digest(gate10_paths["preprocessing_config"])
        != checkpoints["preprocessing_config_sha256"]
    ):
        raise RuntimeError("Frozen preprocessing config hash changed")
    entries = cast(list[dict[str, Any]], checkpoints["checkpoints"])
    if checkpoints["checkpoint_count"] != 13 or len(entries) != 13:
        raise RuntimeError("Gate 10 must pin exactly 13 checkpoints")
    for entry in entries:
        checkpoint = Path(str(entry["checkpoint_path"]))
        model_config = Path(str(entry["model_config_path"]))
        if file_digest(checkpoint) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash changed: {checkpoint}")
        if checkpoint.stat().st_size != entry["checkpoint_size_bytes"]:
            raise RuntimeError(f"Checkpoint size changed: {checkpoint}")
        if file_digest(model_config) != entry["model_config_sha256"]:
            raise RuntimeError(f"Model config hash changed: {model_config}")
    return analysis, checkpoints, split_metadata


def gate11_preflight(plan_path: Path) -> dict[str, Any]:
    """Validate every prerequisite without opening the test manifest."""
    plan = load_gate11_plan(plan_path)
    gate10_paths = _gate10_paths(plan)
    access = _mapping(plan["access"], "access")
    environment = _mapping(plan["environment"], "environment")
    audit_path = Path(str(access["audit_log"]))
    dataset_variable = str(environment["dataset_root_variable"])
    cache_variable = str(environment["cache_root_variable"])
    dataset_raw = os.environ.get(dataset_variable)
    cache_raw = os.environ.get(cache_variable)
    checks: dict[str, bool] = {
        "repository_clean": False,
        "gate10_hashes_valid": False,
        "no_prior_test_access": not _prior_test_access_events(audit_path),
        "dataset_root_configured": bool(dataset_raw),
        "cache_root_configured": bool(cache_raw),
        "mps_available": accelerator_available("mps"),
        "device_name_matches": str(environment["accelerator_device_name"])
        in accelerator_device_names("mps"),
        "disk_space_sufficient": False,
    }
    try:
        assert_clean_repository()
        checks["repository_clean"] = True
    except RuntimeError:
        pass
    try:
        _validate_gate10_hashes(gate10_paths)
        checks["gate10_hashes_valid"] = True
    except (OSError, KeyError, RuntimeError, ValueError):
        pass
    if dataset_raw and cache_raw:
        dataset_root = Path(dataset_raw).expanduser().resolve()
        cache_root = Path(cache_raw).expanduser().resolve()
        try:
            assert_output_paths_safe([cache_root], [dataset_root])
            checks["cache_outside_raw_root"] = True
        except ValueError:
            checks["cache_outside_raw_root"] = False
        free_bytes = shutil.disk_usage(cache_root.parent).free
        checks["disk_space_sufficient"] = free_bytes >= int(
            environment["minimum_free_disk_bytes"]
        )
    else:
        checks["cache_outside_raw_root"] = False
        free_bytes = None
    return {
        "status": "eligible" if all(checks.values()) else "ineligible",
        "eligible": all(checks.values()),
        "gate": 11,
        "plan_path": plan_path.as_posix(),
        "plan_sha256": file_digest(plan_path),
        "git_commit": _git_commit(),
        "checks": checks,
        "accelerator_backend": "mps",
        "accelerator_device_names": accelerator_device_names("mps"),
        "free_disk_bytes": free_bytes,
        "test_access_event_count": len(_prior_test_access_events(audit_path)),
        "test_manifest_opened": False,
    }


def write_gate11_preflight(plan_path: Path, output: Path) -> dict[str, Any]:
    """Write the current-host Gate 11 preflight."""
    report = gate11_preflight(plan_path)
    atomic_write_json(output, report)
    return report


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_model_slices(
    model: nn.Module,
    image: np.ndarray,
    *,
    slice_axis: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    slices = np.moveaxis(image, slice_axis + 1, 1)
    predictions: list[np.ndarray] = []
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, slices.shape[1], batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(
                    slices[:, start : start + batch_size].transpose(1, 0, 2, 3),
                    dtype=np.float32,
                )
            ).to(device)
            logits = model(batch)
            labels = class_indices_to_labels(torch.argmax(logits, dim=1))
            predictions.append(labels.detach().cpu().numpy().astype(np.int16))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    stacked = np.concatenate(predictions, axis=0)
    volume = np.moveaxis(stacked, 0, slice_axis)
    return np.ascontiguousarray(volume, dtype=np.int16), elapsed


def _warmup(
    model: nn.Module,
    volume: np.ndarray,
    *,
    slice_axis: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> None:
    slices = np.moveaxis(volume, slice_axis + 1, 1)
    with torch.no_grad():
        for batch_index in range(batches):
            start = (batch_index * batch_size) % slices.shape[1]
            chunk = slices[:, start : start + batch_size].transpose(1, 0, 2, 3)
            model(
                torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)).to(
                    device
                )
            )
    _synchronize(device)


def _atomic_savez(destination: Path, **arrays: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _qualitative_assets(
    root: Path,
    *,
    candidate: str,
    subject_id: str,
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    slice_axis: int,
    context_candidate: str,
) -> None:
    subject_root = root / "qualitative" / subject_id
    _atomic_savez(
        subject_root / f"{candidate}.npz",
        prediction_label=np.asarray(prediction, dtype=np.int16),
    )
    if candidate != context_candidate:
        return
    target_wt = target != 0
    lesion_counts = np.sum(
        target_wt,
        axis=tuple(axis for axis in range(3) if axis != slice_axis),
    )
    error_counts = np.sum(
        prediction != target,
        axis=tuple(axis for axis in range(3) if axis != slice_axis),
    )
    lesion_slice = int(np.argmax(lesion_counts))
    error_slice = int(np.argmax(error_counts))
    _atomic_savez(
        subject_root / "context.npz",
        target_label=np.asarray(target, dtype=np.int16),
        lesion_slice_index=np.asarray(lesion_slice, dtype=np.int16),
        error_slice_index=np.asarray(error_slice, dtype=np.int16),
        lesion_image=np.asarray(
            np.take(image, lesion_slice, axis=slice_axis + 1),
            dtype=np.float32,
        ),
        error_image=np.asarray(
            np.take(image, error_slice, axis=slice_axis + 1),
            dtype=np.float32,
        ),
    )


def _model_complexity(model_config_path: Path) -> tuple[int, int]:
    model = model_from_config(load_model_config(model_config_path)).eval()
    sample = torch.zeros((1, 4, 240, 240), dtype=torch.float32)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(sample)
    flops = int(counter.get_total_flops())
    return flops // 2, flops


def _load_model(entry: Mapping[str, Any], device: torch.device) -> nn.Module:
    model_config_path = Path(str(entry["model_config_path"]))
    model = model_from_config(load_model_config(model_config_path))
    payload = cast(
        dict[str, Any],
        torch.load(
            Path(str(entry["checkpoint_path"])),
            map_location="cpu",
            weights_only=False,
        ),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    if count_trainable_parameters(model) != int(entry["parameter_count"]):
        raise RuntimeError("Loaded model parameter count differs from Gate 10")
    return model


def _cohort_metadata(dataset: BraTSSliceDataset) -> list[dict[str, Any]]:
    columns = [
        "subject_id",
        "grade",
        "wt_voxel_count",
        "tc_voxel_count",
        "et_voxel_count",
        "voxel_volume_mm3",
        "wt_volume_mm3",
        "tc_volume_mm3",
        "et_volume_mm3",
        "et_present",
    ]
    return cast(
        list[dict[str, Any]],
        dataset.manifest.loc[:, columns].to_dict(orient="records"),
    )


def _run_checkpoint(
    entry: Mapping[str, Any],
    *,
    dataset: BraTSSliceDataset,
    evaluator: CentralEvaluator,
    device: torch.device,
    plan_path: Path,
    plan: Mapping[str, Any],
    artifact_root: Path,
    split_hashes: Mapping[str, str],
    qualitative_candidates: set[str],
    qualitative_seed: int,
    context_candidate: str,
) -> None:
    candidate = str(entry["candidate_id"])
    seed = int(entry["seed"])
    output = artifact_root / candidate / str(seed)
    if output.exists():
        raise FileExistsError(f"Gate 11 output already exists: {output}")
    output.mkdir(parents=True)
    inference = _mapping(plan["inference"], "inference")
    batch_size = int(inference["batch_size_slices"])
    model = _load_model(entry, device)
    first_volume = dataset.subject_volume(0)
    _warmup(
        model,
        first_volume.image,
        slice_axis=dataset.config.slice_axis,
        batch_size=batch_size,
        batches=int(inference["warmup_batches_per_checkpoint"]),
        device=device,
    )
    rows: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    started = datetime.now(UTC)
    for patient_index in range(len(dataset.manifest)):
        volume = dataset.subject_volume(patient_index)
        subject_id = str(dataset.manifest.iloc[patient_index]["subject_id"])
        prediction, latency = _decode_model_slices(
            model,
            volume.image,
            slice_axis=dataset.config.slice_axis,
            batch_size=batch_size,
            device=device,
        )
        patient_rows = evaluator.evaluate_batch(
            prediction,
            volume.label,
            patient_ids=[subject_id],
            spacings_mm=[volume.spacing_mm],
        )
        if len(patient_rows) != 1:
            raise RuntimeError("Gate 11 raw evaluator must return one patient row")
        rows.append(
            {
                "candidate_id": candidate,
                "seed": seed,
                "run_id": str(entry["run_id"]),
                **patient_rows[0],
            }
        )
        latencies.append(
            {
                "candidate_id": candidate,
                "seed": seed,
                "run_id": str(entry["run_id"]),
                "patient_id": subject_id,
                "latency_seconds": latency,
            }
        )
        if seed == qualitative_seed and candidate in qualitative_candidates:
            _qualitative_assets(
                artifact_root,
                candidate=candidate,
                subject_id=subject_id,
                image=volume.image,
                target=volume.label,
                prediction=prediction,
                slice_axis=dataset.config.slice_axis,
                context_candidate=context_candidate,
            )
        print(
            json.dumps(
                {
                    "event": "gate11_patient_completed",
                    "candidate_id": candidate,
                    "seed": seed,
                    "patient_index": patient_index + 1,
                    "patient_count": len(dataset.manifest),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    atomic_write_csv(output / "patient_metrics.csv", rows)
    atomic_write_csv(output / "latency.csv", latencies)
    macs, flops = _model_complexity(Path(str(entry["model_config_path"])))
    metadata = collect_run_metadata(
        config_path=plan_path,
        split_hashes=split_hashes,
        seed=seed,
        device=device,
        mixed_precision=False,
        run_kind="gate11_internal_test_inference",
    )
    metadata.update(
        {
            "status": "completed",
            "candidate_id": candidate,
            "run_id": str(entry["run_id"]),
            "checkpoint_path": str(entry["checkpoint_path"]),
            "checkpoint_sha256": str(entry["checkpoint_sha256"]),
            "model_config_path": str(entry["model_config_path"]),
            "model_config_sha256": str(entry["model_config_sha256"]),
            "patient_count": len(rows),
            "latency_scope": str(inference["latency_scope"]),
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "parameter_count": int(entry["parameter_count"]),
            "macs_per_slice": macs,
            "flops_per_slice": flops,
            "input_specification": [1, 4, 240, 240],
        }
    )
    atomic_write_json(output / "metadata.json", metadata)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def run_gate11(
    plan_path: Path,
    *,
    allow_test_evaluation: bool,
) -> None:
    """Open the frozen test manifest once and evaluate all 13 checkpoints."""
    if not allow_test_evaluation:
        raise PermissionError("Gate 11 requires --allow-test-evaluation")
    assert_clean_repository()
    plan = load_gate11_plan(plan_path)
    preflight = gate11_preflight(plan_path)
    if not preflight["eligible"]:
        failed = [
            name
            for name, passed in cast(Mapping[str, bool], preflight["checks"]).items()
            if not passed
        ]
        raise RuntimeError(f"Gate 11 preflight failed: {failed}")
    gate10_paths = _gate10_paths(plan)
    analysis, checkpoint_manifest, split_metadata = _validate_gate10_hashes(
        gate10_paths
    )
    del analysis
    access = _mapping(plan["access"], "access")
    audit_log = Path(str(access["audit_log"]))
    if _prior_test_access_events(audit_log):
        raise RuntimeError("Gate 11 refuses a second internal-test manifest opening")
    inference = _mapping(plan["inference"], "inference")
    environment = _mapping(plan["environment"], "environment")
    qualitative = _mapping(plan["qualitative"], "qualitative")
    artifact_root = Path(str(inference["artifact_root"]))
    if artifact_root.exists():
        raise FileExistsError(f"Gate 11 artifact root already exists: {artifact_root}")
    dataset_root = Path(
        os.environ[str(environment["dataset_root_variable"])]
    ).expanduser()
    cache_root = Path(os.environ[str(environment["cache_root_variable"])]).expanduser()
    assert_output_paths_safe([artifact_root, cache_root], [dataset_root])
    device = accelerator_device("mps")
    seed = int(inference["seed"])
    seed_everything(seed)
    preprocessing = load_preprocessing_config(gate10_paths["preprocessing_config"])
    dataset = build_internal_test_dataset(
        gate10_paths["frozen_split_dir"],
        dataset_root,
        preprocessing,
        seed=seed,
        allow_test_evaluation=True,
        purpose=str(access["purpose"]),
        audit_log=audit_log,
    )
    if len(dataset.manifest) != int(inference["expected_patients"]):
        raise RuntimeError("Internal-test patient count differs from Gate 11")
    artifact_root.mkdir(parents=True)
    atomic_write_csv(
        artifact_root / "cohort_metadata.csv",
        _cohort_metadata(dataset),
    )
    evaluator = CentralEvaluator(
        load_evaluation_config(gate10_paths["evaluation_config"])
    )
    qualitative_candidates = {
        str(value)
        for value in cast(
            Sequence[Any],
            qualitative["save_full_prediction_labels_for_candidates"],
        )
    }
    entries = cast(
        Sequence[Mapping[str, Any]],
        checkpoint_manifest["checkpoints"],
    )
    split_hashes = cast(Mapping[str, str], split_metadata["manifest_sha256"])
    run_metadata = {
        "status": "running",
        "gate": 11,
        "git_commit": _git_commit(),
        "repository_dirty_before_access": False,
        "plan_path": plan_path.as_posix(),
        "plan_sha256": file_digest(plan_path),
        "checkpoint_manifest_sha256": file_digest(
            gate10_paths["checkpoint_manifest"]
        ),
        "frozen_split_metadata_sha256": file_digest(
            gate10_paths["frozen_split_dir"] / "split_metadata.json"
        ),
        "test_manifest_sha256": split_hashes["test"],
        "checkpoint_count": len(entries),
        "patient_count": len(dataset.manifest),
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "device_names": accelerator_device_names("mps"),
        },
        "access_audit_log": audit_log.as_posix(),
    }
    atomic_write_json(artifact_root / "run_metadata.json", run_metadata)
    for index, entry in enumerate(entries, start=1):
        print(
            json.dumps(
                {
                    "event": "gate11_checkpoint_started",
                    "index": index,
                    "total": len(entries),
                    "candidate_id": entry["candidate_id"],
                    "seed": entry["seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _run_checkpoint(
            entry,
            dataset=dataset,
            evaluator=evaluator,
            device=device,
            plan_path=plan_path,
            plan=plan,
            artifact_root=artifact_root,
            split_hashes=split_hashes,
            qualitative_candidates=qualitative_candidates,
            qualitative_seed=int(qualitative["frozen_seed"]),
            context_candidate=str(qualitative["context_candidate"]),
        )
        print(
            json.dumps(
                {
                    "event": "gate11_checkpoint_completed",
                    "index": index,
                    "total": len(entries),
                    "candidate_id": entry["candidate_id"],
                    "seed": entry["seed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    run_metadata["status"] = "completed"
    run_metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    atomic_write_json(artifact_root / "run_metadata.json", run_metadata)
