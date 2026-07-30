"""Guarded, immutable volume access for the frozen BraTS-Africa cohort."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.data.external_audit import MODALITY_ORDER
from bratsarticle.data.preprocessing import zscore_nonzero
from bratsarticle.utils.hashing import file_digest, text_digest


@dataclass(frozen=True)
class ExternalVolume:
    """Normalized four-channel image, mapped BraTS label, and geometry."""

    patient_id: str
    cohort_role: str
    image: np.ndarray
    label: np.ndarray
    spacing_mm: tuple[float, float, float]
    metadata: dict[str, Any]


def _inventory_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {
        "patient_id",
        "disease_group",
        "eligibility_status",
        "t1_path",
        "t1ce_path",
        "t2_path",
        "flair_path",
        "label_path",
        "t1_sha256",
        "t1ce_sha256",
        "t2_sha256",
        "flair_sha256",
        "label_sha256",
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"External inventory misses columns: {sorted(missing)}")
    eligible = frame.loc[frame["eligibility_status"].eq("eligible")].copy()
    if len(eligible) != 146 or not eligible["patient_id"].is_unique:
        raise ValueError("External inventory must contain 146 unique eligible patients")
    counts = eligible["disease_group"].value_counts().to_dict()
    if counts != {"glioma": 95, "other_neoplasm": 51}:
        raise ValueError(f"External disease-group counts differ: {counts}")
    return eligible.sort_values("patient_id").reset_index(drop=True)


def _authorized_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError(f"External path escapes the authorized root: {relative}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def verify_external_files(
    *,
    data_root: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    """Re-hash all frozen external inputs once after Gate G opens Gate H."""
    root = data_root.resolve()
    frame = _inventory_frame(inventory_path)
    observed: dict[str, dict[str, str]] = {}
    roles = ("t1", "t1ce", "t2", "flair", "label")
    for row in frame.to_dict(orient="records"):
        patient_id = str(row["patient_id"])
        hashes: dict[str, str] = {}
        for role in roles:
            path = _authorized_path(root, str(row[f"{role}_path"]))
            digest = file_digest(path)
            if digest != str(row[f"{role}_sha256"]):
                raise ValueError(
                    f"External source hash differs for {patient_id}:{role}"
                )
            hashes[role] = digest
        observed[patient_id] = hashes
    return {
        "schema_version": 1,
        "status": "verified",
        "patient_count": len(frame),
        "inventory_path": inventory_path.as_posix(),
        "inventory_sha256": file_digest(inventory_path),
        "source_hash_index_sha256": text_digest(
            json.dumps(observed, sort_keys=True, separators=(",", ":"))
        ),
    }


class ExternalVolumeDataset:
    """Read frozen external patients through an immutable mmap cache."""

    def __init__(
        self,
        *,
        data_root: Path,
        inventory_path: Path,
        cache_root: Path,
    ) -> None:
        self.data_root = data_root.resolve()
        self.inventory_path = inventory_path
        self.inventory_sha256 = file_digest(inventory_path)
        self.frame = _inventory_frame(inventory_path)
        self.cache_root = cache_root.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def _cache_directory(self, patient_id: str) -> Path:
        return self.cache_root / f"{patient_id}-{self.inventory_sha256[:16]}.npycache"

    def _load_raw(self, row: dict[str, Any]) -> ExternalVolume:
        images: list[np.ndarray] = []
        reference_shape: tuple[int, ...] | None = None
        reference_affine: np.ndarray | None = None
        spacing: tuple[float, float, float] | None = None
        for role in MODALITY_ORDER:
            path = _authorized_path(self.data_root, str(row[f"{role}_path"]))
            image = cast(nib.Nifti1Image, nib.load(str(path), mmap="r"))
            values = np.asarray(image.dataobj)
            affine = np.asarray(image.affine, dtype=np.float64)
            current_spacing = tuple(
                float(value)
                for value in image.header.get_zooms()[:3]  # type: ignore[no-untyped-call]
            )
            if reference_shape is None:
                reference_shape = values.shape
                reference_affine = affine
                spacing = (
                    current_spacing[0],
                    current_spacing[1],
                    current_spacing[2],
                )
            else:
                assert reference_affine is not None
                if values.shape != reference_shape or not np.allclose(
                    affine,
                    reference_affine,
                    rtol=0.0,
                    atol=1e-5,
                ):
                    raise ValueError(
                        f"External modality geometry differs for {row['patient_id']}"
                    )
            images.append(zscore_nonzero(values))
        label_path = _authorized_path(self.data_root, str(row["label_path"]))
        label_image = cast(nib.Nifti1Image, nib.load(str(label_path), mmap="r"))
        source_label = np.asarray(label_image.dataobj)
        if (
            source_label.shape != reference_shape
            or reference_affine is None
            or not np.allclose(
                label_image.affine,
                reference_affine,
                rtol=0.0,
                atol=1e-5,
            )
            or spacing is None
        ):
            raise ValueError(f"External label geometry differs for {row['patient_id']}")
        observed = set(int(value) for value in np.unique(source_label))
        if not observed.issubset({0, 1, 2, 3}):
            raise ValueError(f"Unexpected external labels: {sorted(observed)}")
        mapped_label = np.where(source_label == 3, 4, source_label).astype(
            np.int16,
            copy=False,
        )
        role = (
            "external_confirmatory"
            if str(row["disease_group"]) == "glioma"
            else "external_supportive_other_neoplasm"
        )
        metadata = {
            "disease_group": str(row["disease_group"]),
            "institution": str(row["institution"]),
            "scanner_vendor": str(row["scanner_vendor"]),
            "scanner_model": str(row["scanner_model"]),
            "field_strength_t": row["field_strength_t"],
        }
        return ExternalVolume(
            patient_id=str(row["patient_id"]),
            cohort_role=role,
            image=np.ascontiguousarray(np.stack(images), dtype=np.float32),
            label=np.ascontiguousarray(mapped_label, dtype=np.int16),
            spacing_mm=spacing,
            metadata=metadata,
        )

    def _write_cache(self, destination: Path, volume: ExternalVolume) -> None:
        temporary = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
            )
        )
        try:
            np.save(
                temporary / "image.npy",
                volume.image,
                allow_pickle=False,
            )
            np.save(
                temporary / "label.npy",
                volume.label,
                allow_pickle=False,
            )
            np.save(
                temporary / "spacing_mm.npy",
                np.asarray(volume.spacing_mm, dtype=np.float32),
                allow_pickle=False,
            )
            (temporary / "metadata.json").write_text(
                json.dumps(
                    {
                        "patient_id": volume.patient_id,
                        "cohort_role": volume.cohort_role,
                        "metadata": volume.metadata,
                        "inventory_sha256": self.inventory_sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (temporary / "COMPLETE").touch()
            try:
                temporary.replace(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def load(self, index: int) -> ExternalVolume:
        """Load one patient, creating only a derived cache when absent."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = {str(key): value for key, value in self.frame.iloc[index].items()}
        patient_id = str(row["patient_id"])
        cache = self._cache_directory(patient_id)
        if not (cache / "COMPLETE").is_file():
            self._write_cache(cache, self._load_raw(row))
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata.get("patient_id") != patient_id
            or metadata.get("inventory_sha256") != self.inventory_sha256
        ):
            raise ValueError(f"External cache provenance differs for {patient_id}")
        image = np.load(cache / "image.npy", mmap_mode="r", allow_pickle=False)
        label = np.load(cache / "label.npy", mmap_mode="r", allow_pickle=False)
        spacing_values = np.load(
            cache / "spacing_mm.npy",
            allow_pickle=False,
        )
        if image.shape[0] != 4 or image.shape[1:] != label.shape:
            raise ValueError(f"External cache shape differs for {patient_id}")
        spacing = tuple(float(value) for value in spacing_values)
        if len(spacing) != 3:
            raise ValueError(f"External cache spacing differs for {patient_id}")
        return ExternalVolume(
            patient_id=patient_id,
            cohort_role=str(metadata["cohort_role"]),
            image=image,
            label=label,
            spacing_mm=(spacing[0], spacing[1], spacing[2]),
            metadata=cast(dict[str, Any], metadata["metadata"]),
        )

    def materialize(self) -> dict[str, Any]:
        """Materialize all 146 immutable caches within an authorized session."""
        patient_ids = [self.load(index).patient_id for index in range(len(self))]
        return {
            "schema_version": 1,
            "status": "complete",
            "patient_count": len(patient_ids),
            "patient_ids_sha256": text_digest(
                json.dumps(patient_ids, separators=(",", ":"))
            ),
            "inventory_sha256": self.inventory_sha256,
            "cache_root": self.cache_root.as_posix(),
        }


__all__ = [
    "ExternalVolume",
    "ExternalVolumeDataset",
    "verify_external_files",
]
