from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.experiments.q1q2_external_inference import (
    prepare_nnunet_external_input,
)


class _ExternalInputFixture:
    def __init__(self, root: Path, frame: pd.DataFrame) -> None:
        self.data_root = root
        self.frame = frame
        self.inventory_sha256 = "inventory-sha256"

    def __len__(self) -> int:
        return len(self.frame)


def test_nnunet_external_derivation_preserves_raw_not_normalized_values(
    tmp_path: Path,
) -> None:
    patient_id = "BraTS-SSA-00001-000"
    patient_dir = tmp_path / patient_id
    patient_dir.mkdir()
    row: dict[str, Any] = {"patient_id": patient_id}
    raw = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
    for role in ("t1", "t1ce", "t2", "flair"):
        path = patient_dir / f"{role}.nii.gz"
        nib.save(nib.Nifti1Image(raw, np.eye(4)), str(path))
        row[f"{role}_path"] = path.relative_to(tmp_path).as_posix()
    dataset = _ExternalInputFixture(tmp_path, pd.DataFrame([row]))

    report = prepare_nnunet_external_input(
        dataset=dataset,  # type: ignore[arg-type]
        destination=tmp_path / "derived/imagesTs",
    )

    derived = np.asarray(
        nib.load(str(tmp_path / f"derived/imagesTs/{patient_id}_0000.nii")).dataobj
    )
    assert report["status"] == "complete"
    assert np.array_equal(derived, raw)
