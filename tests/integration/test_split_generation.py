from pathlib import Path

import pandas as pd

from bratsarticle.data.splits import SplitSettings, generate_split


def _write_manifest(path: Path, count: int = 100) -> None:
    rows = []
    for index in range(count):
        group = index % 4
        rows.append(
            {
                "subject_id": f"subject_{index:03d}",
                "grade": "HGG" if group < 3 else "LGG",
                "eligible": True,
                "wt_voxel_count": 100 + index * 7,
                "tc_voxel_count": 40 + index * 3,
                "et_voxel_count": 0 if group == 3 else 5 + index,
                "voxel_volume_mm3": 1.0,
                "t1_sha256": f"t1_{index}",
                "t1ce_sha256": f"t1ce_{index}",
                "t2_sha256": f"t2_{index}",
                "flair_sha256": f"flair_{index}",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _settings(tmp_path: Path, output_name: str) -> SplitSettings:
    return SplitSettings(
        canonical_manifest=tmp_path / "canonical.csv",
        output_dir=tmp_path / output_name / "splits",
        figure_dir=tmp_path / output_name / "figures",
        report_path=tmp_path / output_name / "report.md",
        metadata_path=tmp_path / output_name / "splits" / "metadata.json",
        seed=42,
        candidate_seeds=16,
        counts={"train": 70, "validation": 10, "test": 20},
        max_categorical_prevalence_deviation=0.30,
        max_absolute_standardized_mean_difference=0.60,
    )


def test_split_generation_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "canonical.csv")
    first = generate_split(_settings(tmp_path, "first"))
    second = generate_split(_settings(tmp_path, "second"))

    assert first["manifest_sha256"] == second["manifest_sha256"]
    memberships = {
        split: set(
            pd.read_csv(tmp_path / "first" / "splits" / f"{split}.csv")["subject_id"]
        )
        for split in ("train", "validation", "test")
    }
    assert len(memberships["train"]) == 70
    assert len(memberships["validation"]) == 10
    assert len(memberships["test"]) == 20
    assert memberships["train"].isdisjoint(memberships["validation"])
    assert memberships["train"].isdisjoint(memberships["test"])
    assert memberships["validation"].isdisjoint(memberships["test"])
    assert len(set().union(*memberships.values())) == 100
    for hash_column in ("t1_sha256", "t1ce_sha256", "t2_sha256", "flair_sha256"):
        hash_sets = {
            split: set(
                pd.read_csv(tmp_path / "first" / "splits" / f"{split}.csv")[hash_column]
            )
            for split in ("train", "validation", "test")
        }
        assert hash_sets["train"].isdisjoint(hash_sets["validation"])
        assert hash_sets["train"].isdisjoint(hash_sets["test"])
        assert hash_sets["validation"].isdisjoint(hash_sets["test"])
