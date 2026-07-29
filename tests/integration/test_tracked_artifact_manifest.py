import json
from pathlib import Path

from bratsarticle.utils.hashing import file_digest


def test_tracked_artifact_manifest_hashes_are_current() -> None:
    manifest = json.loads(
        Path("reports/tracked_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == 1
    assert manifest["gate"] == 13
    assert manifest["algorithm"] == "sha256"
    assert manifest["entry_count"] == len(manifest["entries"])
    assert manifest["entry_count"] >= 200
    assert len({entry["path"] for entry in manifest["entries"]}) == manifest[
        "entry_count"
    ]
    for entry in manifest["entries"]:
        path = Path(entry["path"])
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert file_digest(path) == entry["sha256"]
