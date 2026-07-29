import hashlib
import json
import subprocess
from pathlib import Path


def test_legacy_tracked_artifact_manifest_matches_immutable_tag() -> None:
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
        payload = subprocess.run(
            [
                "git",
                "show",
                f"v1-bounded-2d-component-study:{entry['path']}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
