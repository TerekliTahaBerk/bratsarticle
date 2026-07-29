"""Create or verify the Gate 13 tracked-artifact manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json

DEFAULT_MANIFEST = Path("reports/tracked_artifact_manifest.json")
EXCLUDED_PATHS = {
    DEFAULT_MANIFEST.as_posix(),
}
EXCLUDED_PREFIXES = (
    "reports/gate13_",
)
INCLUDED_FILES = {
    "AGENTS.md",
    "pyproject.toml",
}
INCLUDED_PREFIXES = (
    "artifacts/internal_test/gate11/",
    "artifacts/test_access_log.jsonl",
    "configs/",
    "environment/",
    "figures/final/",
    "reports/",
    "scripts/",
    "splits/frozen/",
    "src/",
    "tables/final/",
    "tests/",
)


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    candidates = [
        Path(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    ]
    selected = []
    for path in candidates:
        value = path.as_posix()
        if value in EXCLUDED_PATHS:
            continue
        if any(value.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if value in INCLUDED_FILES or any(
            value.startswith(prefix) for prefix in INCLUDED_PREFIXES
        ):
            selected.append(path)
    return sorted(selected, key=lambda path: path.as_posix())


def _entry(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": file_digest(path),
        "size_bytes": path.stat().st_size,
    }


def build_manifest() -> dict[str, Any]:
    """Return a deterministic manifest of tracked computational artifacts."""
    paths = _tracked_paths()
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Tracked files are absent: {missing}")
    return {
        "schema_version": 1,
        "gate": 13,
        "algorithm": "sha256",
        "selection": {
            "included_files": sorted(INCLUDED_FILES),
            "included_prefixes": list(INCLUDED_PREFIXES),
            "excluded_paths": sorted(EXCLUDED_PATHS),
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
        },
        "entry_count": len(paths),
        "entries": [_entry(path) for path in paths],
    }


def verify_manifest(path: Path) -> dict[str, Any]:
    """Verify file membership, sizes, and hashes against a saved manifest."""
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_manifest()
    expected_by_path = {
        entry["path"]: entry for entry in expected.get("entries", [])
    }
    actual_by_path = {
        entry["path"]: entry for entry in actual.get("entries", [])
    }
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    unexpected = sorted(set(actual_by_path) - set(expected_by_path))
    mismatched = []
    for file_path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected_entry = expected_by_path[file_path]
        actual_entry = actual_by_path[file_path]
        if (
            expected_entry["sha256"] != actual_entry["sha256"]
            or expected_entry["size_bytes"] != actual_entry["size_bytes"]
        ):
            mismatched.append(
                {
                    "path": file_path,
                    "expected_sha256": expected_entry["sha256"],
                    "actual_sha256": actual_entry["sha256"],
                    "expected_size_bytes": expected_entry["size_bytes"],
                    "actual_size_bytes": actual_entry["size_bytes"],
                }
            )
    valid = (
        expected.get("schema_version") == 1
        and expected.get("algorithm") == "sha256"
        and expected.get("entry_count") == len(expected_by_path)
        and not missing
        and not unexpected
        and not mismatched
    )
    return {
        "valid": valid,
        "entry_count": len(actual_by_path),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create or verify the manifest."""
    arguments = _parser().parse_args(argv)
    if arguments.write:
        manifest = build_manifest()
        atomic_write_json(arguments.manifest, manifest)
        print(
            json.dumps(
                {
                    "status": "written",
                    "path": arguments.manifest.as_posix(),
                    "entry_count": manifest["entry_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = verify_manifest(arguments.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
