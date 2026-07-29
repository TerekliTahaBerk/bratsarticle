"""Run the Gate 13 verification suite inside a data-free clean clone."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

RAW_ENVIRONMENT_KEYS = (
    "BRATS2019_ROOT",
    "BRATS2020_ROOT",
    "BRATS_CACHE_ROOT",
)


def _run(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    start = time.monotonic()
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": name,
        "command": list(command),
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - start, 3),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _git_value(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate 13 Clean-Clone Reproducibility Audit",
        "",
        f"**Decision:** {report['decision']}",
        "",
        "## Audited snapshot",
        "",
        f"- Commit: `{report['audited_commit']}`",
        f"- Tree: `{report['audited_tree']}`",
        f"- Tracked manifest entries: {report['manifest_entry_count']}",
        f"- Raw-data environment variables removed: "
        f"{', '.join(report['removed_environment_keys'])}",
        f"- Started (UTC): {report['started_at_utc']}",
        f"- Finished (UTC): {report['finished_at_utc']}",
        "",
        "## Checks",
        "",
        "| Check | Result | Seconds |",
        "|---|---|---:|",
    ]
    for result in report["commands"]:
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        lines.append(
            f"| {result['name']} | {status} | "
            f"{result['duration_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Test result",
            "",
            f"- Pytest summary: `{report['pytest_summary']}`",
            f"- Generated reporting outputs reproduced byte-for-byte: "
            f"{str(report['reporting_outputs_byte_identical']).lower()}",
            f"- Final clean-clone worktree clean: "
            f"{str(report['final_worktree_clean']).lower()}",
            "",
            "## Scope boundary",
            "",
            "This audit reproduces all tracked analyses, figures, tables, hashes, "
            "and software tests without raw BraTS roots, caches, or local model "
            "checkpoints. Full retraining and a new internal-test inference pass "
            "are deliberately outside this clean-clone check: they require the "
            "authorized dataset and frozen checkpoint bundle, and a new test pass "
            "would require a separately logged access event.",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("reports/gate13_reproduction.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("reports/gate13_reproduction.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolated audit and write machine- and human-readable reports."""
    arguments = _parser().parse_args(argv)
    repository = arguments.repository.resolve()
    if _git_value(repository, "status", "--porcelain"):
        raise RuntimeError("Gate 13 requires a clean source worktree")

    started_at = datetime.now(UTC)
    commit = _git_value(repository, "rev-parse", "HEAD")
    tree = _git_value(repository, "rev-parse", "HEAD^{tree}")
    python_executable = Path(sys.executable).absolute()
    executable_root = python_executable.parent
    ruff_executable = executable_root / "ruff"
    mypy_executable = executable_root / "mypy"
    commands: list[dict[str, Any]] = []
    clone_clean = False
    reporting_outputs_identical = False
    pytest_summary = "not run"
    manifest_entry_count = 0

    with tempfile.TemporaryDirectory(prefix="bratsarticle-gate13-") as temporary:
        temporary_root = Path(temporary)
        clone = temporary_root / "repository"
        environment = dict(os.environ)
        for key in RAW_ENVIRONMENT_KEYS:
            environment.pop(key, None)
        environment["MPLCONFIGDIR"] = (temporary_root / "matplotlib").as_posix()

        clone_result = _run(
            "clean local clone",
            [
                "git",
                "clone",
                "--no-local",
                "--branch",
                "main",
                "--single-branch",
                repository.as_posix(),
                clone.as_posix(),
            ],
            cwd=temporary_root,
            environment=environment,
        )
        commands.append(clone_result)
        if clone_result["returncode"] == 0:
            environment["PYTHONPATH"] = (clone / "src").as_posix()
            manifest_path = clone / "reports/tracked_artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_entry_count = int(manifest["entry_count"])
            audit_commands = (
                (
                    "tracked artifact hashes",
                    [
                        python_executable.as_posix(),
                        "scripts/manage_tracked_artifact_manifest.py",
                        "--verify",
                    ],
                ),
                (
                    "ruff",
                    [
                        ruff_executable.as_posix(),
                        "check",
                        "src",
                        "tests",
                        "scripts",
                    ],
                ),
                (
                    "mypy",
                    [mypy_executable.as_posix()],
                ),
                (
                    "pytest",
                    [python_executable.as_posix(), "-m", "pytest", "-q"],
                ),
                (
                    "Gate 12 generation pass 1",
                    [
                        python_executable.as_posix(),
                        "scripts/generate_gate12_outputs.py",
                    ],
                ),
                (
                    "Gate 12 byte identity pass 1",
                    ["git", "diff", "--exit-code"],
                ),
                (
                    "Gate 12 generation pass 2",
                    [
                        python_executable.as_posix(),
                        "scripts/generate_gate12_outputs.py",
                    ],
                ),
                (
                    "Gate 12 byte identity pass 2",
                    ["git", "diff", "--exit-code"],
                ),
                (
                    "final tracked artifact hashes",
                    [
                        python_executable.as_posix(),
                        "scripts/manage_tracked_artifact_manifest.py",
                        "--verify",
                    ],
                ),
            )
            for name, command in audit_commands:
                result = _run(
                    name,
                    command,
                    cwd=clone,
                    environment=environment,
                )
                commands.append(result)
                if name == "pytest":
                    match = re.search(
                        r"(\d+ passed(?:, \d+ skipped)?[^\\n]*)",
                        result["stdout"],
                    )
                    pytest_summary = match.group(1) if match else "unparsed"
            identity_results = [
                result
                for result in commands
                if result["name"].startswith("Gate 12 byte identity")
            ]
            reporting_outputs_identical = bool(identity_results) and all(
                result["returncode"] == 0 for result in identity_results
            )
            status = _git_value(
                clone,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
            clone_clean = not status

    decision = (
        "PASS"
        if commands
        and all(result["returncode"] == 0 for result in commands)
        and clone_clean
        and reporting_outputs_identical
        else "FAIL"
    )
    finished_at = datetime.now(UTC)
    report = {
        "schema_version": 1,
        "gate": 13,
        "decision": decision,
        "audited_commit": commit,
        "audited_tree": tree,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "source_repository": repository.as_posix(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "requirements_lock_sha256": file_digest(
            repository / "environment/requirements-lock.txt"
        ),
        "removed_environment_keys": list(RAW_ENVIRONMENT_KEYS),
        "manifest_entry_count": manifest_entry_count,
        "commands": commands,
        "pytest_summary": pytest_summary,
        "reporting_outputs_byte_identical": reporting_outputs_identical,
        "final_worktree_clean": clone_clean,
        "scope_boundary": {
            "reproduced": (
                "Tracked analyses, figures, tables, hashes, and software tests"
            ),
            "not_reexecuted": (
                "Full training and internal-test inference requiring authorized "
                "raw data and frozen local checkpoints"
            ),
        },
    }
    json_report = (
        arguments.json_report
        if arguments.json_report.is_absolute()
        else repository / arguments.json_report
    )
    markdown_report = (
        arguments.markdown_report
        if arguments.markdown_report.is_absolute()
        else repository / arguments.markdown_report
    )
    atomic_write_json(json_report, report)
    atomic_write_text(markdown_report, _markdown(report))
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
