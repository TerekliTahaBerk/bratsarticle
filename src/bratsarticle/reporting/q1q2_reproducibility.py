"""Gate I provenance manifest for the frozen Q1/Q2 experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def run_gate_i_clean_clone_audit(
    *,
    output_path: Path,
) -> dict[str, Any]:
    repository = Path.cwd().resolve()
    commit = _git_value("rev-parse", "HEAD")
    branch = _git_value("branch", "--show-current")
    if not branch:
        raise RuntimeError("Gate I clean-clone audit requires a named branch")
    environment = dict(os.environ)
    removed_keys = (
        "BRATS2019_ROOT",
        "BRATS2020_ROOT",
        "BRATS_AFRICA_ROOT",
        "BRATS_CACHE_ROOT",
    )
    for key in removed_keys:
        environment.pop(key, None)
    python = Path(sys.executable).absolute()
    executable_root = python.parent
    results: list[dict[str, Any]] = []
    final_clean = False
    with tempfile.TemporaryDirectory(prefix="bratsarticle-q1q2-gate-i-") as raw:
        root = Path(raw)
        clone = root / "repository"
        environment["MPLCONFIGDIR"] = (root / "matplotlib").as_posix()
        clone_result = _run(
            [
                "git",
                "clone",
                "--no-local",
                "--single-branch",
                "--branch",
                branch,
                repository.as_posix(),
                clone.as_posix(),
            ],
            cwd=root,
            environment=environment,
        )
        results.append(clone_result)
        if clone_result["returncode"] == 0:
            cloned_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=clone,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if cloned_commit != commit:
                raise RuntimeError("Gate I clean clone resolved a different commit")
            environment["PYTHONPATH"] = (clone / "src").as_posix()
            for command in (
                [
                    (executable_root / "ruff").as_posix(),
                    "check",
                    "src",
                    "tests",
                    "scripts",
                ],
                [(executable_root / "mypy").as_posix()],
                [python.as_posix(), "-m", "pytest", "-q"],
            ):
                results.append(
                    _run(
                        command,
                        cwd=clone,
                        environment=environment,
                    )
                )
            final_clean = not subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                cwd=clone,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
    passed = all(result["returncode"] == 0 for result in results) and final_clean
    report = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "source_commit": commit,
        "source_branch": branch,
        "raw_data_environment_keys_removed": list(removed_keys),
        "commands": results,
        "final_clone_clean": final_clean,
        "raw_data_opened": False,
        "external_inference_performed": False,
    }
    atomic_write_json(output_path, report)
    if not passed:
        raise RuntimeError("Gate I clean-clone audit failed")
    return report


def _result_output_paths(prerequisites: dict[str, Path]) -> list[Path]:
    paths = [
        prerequisites["statistics_completion"],
        prerequisites["subgroup_completion"],
        prerequisites["resource_completion"],
        prerequisites["figure_completion"],
    ]
    statistics = _load_json(prerequisites["statistics_completion"])
    subgroups = _load_json(prerequisites["subgroup_completion"])
    resources = _load_json(prerequisites["resource_completion"])
    figures = _load_json(prerequisites["figure_completion"])
    for completion in (statistics, subgroups):
        for entry in cast(
            dict[str, dict[str, str]],
            completion["outputs"],
        ).values():
            paths.append(Path(entry["path"]))
    paths.extend(Path(path) for path in cast(dict[str, str], resources["artifacts"]))
    for outputs in cast(dict[str, dict[str, str]], figures["figures"]).values():
        paths.extend(Path(path) for path in outputs)
    return sorted(set(paths), key=lambda path: path.as_posix())


def _reproduce_numerical_outputs(
    prerequisites: dict[str, Path],
) -> dict[str, Any]:
    paths = _result_output_paths(prerequisites)
    before = {path.as_posix(): file_digest(path) for path in paths}
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (Path.cwd() / "src").as_posix()
    with tempfile.TemporaryDirectory(prefix="bratsarticle-q1q2-mpl-") as raw:
        environment["MPLCONFIGDIR"] = raw
        commands = [
            [
                sys.executable,
                "scripts/analyze_q1q2_statistics.py",
            ],
            [
                sys.executable,
                "scripts/analyze_q1q2_external_subgroups.py",
            ],
            [
                sys.executable,
                "scripts/analyze_q1q2_resources.py",
            ],
            [
                sys.executable,
                "scripts/generate_q1q2_figures.py",
            ],
        ]
        results = [
            _run(command, cwd=Path.cwd(), environment=environment)
            for command in commands
        ]
    if any(result["returncode"] != 0 for result in results):
        raise RuntimeError("Gate I numerical artifact reproduction failed")
    after = {path.as_posix(): file_digest(path) for path in paths}
    mismatches = [path for path in sorted(before) if before[path] != after.get(path)]
    if mismatches:
        raise RuntimeError(
            f"Gate I outputs are not byte reproducible: {mismatches[:10]}"
        )
    return {
        "status": "pass",
        "output_count": len(paths),
        "byte_identical": True,
        "commands": results,
        "qualitative_panels_rerendered": False,
        "qualitative_scope": (
            "Existing panels hash-verified; rerender requires ignored derived "
            "external-image cache"
        ),
    }


class ArtifactIndex:
    """Hash-verify artifacts while rejecting path/hash contradictions."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def add(
        self,
        path: Path,
        *,
        role: str,
        expected_sha256: str | None = None,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Required Gate I artifact is absent: {path}")
        digest = file_digest(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"Gate I artifact hash differs: {path}")
        key = path.as_posix()
        previous = self._entries.get(key)
        if previous is not None and previous["sha256"] != digest:
            raise RuntimeError(f"Contradictory hashes for Gate I artifact: {path}")
        roles = set(previous["roles"]) if previous is not None else set()
        roles.add(role)
        self._entries[key] = {
            "path": key,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "roles": sorted(roles),
        }

    def entries(self) -> list[dict[str, Any]]:
        return [self._entries[key] for key in sorted(self._entries)]


def _add_named_outputs(
    index: ArtifactIndex,
    completion: dict[str, Any],
    *,
    role: str,
) -> None:
    outputs = cast(dict[str, dict[str, str]], completion.get("outputs", {}))
    for entry in outputs.values():
        index.add(
            Path(str(entry["path"])),
            role=role,
            expected_sha256=str(entry["sha256"]),
        )


def _add_gate_g(
    index: ArtifactIndex,
    freeze_path: Path,
    manifest_path: Path,
    *,
    expected_runs: int,
) -> str:
    freeze = _load_json(freeze_path)
    manifest = _load_json(manifest_path)
    if (
        freeze.get("status") != "frozen_external_inference_permitted"
        or manifest.get("status") != "frozen"
        or int(manifest.get("run_count", -1)) != expected_runs
        or freeze.get("checkpoint_manifest_sha256") != file_digest(manifest_path)
    ):
        raise PermissionError("Passing frozen Gate G is required for Gate I")
    index.add(freeze_path, role="gate_g_freeze")
    index.add(
        manifest_path,
        role="gate_g_checkpoint_manifest",
        expected_sha256=str(freeze["checkpoint_manifest_sha256"]),
    )
    frozen_inputs = cast(dict[str, str], freeze["analysis_input_sha256"])
    for raw_path, digest in frozen_inputs.items():
        index.add(
            Path(raw_path),
            role="gate_g_frozen_analysis_input",
            expected_sha256=digest,
        )
    commits: set[str] = set()
    for raw in cast(list[dict[str, Any]], manifest["runs"]):
        commits.add(str(raw["git_commit"]))
        for path_key, hash_key, label in (
            (
                "best_checkpoint_path",
                "best_checkpoint_sha256",
                "development_best_checkpoint",
            ),
            (
                "terminal_checkpoint_path",
                "terminal_checkpoint_sha256",
                "development_terminal_checkpoint",
            ),
            (
                "patient_metrics_path",
                "patient_metrics_sha256",
                "development_patient_metrics",
            ),
            (
                "metric_summary_path",
                "metric_summary_sha256",
                "development_metric_summary",
            ),
            (
                "resource_profile_path",
                "resource_profile_sha256",
                "development_resource_profile",
            ),
        ):
            index.add(
                Path(str(raw[path_key])),
                role=label,
                expected_sha256=str(raw[hash_key]),
            )
    if len(commits) != 1:
        raise RuntimeError("Gate I requires one training source commit")
    return next(iter(commits))


def _add_gate_h(
    index: ArtifactIndex,
    completion_path: Path,
    *,
    expected_checkpoints: int,
    expected_models: int,
) -> None:
    completion = _load_json(completion_path)
    if (
        completion.get("status") != "pass"
        or completion.get("gate_h_pass") is not True
        or int(completion.get("completed_checkpoint_count", -1)) != expected_checkpoints
        or completion.get("all_model_predictions_retained") is not True
    ):
        raise PermissionError("Passing complete Gate H is required for Gate I")
    index.add(completion_path, role="gate_h_completion")
    for path_key, hash_key, role in (
        (
            "checkpoint_patient_metrics",
            "checkpoint_patient_metrics_sha256",
            "external_checkpoint_patient_metrics",
        ),
        (
            "model_patient_metrics",
            "model_patient_metrics_sha256",
            "external_model_patient_metrics",
        ),
        (
            "checkpoint_patient_timing",
            "checkpoint_patient_timing_sha256",
            "external_checkpoint_timing",
        ),
        (
            "model_inference_resources",
            "model_inference_resources_sha256",
            "external_inference_resources",
        ),
    ):
        index.add(
            Path(str(completion[path_key])),
            role=role,
            expected_sha256=str(completion[hash_key]),
        )
    manifests = cast(
        dict[str, dict[str, str]],
        completion["model_prediction_manifests"],
    )
    if len(manifests) != expected_models:
        raise RuntimeError("Gate I requires all model prediction manifests")
    for model_id, entry in manifests.items():
        manifest_path = Path(entry["path"])
        index.add(
            manifest_path,
            role="external_model_prediction_manifest",
            expected_sha256=entry["sha256"],
        )
        manifest = _load_json(manifest_path)
        if manifest.get("model_id") != model_id or manifest.get("status") != "complete":
            raise RuntimeError(f"Invalid retained model prediction: {model_id}")
        patients = cast(dict[str, dict[str, str]], manifest["patients"])
        if len(patients) != 146:
            raise RuntimeError(f"Model prediction patient count differs: {model_id}")
        for patient_id, patient in patients.items():
            index.add(
                manifest_path.parent / f"{patient_id}.npz",
                role="external_model_prediction",
                expected_sha256=patient["sha256"],
            )


def _add_downstream(
    index: ArtifactIndex,
    prerequisite_paths: dict[str, Path],
) -> None:
    for name in (
        "statistics_completion",
        "subgroup_completion",
    ):
        path = prerequisite_paths[name]
        completion = _load_json(path)
        if completion.get("status") != "complete":
            raise PermissionError(f"Incomplete Gate I prerequisite: {name}")
        index.add(path, role=name)
        _add_named_outputs(index, completion, role=f"{name}_output")

    resource_path = prerequisite_paths["resource_completion"]
    resources = _load_json(resource_path)
    if resources.get("status") != "complete":
        raise PermissionError("Incomplete Gate I prerequisite: resources")
    index.add(resource_path, role="resource_completion")
    for raw_path, digest in cast(dict[str, str], resources["artifacts"]).items():
        index.add(
            Path(raw_path),
            role="resource_analysis_output",
            expected_sha256=digest,
        )

    figure_path = prerequisite_paths["figure_completion"]
    figures = _load_json(figure_path)
    if figures.get("status") != "complete":
        raise PermissionError("Incomplete Gate I prerequisite: figures")
    index.add(figure_path, role="figure_completion")
    for outputs in cast(dict[str, dict[str, str]], figures["figures"]).values():
        for raw_path, digest in outputs.items():
            index.add(
                Path(raw_path),
                role="result_figure",
                expected_sha256=digest,
            )

    qualitative_path = prerequisite_paths["qualitative_completion"]
    qualitative = _load_json(qualitative_path)
    if qualitative.get("status") != "complete":
        raise PermissionError("Incomplete Gate I prerequisite: qualitative")
    index.add(qualitative_path, role="qualitative_completion")
    qualitative_config = _load_yaml(Path("configs/q1q2_v2/qualitative_execution.yaml"))
    qualitative_outputs = cast(dict[str, str], qualitative_config["outputs"])
    for key, hash_key in (
        ("selected_cases", "selected_cases_sha256"),
        ("patient_disagreement", "patient_disagreement_sha256"),
        ("panel_manifest", "panel_manifest_sha256"),
    ):
        index.add(
            Path(qualitative_outputs[key]),
            role=f"qualitative_{key}",
            expected_sha256=str(qualitative[hash_key]),
        )
    panel_manifest = _load_json(Path(qualitative_outputs["panel_manifest"]))
    for panel in cast(dict[str, dict[str, Any]], panel_manifest["panels"]).values():
        for format_name in ("png", "svg"):
            index.add(
                Path(str(panel[format_name])),
                role="qualitative_panel",
                expected_sha256=str(panel[f"{format_name}_sha256"]),
            )


def build_gate_i_manifest(
    config_path: Path = Path("configs/q1q2_v2/reproducibility_execution.yaml"),
) -> dict[str, Any]:
    """Verify the completed study and write its exhaustive artifact index."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_main_results":
        raise PermissionError("Gate I execution contract is not frozen")
    requirements = cast(dict[str, Any], config["requirements"])
    if requirements["require_clean_tracked_source"]:
        if _git_value("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Gate I requires a clean tracked source tree")
    index = ArtifactIndex()
    for raw_path in cast(list[str], config["frozen_inputs"]):
        index.add(Path(raw_path), role="gate_i_frozen_input")
    prerequisites = {
        key: Path(str(value))
        for key, value in cast(dict[str, str], config["prerequisites"]).items()
    }
    outputs = cast(dict[str, str], config["outputs"])
    clean_clone = run_gate_i_clean_clone_audit(
        output_path=Path(outputs["clean_clone_audit"])
    )
    reproduction = _reproduce_numerical_outputs(prerequisites)
    index.add(
        Path(outputs["clean_clone_audit"]),
        role="gate_i_clean_clone_audit",
    )
    training_commit = _add_gate_g(
        index,
        prerequisites["gate_g_analysis_freeze"],
        prerequisites["gate_g_checkpoint_manifest"],
        expected_runs=600,
    )
    _add_gate_h(
        index,
        prerequisites["gate_h_completion"],
        expected_checkpoints=300,
        expected_models=12,
    )
    _add_downstream(index, prerequisites)
    entries = index.entries()
    manifest = {
        "schema_version": 1,
        "status": "verified",
        "gate": "I",
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "training_commit": training_commit,
        "entry_count": len(entries),
        "entries": entries,
        "scope": config["scope"],
        "artifact_only_reproduction_commands": config[
            "artifact_only_reproduction_commands"
        ],
        "clean_clone_checks": config["clean_clone_checks"],
        "clean_clone_audit": clean_clone,
        "numerical_artifact_reproduction": reproduction,
        "full_retraining_claimed": False,
        "external_reinference_performed": False,
    }
    manifest_path = Path(outputs["manifest"])
    atomic_write_json(manifest_path, manifest)
    completion = {
        "schema_version": 1,
        "status": "pass",
        "gate": "I",
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": file_digest(manifest_path),
        "artifact_count": len(entries),
        "source_commit": manifest["source_commit"],
        "training_commit": training_commit,
        "clean_tracked_source": True,
        "clean_clone_audit_pass": True,
        "numerical_artifact_reproduction_byte_identical": True,
        "qualitative_panels_hash_verified": True,
        "qualitative_panels_rerendered": False,
        "full_training_reproduction_completed": False,
        "scope_boundary_explicit": True,
    }
    atomic_write_json(Path(outputs["completion"]), completion)
    return completion


def verify_gate_i_manifest(manifest_path: Path) -> dict[str, Any]:
    """Rehash a saved Gate I manifest without opening any raw dataset."""
    manifest = _load_json(manifest_path)
    mismatches: list[str] = []
    for entry in cast(list[dict[str, Any]], manifest.get("entries", [])):
        path = Path(str(entry["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["size_bytes"])
            or file_digest(path) != str(entry["sha256"])
        ):
            mismatches.append(path.as_posix())
    valid = (
        manifest.get("status") == "verified"
        and int(manifest.get("entry_count", -1))
        == len(cast(list[Any], manifest.get("entries", [])))
        and not mismatches
    )
    return {
        "valid": valid,
        "entry_count": int(manifest.get("entry_count", 0)),
        "mismatches": mismatches,
        "raw_data_opened": False,
        "external_inference_performed": False,
    }


__all__ = [
    "ArtifactIndex",
    "build_gate_i_manifest",
    "run_gate_i_clean_clone_audit",
    "verify_gate_i_manifest",
]
