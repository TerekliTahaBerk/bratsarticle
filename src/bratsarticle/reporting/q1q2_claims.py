"""Gate J artifact-bound claim registry and manuscript token renderer."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

TOKEN_PATTERN = re.compile(
    r"\{\{claim:([A-Z0-9_.-]+)\|(raw|integer|2f|3f|4f|percent1|percent2|pvalue)\}\}"
)
UNRESOLVED_PATTERN = re.compile(r"\{\{[^{}]+\}\}")
STANDALONE_NUMBER_PATTERN = re.compile(
    r"(?<![\w])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![\w])"
)
NEGATION_PATTERN = re.compile(
    r"\b(no|not|cannot|without|unknown|pending|prohibited|does not|"
    r"did not|is not|are not)\b",
    re.IGNORECASE,
)


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


def _identifier(value: Any) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")
    return normalized or "EMPTY"


def _native(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _require_output(
    completion: dict[str, Any],
    *,
    key: str,
    path: Path,
) -> None:
    entry = cast(dict[str, str], cast(dict[str, Any], completion["outputs"])[key])
    if (
        entry.get("path") != path.as_posix()
        or not path.is_file()
        or entry.get("sha256") != file_digest(path)
    ):
        raise RuntimeError(f"Gate J source differs from completion: {path}")


def _require_resource(
    completion: dict[str, Any],
    *,
    path: Path,
) -> None:
    artifacts = cast(dict[str, str], completion["artifacts"])
    if not path.is_file() or artifacts.get(path.as_posix()) != file_digest(path):
        raise RuntimeError(f"Gate J resource source differs: {path}")


class ClaimRegistry:
    """Accumulate unique scalar claims with exact source-cell provenance."""

    def __init__(self) -> None:
        self._claims: dict[str, dict[str, Any]] = {}

    def add(
        self,
        claim_id: str,
        value: Any,
        *,
        source_path: Path,
        selector: dict[str, Any],
        column: str,
        inferential_role: str,
    ) -> None:
        if claim_id in self._claims:
            raise RuntimeError(f"Duplicate Gate J claim identifier: {claim_id}")
        native = _native(value)
        value_status = "available"
        if native is None or (isinstance(native, float) and not math.isfinite(native)):
            value_status = "nonfinite_or_missing"
            native = None
        self._claims[claim_id] = {
            "claim_id": claim_id,
            "value": native,
            "value_status": value_status,
            "source": {
                "path": source_path.as_posix(),
                "sha256": file_digest(source_path),
                "selector": selector,
                "column": column,
            },
            "inferential_role": inferential_role,
        }

    def claims(self) -> list[dict[str, Any]]:
        return [self._claims[key] for key in sorted(self._claims)]


def _add_frame(
    registry: ClaimRegistry,
    frame: pd.DataFrame,
    *,
    path: Path,
    prefix: str,
    identity_columns: tuple[str, ...],
    inferential_role: str,
) -> None:
    missing = set(identity_columns).difference(frame.columns)
    if missing:
        raise RuntimeError(f"Claim source lacks identities: {path}: {sorted(missing)}")
    if frame.duplicated(list(identity_columns)).any():
        raise RuntimeError(f"Claim source has duplicate identities: {path}")
    ordered = frame.sort_values(list(identity_columns)).reset_index(drop=True)
    for _, row in ordered.iterrows():
        selector = {column: _native(row[column]) for column in identity_columns}
        identity = ".".join(
            _identifier(selector[column]) for column in identity_columns
        )
        for column in ordered.columns:
            if column in identity_columns:
                continue
            registry.add(
                f"{prefix}.{identity}.{_identifier(column)}",
                row[column],
                source_path=path,
                selector=selector,
                column=column,
                inferential_role=inferential_role,
            )


def build_claim_registry(
    config_path: Path = Path("configs/q1q2_v2/claim_execution.yaml"),
) -> dict[str, Any]:
    """Build all reportable scalar values from hash-verified result artifacts."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_main_results":
        raise PermissionError("Gate J execution contract is not frozen")
    sources = {
        key: Path(str(value))
        for key, value in cast(dict[str, str], config["sources"]).items()
    }
    statistics = _load_json(sources["statistical_completion"])
    subgroups = _load_json(sources["subgroup_completion"])
    resources = _load_json(sources["resource_completion"])
    qualitative = _load_json(sources["qualitative_completion"])
    if any(
        completion.get("status") != "complete"
        for completion in (statistics, subgroups, resources, qualitative)
    ):
        raise PermissionError("All Gate J result analyses must be complete")
    _require_output(
        statistics,
        key="primary_contrasts",
        path=sources["primary_contrasts"],
    )
    _require_output(
        statistics,
        key="model_metric_summary",
        path=sources["model_metric_summary"],
    )
    _require_output(
        subgroups,
        key="contrast_subgroup_summary",
        path=sources["contrast_subgroup_summary"],
    )
    _require_resource(resources, path=sources["accuracy_cost_pareto"])
    selected_cases_hash = str(qualitative["selected_cases_sha256"])
    if (
        not sources["qualitative_selected_cases"].is_file()
        or file_digest(sources["qualitative_selected_cases"]) != selected_cases_hash
    ):
        raise RuntimeError("Gate J qualitative selection source differs")

    registry = ClaimRegistry()
    selected_loss = _load_yaml(sources["selected_loss"])
    if (
        selected_loss.get("status") != "frozen_from_complete_development_cv"
        or selected_loss.get("external_data_used_for_selection") is not False
        or selected_loss.get("legacy_internal_test_used_for_selection") is not False
    ):
        raise PermissionError("Gate J requires the development-only loss freeze")
    registry.add(
        "METHOD.SELECTED_LOSS",
        selected_loss["selected_loss"],
        source_path=sources["selected_loss"],
        selector={"section": "root"},
        column="selected_loss",
        inferential_role="development_only_model_selection",
    )
    training_protocol = _load_yaml(sources["training_protocol"])
    training = cast(dict[str, Any], training_protocol["training"])
    convergence = cast(dict[str, Any], training_protocol["convergence_matched"])
    early_stopping = cast(dict[str, Any], convergence["early_stopping"])
    compute = cast(dict[str, Any], training_protocol["compute_matched"])
    method_values = {
        "INITIAL_LEARNING_RATE_NATIVE_2D": cast(
            dict[str, Any], training["initial_learning_rate"]
        )["native_2d"],
        "WEIGHT_DECAY": training["weight_decay"],
        "EFFECTIVE_BATCH_SIZE_NATIVE_2D": cast(
            dict[str, Any], training["effective_batch_size"]
        )["native_2d"],
        "VALIDATION_FREQUENCY_OPTIMIZER_STEPS": training[
            "validation_frequency_optimizer_steps"
        ],
        "MAXIMUM_OPTIMIZER_STEPS": convergence["maximum_optimizer_steps"],
        "MINIMUM_OPTIMIZER_STEPS_BEFORE_EARLY_STOPPING": convergence[
            "minimum_optimizer_steps_before_early_stopping"
        ],
        "EARLY_STOPPING_MINIMUM_DELTA": early_stopping["minimum_delta"],
        "EARLY_STOPPING_PATIENCE_CHECKS": early_stopping["patience_validation_checks"],
        "COMPUTE_MATCHED_HOURS_PER_RUN": compute["maximum_accelerator_hours_per_run"],
    }
    for name, value in method_values.items():
        registry.add(
            f"METHOD.{name}",
            value,
            source_path=sources["training_protocol"],
            selector={"section": "training_or_budget"},
            column=name.lower(),
            inferential_role="frozen_method_fact",
        )
    plan = _load_yaml(sources["statistical_analysis_plan"])
    estimation = cast(dict[str, Any], plan["estimation"])
    bootstrap = cast(dict[str, Any], estimation["paired_patient_bootstrap"])
    permutation = cast(dict[str, Any], estimation["paired_permutation"])
    multiplicity = cast(dict[str, Any], plan["multiplicity"])
    practical = cast(dict[str, Any], plan["practical_interpretation"])
    statistical_values = {
        "CONFIDENCE_LEVEL": bootstrap["confidence_level"],
        "PAIRED_BOOTSTRAP_RESAMPLES": bootstrap["resamples"],
        "PAIRED_PERMUTATION_RESAMPLES": permutation["resamples"],
        "MULTIPLICITY_ALPHA": multiplicity["alpha_two_sided"],
        "PRACTICAL_THRESHOLD": practical["smallest_effect_size_of_interest"],
    }
    for name, value in statistical_values.items():
        registry.add(
            f"METHOD.{name}",
            value,
            source_path=sources["statistical_analysis_plan"],
            selector={"section": "estimation_or_interpretation"},
            column=name.lower(),
            inferential_role="frozen_statistical_method_fact",
        )
    figure_execution = _load_yaml(sources["figure_execution"])
    design = cast(dict[str, Any], figure_execution["design"])
    for field, value in sorted(design.items()):
        registry.add(
            f"DESIGN.{_identifier(field)}",
            value,
            source_path=sources["figure_execution"],
            selector={"section": "design"},
            column=field,
            inferential_role="design_fact",
        )
    contrast_frame = pd.read_csv(sources["primary_contrasts"])
    _add_frame(
        registry,
        contrast_frame,
        path=sources["primary_contrasts"],
        prefix="CONTRAST",
        identity_columns=("contrast_id",),
        inferential_role="confirmatory_prespecified_contrast",
    )
    interpretation_text = {
        "positive_and_practically_relevant": (
            "The difference met both the multiplicity-adjusted statistical "
            "criterion and the prespecified practical threshold."
        ),
        "positive_but_below_practical_threshold": (
            "The adjusted statistical criterion was met, but the mean "
            "difference remained below the prespecified practical threshold."
        ),
        "no_confirmatory_superiority": (
            "The confirmatory comparison did not establish a "
            "capacity-controlled benefit."
        ),
    }
    for _, row in contrast_frame.iterrows():
        contrast_id = str(row["contrast_id"])
        interpretation = str(row["claim_interpretation"])
        if interpretation not in interpretation_text:
            raise RuntimeError(f"Unknown confirmatory interpretation: {interpretation}")
        registry.add(
            f"CONTRAST.{_identifier(contrast_id)}.INTERPRETATION_TEXT",
            interpretation_text[interpretation],
            source_path=sources["primary_contrasts"],
            selector={"contrast_id": contrast_id},
            column="claim_interpretation",
            inferential_role="confirmatory_prespecified_contrast",
        )
    _add_frame(
        registry,
        pd.read_csv(sources["model_metric_summary"]),
        path=sources["model_metric_summary"],
        prefix="METRIC",
        identity_columns=("cohort", "model_id", "endpoint"),
        inferential_role="cohort_model_estimate",
    )
    _add_frame(
        registry,
        pd.read_csv(sources["accuracy_cost_pareto"]),
        path=sources["accuracy_cost_pareto"],
        prefix="RESOURCE",
        identity_columns=("model_id",),
        inferential_role="measured_accuracy_resource_tradeoff",
    )
    _add_frame(
        registry,
        pd.read_csv(sources["contrast_subgroup_summary"]),
        path=sources["contrast_subgroup_summary"],
        prefix="SUBGROUP",
        identity_columns=("dimension", "category", "contrast_id"),
        inferential_role="exploratory_estimation_only",
    )
    selected = _load_json(sources["qualitative_selected_cases"])
    for rule, entry in sorted(
        cast(dict[str, dict[str, Any]], selected["rules"]).items()
    ):
        for column, value in sorted(entry.items()):
            registry.add(
                f"QUALITATIVE.{_identifier(rule)}.{_identifier(column)}",
                value,
                source_path=sources["qualitative_selected_cases"],
                selector={"rule": rule},
                column=column,
                inferential_role="prespecified_post_evaluation_case_selection",
            )
    claims = registry.claims()
    source_hashes = {
        path.as_posix(): file_digest(path)
        for path in sorted(set(sources.values()), key=lambda item: item.as_posix())
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "gate": "J",
        "manual_result_entry": False,
        "claim_count": len(claims),
        "claims": claims,
        "source_hashes": source_hashes,
    }
    outputs = cast(dict[str, str], config["outputs"])
    output_path = Path(outputs["registry"])
    atomic_write_json(output_path, payload)
    return payload


def _artifact_bound_sections(source: str, *, start: str, end: str) -> list[str]:
    sections: list[str] = []
    cursor = 0
    while True:
        begin = source.find(start, cursor)
        if begin < 0:
            break
        finish = source.find(end, begin + len(start))
        if finish < 0:
            raise RuntimeError("Artifact-bound result block is not closed")
        sections.append(source[begin + len(start) : finish])
        cursor = finish + len(end)
    return sections


def audit_claim_template(
    template_path: Path,
    *,
    config_path: Path = Path("configs/q1q2_v2/claim_execution.yaml"),
) -> dict[str, Any]:
    """Reject malformed tokens and manual numbers in bounded Results blocks."""
    config = _load_yaml(config_path)
    contract = cast(dict[str, Any], config["template_contract"])
    source = template_path.read_text(encoding="utf-8")
    sections = _artifact_bound_sections(
        source,
        start=str(contract["artifact_bound_start"]),
        end=str(contract["artifact_bound_end"]),
    )
    if not sections:
        raise RuntimeError("Manuscript template has no artifact-bound result block")
    raw_tokens = UNRESOLVED_PATTERN.findall(source)
    malformed = [token for token in raw_tokens if not TOKEN_PATTERN.fullmatch(token)]
    if malformed:
        raise RuntimeError(f"Malformed Gate J claim tokens: {malformed[:5]}")
    manual_numbers: list[str] = []
    for section in sections:
        without_tokens = TOKEN_PATTERN.sub("", section)
        manual_numbers.extend(STANDALONE_NUMBER_PATTERN.findall(without_tokens))
    if manual_numbers:
        raise RuntimeError(
            "Manual numeric literals occur in artifact-bound results: "
            f"{manual_numbers[:5]}"
        )
    if not raw_tokens:
        raise RuntimeError("Manuscript template has no Gate J claim tokens")
    return {
        "valid": True,
        "artifact_bound_section_count": len(sections),
        "claim_token_count": len(raw_tokens),
        "manual_numeric_literal_count": 0,
    }


def audit_reviewer_response_template(
    template_path: Path,
    *,
    response_config_path: Path = Path(
        "configs/q1q2_v2/reviewer_response_execution.yaml"
    ),
    claim_config_path: Path = Path("configs/q1q2_v2/claim_execution.yaml"),
) -> dict[str, Any]:
    """Require complete one-to-one reviewer-concern coverage and claim binding."""
    config = _load_yaml(response_config_path)
    if config.get("status") != "frozen_before_main_results":
        raise PermissionError("Reviewer-response execution contract is not frozen")
    source = template_path.read_text(encoding="utf-8")
    concern_entries = cast(list[dict[str, Any]], config["concerns"])
    expected_ids = [str(entry["id"]) for entry in concern_entries]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Reviewer-response concern identifiers are duplicated")
    observed_ids = re.findall(r"^## (R\d{2})\s+—", source, flags=re.MULTILINE)
    missing = sorted(set(expected_ids).difference(observed_ids))
    unexpected = sorted(set(observed_ids).difference(expected_ids))
    duplicates = sorted(
        concern_id
        for concern_id in set(observed_ids)
        if observed_ids.count(concern_id) != 1
    )
    if missing or unexpected or duplicates:
        raise RuntimeError(
            "Reviewer-response concern coverage differs: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    required_fields = [
        str(value) for value in cast(list[str], config["required_fields"])
    ]
    sections = re.split(r"^## R\d{2}\s+—.*$", source, flags=re.MULTILINE)[1:]
    for concern_id, section in zip(observed_ids, sections, strict=True):
        absent_fields = [
            field for field in required_fields if f"**{field}.**" not in section
        ]
        if absent_fields:
            raise RuntimeError(
                f"Reviewer response {concern_id} lacks fields: {absent_fields}"
            )
    missing_evidence: list[str] = []
    for entry in concern_entries:
        for raw_path in cast(list[str], entry["evidence"]):
            if not Path(raw_path).exists():
                missing_evidence.append(raw_path)
    if missing_evidence:
        raise RuntimeError(
            "Reviewer-response evidence paths do not exist: "
            f"{sorted(set(missing_evidence))}"
        )
    claim_audit = audit_claim_template(
        template_path,
        config_path=claim_config_path,
    )
    return {
        "valid": True,
        "concern_count": len(expected_ids),
        "required_field_count": len(required_fields),
        "all_evidence_paths_exist": True,
        "claim_template_audit": claim_audit,
    }


def _format_claim(claim: dict[str, Any], format_name: str) -> str:
    if claim["value_status"] != "available":
        raise RuntimeError(f"Cannot render nonfinite claim: {claim['claim_id']}")
    value = claim["value"]
    if format_name == "raw":
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)
    if format_name == "integer":
        return str(int(cast(float, value)))
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"Cannot render nonfinite claim: {claim['claim_id']}")
    if format_name in {"2f", "3f", "4f"}:
        return f"{numeric:.{int(format_name[0])}f}"
    if format_name in {"percent1", "percent2"}:
        digits = int(format_name[-1])
        return f"{100.0 * numeric:.{digits}f}%"
    if format_name == "pvalue":
        return "<0.0001" if numeric < 0.0001 else f"{numeric:.4f}"
    raise ValueError(f"Unknown Gate J claim format: {format_name}")


def render_claim_template(
    *,
    template_path: Path,
    registry_path: Path,
    output_path: Path,
    trace_path: Path,
    config_path: Path = Path("configs/q1q2_v2/claim_execution.yaml"),
) -> dict[str, Any]:
    """Resolve only registered values and record every template substitution."""
    audit_claim_template(template_path, config_path=config_path)
    source = template_path.read_text(encoding="utf-8")
    registry = _load_json(registry_path)
    if registry.get("status") != "complete":
        raise PermissionError("Complete Gate J claim registry is required")
    by_id = {
        str(entry["claim_id"]): entry
        for entry in cast(list[dict[str, Any]], registry["claims"])
    }
    trace: list[dict[str, Any]] = []

    def replacement(match: re.Match[str]) -> str:
        claim_id, format_name = match.groups()
        if claim_id not in by_id:
            raise RuntimeError(f"Unknown Gate J claim token: {claim_id}")
        rendered = _format_claim(by_id[claim_id], format_name)
        trace.append(
            {
                "claim_id": claim_id,
                "format": format_name,
                "rendered": rendered,
                "source": by_id[claim_id]["source"],
            }
        )
        return rendered

    rendered = TOKEN_PATTERN.sub(replacement, source)
    unresolved = UNRESOLVED_PATTERN.findall(rendered)
    if unresolved:
        raise RuntimeError(f"Unresolved manuscript tokens: {unresolved[:5]}")
    if not trace:
        raise RuntimeError("Manuscript template did not use any registered claim")
    atomic_write_text(output_path, rendered)
    trace_payload = {
        "schema_version": 1,
        "status": "complete",
        "template": template_path.as_posix(),
        "template_sha256": file_digest(template_path),
        "registry": registry_path.as_posix(),
        "registry_sha256": file_digest(registry_path),
        "rendered": output_path.as_posix(),
        "rendered_sha256": file_digest(output_path),
        "resolved_token_count": len(trace),
        "unique_claim_count": len({entry["claim_id"] for entry in trace}),
        "manual_result_entry": False,
        "substitutions": trace,
    }
    atomic_write_json(trace_path, trace_payload)
    return trace_payload


def audit_claim_package(
    *,
    registry_path: Path,
    rendered_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    """Verify claim sources and the exact rendered manuscript hash."""
    registry = _load_json(registry_path)
    trace = _load_json(trace_path)
    mismatches: list[str] = []
    for raw_path, digest in cast(
        dict[str, str],
        registry.get("source_hashes", {}),
    ).items():
        path = Path(raw_path)
        if not path.is_file() or file_digest(path) != digest:
            mismatches.append(raw_path)
    valid = (
        registry.get("status") == "complete"
        and trace.get("status") == "complete"
        and trace.get("registry_sha256") == file_digest(registry_path)
        and rendered_path.is_file()
        and trace.get("rendered_sha256") == file_digest(rendered_path)
        and int(trace.get("resolved_token_count", 0)) > 0
        and not UNRESOLVED_PATTERN.search(rendered_path.read_text(encoding="utf-8"))
        and not mismatches
    )
    return {
        "valid": valid,
        "claim_count": int(registry.get("claim_count", 0)),
        "resolved_token_count": int(trace.get("resolved_token_count", 0)),
        "source_hash_mismatches": mismatches,
        "manual_result_entry": False,
    }


def _audit_inferential_language(
    rendered_text: str,
    trace: dict[str, Any],
) -> dict[str, Any]:
    lowered = rendered_text.lower()
    substitutions = cast(
        list[dict[str, Any]],
        trace.get("substitutions", []),
    )
    used = {
        str(entry["claim_id"]): str(entry["rendered"]).lower()
        for entry in substitutions
    }
    holm_positive = any(
        claim_id.endswith(".HOLM_REJECT_AT_ALPHA") and value == "true"
        for claim_id, value in used.items()
    )
    practical_positive = any(
        claim_id.endswith(".MEAN_REACHES_PRACTICAL_THRESHOLD") and value == "true"
        for claim_id, value in used.items()
    )
    positive_interpretation = any(
        claim_id.endswith(".CLAIM_INTERPRETATION")
        and value == "positive_and_practically_relevant"
        for claim_id, value in used.items()
    )
    problems: list[str] = []
    affirmative_superiority = any(
        ("superior" in line.lower() or "superiority" in line.lower())
        and "probability of superiority" not in line.lower()
        and not NEGATION_PATTERN.search(line)
        for line in rendered_text.splitlines()
    )
    if affirmative_superiority and not (
        holm_positive and practical_positive and positive_interpretation
    ):
        problems.append(
            "Superiority wording lacks rendered Holm, practical-threshold, "
            "and positive-interpretation bindings"
        )
    if "significant" in lowered and not holm_positive:
        problems.append("Significance wording lacks a rendered positive Holm binding")
    prohibited_affirmative = (
        "state of the art",
        "clinically robust",
        "clinically validated",
        "clinical utility",
        "generalizable",
        "q1/q2-ready",
    )
    for phrase in prohibited_affirmative:
        for line in rendered_text.splitlines():
            if phrase in line.lower() and not NEGATION_PATTERN.search(line):
                problems.append(f"Unsupported affirmative wording: {phrase}")
    return {
        "valid": not problems,
        "problems": problems,
        "holm_positive_binding_used": holm_positive,
        "practical_positive_binding_used": practical_positive,
        "positive_interpretation_binding_used": positive_interpretation,
    }


def complete_gate_j(
    config_path: Path = Path("configs/q1q2_v2/claim_execution.yaml"),
) -> dict[str, Any]:
    """Close Gate J only when provenance and inferential wording both pass."""
    config = _load_yaml(config_path)
    outputs = {
        key: Path(str(value))
        for key, value in cast(dict[str, str], config["outputs"]).items()
    }
    artifact_audit = audit_claim_package(
        registry_path=outputs["registry"],
        rendered_path=outputs["rendered_manuscript"],
        trace_path=outputs["render_trace"],
    )
    response_artifact_audit = audit_claim_package(
        registry_path=outputs["registry"],
        rendered_path=outputs["rendered_reviewer_response"],
        trace_path=outputs["reviewer_response_trace"],
    )
    trace = _load_json(outputs["render_trace"])
    wording_audit = _audit_inferential_language(
        outputs["rendered_manuscript"].read_text(encoding="utf-8"),
        trace,
    )
    response_trace = _load_json(outputs["reviewer_response_trace"])
    response_wording_audit = _audit_inferential_language(
        outputs["rendered_reviewer_response"].read_text(encoding="utf-8"),
        response_trace,
    )
    passed = (
        artifact_audit["valid"]
        and response_artifact_audit["valid"]
        and wording_audit["valid"]
        and response_wording_audit["valid"]
    )
    completion = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "gate": "J",
        "manual_result_entry": False,
        "registry": outputs["registry"].as_posix(),
        "registry_sha256": file_digest(outputs["registry"]),
        "rendered_manuscript": outputs["rendered_manuscript"].as_posix(),
        "rendered_manuscript_sha256": file_digest(outputs["rendered_manuscript"]),
        "render_trace": outputs["render_trace"].as_posix(),
        "render_trace_sha256": file_digest(outputs["render_trace"]),
        "rendered_reviewer_response": outputs[
            "rendered_reviewer_response"
        ].as_posix(),
        "rendered_reviewer_response_sha256": file_digest(
            outputs["rendered_reviewer_response"]
        ),
        "reviewer_response_trace": outputs["reviewer_response_trace"].as_posix(),
        "reviewer_response_trace_sha256": file_digest(
            outputs["reviewer_response_trace"]
        ),
        "artifact_audit": artifact_audit,
        "response_artifact_audit": response_artifact_audit,
        "inferential_wording_audit": wording_audit,
        "response_inferential_wording_audit": response_wording_audit,
    }
    atomic_write_json(outputs["completion"], completion)
    if not passed:
        raise RuntimeError("Gate J claim provenance or wording audit failed")
    return completion


__all__ = [
    "ClaimRegistry",
    "audit_claim_package",
    "audit_claim_template",
    "audit_reviewer_response_template",
    "build_claim_registry",
    "complete_gate_j",
    "render_claim_template",
]
