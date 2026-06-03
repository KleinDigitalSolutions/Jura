"""Kanzlei eval-set runner and deterministic regression checks.

Eval cases define expected legal sources and audit constraints. They do not
define canned answers; the live pipeline remains retrieval and generation based.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 180


@dataclass
class EvalFailure:
    check: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


@dataclass
class EvalCaseResult:
    case_id: str
    group: str
    topic: str
    blocking: bool
    passed: bool
    latency_seconds: float = 0.0
    failures: list[EvalFailure] = field(default_factory=list)
    audit_status: str = ""
    audit_score: int | None = None
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "group": self.group,
            "topic": self.topic,
            "blocking": self.blocking,
            "passed": self.passed,
            "latency_seconds": round(self.latency_seconds, 3),
            "failures": [failure.as_dict() for failure in self.failures],
            "audit_status": self.audit_status,
            "audit_score": self.audit_score,
            "model": self.model,
        }


def load_eval_set(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an eval-set JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Eval set must contain a non-empty 'cases' list")
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError("Each eval case needs an 'id'")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate eval case id: {case_id}")
        seen_ids.add(case_id)
        if not case.get("query"):
            raise ValueError(f"Eval case {case_id} needs a query")
        for key in ("must_include", "must_not_include", "expected_profiles"):
            if key in case and not isinstance(case[key], list):
                raise ValueError(f"Eval case {case_id} field {key} must be a list")
    return data


def normalize_norm_label(value: str) -> str:
    """Normalize labels such as 'BGB §623' and 'bgb § 623'."""
    text = " ".join((value or "").replace("§§", "§").split())
    if not text:
        return ""
    parts = text.split()
    if len(parts) >= 2 and parts[1].startswith("§"):
        law = parts[0].upper().replace("GMBH", "GMBHG")
        paragraph = parts[1]
        number = parts[2] if len(parts) >= 3 and paragraph == "§" else paragraph.lstrip("§")
        return f"{law} § {number}"
    return text.upper()


def _citation_labels(response: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for citation in response.get("citations") or []:
        law = citation.get("gesetz") or citation.get("abkürzung") or ""
        paragraph = citation.get("paragraph") or ""
        if law and paragraph:
            labels.add(normalize_norm_label(f"{law} {paragraph}"))
    return labels


def _answer_mentions_norm(response: dict[str, Any], norm_label: str) -> bool:
    answer = (response.get("answer") or "").upper()
    normalized = normalize_norm_label(norm_label)
    if not normalized:
        return False
    law, _, paragraph = normalized.partition(" § ")
    return law in answer and f"§ {paragraph}".upper() in answer


def _is_blocking_case(case: dict[str, Any]) -> bool:
    if "blocking" in case:
        return bool(case["blocking"])
    return case.get("group") != "known_gap"


def evaluate_response(case: dict[str, Any], response: dict[str, Any], latency_seconds: float = 0.0) -> EvalCaseResult:
    """Evaluate one enhanced-answer API response against one eval case."""
    failures: list[EvalFailure] = []
    citation_labels = _citation_labels(response)
    retrieval_plan = response.get("retrieval_plan") or {}
    source_audit = response.get("source_audit") or {}
    answer_audit = response.get("answer_audit") or {}

    for expected in case.get("must_include") or []:
        label = normalize_norm_label(expected)
        if label not in citation_labels:
            failures.append(EvalFailure("missing_must_include", f"{expected} not present in citations"))

    for excluded in case.get("must_not_include") or []:
        label = normalize_norm_label(excluded)
        if label in citation_labels:
            failures.append(EvalFailure("must_not_include_in_citations", f"{excluded} appeared in citations"))
        if _answer_mentions_norm(response, excluded):
            failures.append(EvalFailure("must_not_include_in_answer", f"{excluded} appeared in answer text"))

    actual_profiles = set(retrieval_plan.get("profiles") or [])
    for profile in case.get("expected_profiles") or []:
        if profile not in actual_profiles:
            failures.append(EvalFailure("missing_expected_profile", f"{profile} not present in retrieval_plan.profiles"))

    missing_required = set(source_audit.get("missing_required") or [])
    for expected in case.get("must_include") or []:
        if expected in missing_required:
            failures.append(EvalFailure("source_missing_required", f"{expected} is listed in source_audit.missing_required"))

    if not answer_audit:
        failures.append(EvalFailure("missing_answer_audit", "Enhanced answer response did not include answer_audit"))
    else:
        max_high = int(case.get("max_high_audit_issues", 0))
        high_count = int(answer_audit.get("high_severity_count", 0) or 0)
        if high_count > max_high:
            failures.append(EvalFailure("too_many_high_audit_issues", f"{high_count} high issues > allowed {max_high}"))
        if answer_audit.get("status") == "error":
            failures.append(EvalFailure("answer_audit_error", "answer_audit.status is error"))

    return EvalCaseResult(
        case_id=case["id"],
        group=case.get("group", "regression_guard"),
        topic=case.get("topic", ""),
        blocking=_is_blocking_case(case),
        passed=not failures,
        latency_seconds=latency_seconds,
        failures=failures,
        audit_status=str(answer_audit.get("status", "")),
        audit_score=answer_audit.get("score"),
        model=str(response.get("model", "")),
    )


def call_enhanced_endpoint(endpoint: str, query: str, top_k: int = 8, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Call the enhanced-answer endpoint using only the standard library."""
    params = urllib.parse.urlencode({"q": query, "top_k": top_k})
    separator = "&" if "?" in endpoint else "?"
    url = f"{endpoint}{separator}{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def run_eval_set(
    eval_set: dict[str, Any],
    endpoint: str | None = None,
    top_k: int = 8,
    include_known_gaps: bool = True,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Run an eval set against the enhanced-answer endpoint."""
    endpoint = endpoint or eval_set.get("default_endpoint")
    if not endpoint:
        raise ValueError("No endpoint provided and eval set has no default_endpoint")

    cases = list(eval_set["cases"])
    if case_id:
        cases = [case for case in cases if case.get("id") == case_id]
        if not cases:
            raise ValueError(f"Eval case not found: {case_id}")

    results: list[EvalCaseResult] = []
    for case in cases:
        if case.get("group") == "known_gap" and not include_known_gaps:
            continue
        started = time.monotonic()
        try:
            response = call_enhanced_endpoint(endpoint, case["query"], top_k=top_k, timeout=timeout)
            latency = time.monotonic() - started
            results.append(evaluate_response(case, response, latency_seconds=latency))
        except Exception as exc:
            latency = time.monotonic() - started
            results.append(
                EvalCaseResult(
                    case_id=case["id"],
                    group=case.get("group", "regression_guard"),
                    topic=case.get("topic", ""),
                    blocking=_is_blocking_case(case),
                    passed=False,
                    latency_seconds=latency,
                    failures=[EvalFailure("request_failed", str(exc))],
                )
            )

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    blocking_failures = [result for result in results if result.blocking and not result.passed]
    known_gap_failures = [result for result in results if not result.blocking and not result.passed]
    return {
        "name": eval_set.get("name", ""),
        "version": eval_set.get("version"),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "blocking_failed": len(blocking_failures),
        "known_gap_failed": len(known_gap_failures),
        "results": [result.as_dict() for result in results],
    }


def format_eval_report(summary: dict[str, Any]) -> str:
    """Render a compact human-readable report."""
    lines = [
        f"Eval set: {summary.get('name')} v{summary.get('version')}",
        f"Cases: {summary['total']} | passed: {summary['passed']} | failed: {summary['failed']} | blocking failed: {summary['blocking_failed']}",
        "",
    ]
    for result in summary["results"]:
        marker = "PASS" if result["passed"] else ("FAIL" if result["blocking"] else "GAP")
        lines.append(
            f"{marker} {result['case_id']} [{result['group']}] "
            f"audit={result.get('audit_status') or '-'} score={result.get('audit_score')} "
            f"latency={result['latency_seconds']:.2f}s"
        )
        for failure in result["failures"]:
            lines.append(f"  - {failure['check']}: {failure['detail']}")
    return "\n".join(lines)
