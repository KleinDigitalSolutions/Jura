"""Tests for Kanzlei eval-set regression checks."""

from pathlib import Path

import pytest

from src.retrieval import eval_runner
from src.retrieval.eval_runner import evaluate_response, format_eval_report, load_eval_set, normalize_norm_label, run_eval_set


def _case(**overrides):
    case = {
        "id": "arbeitsrecht_001",
        "group": "regression_guard",
        "topic": "Ordentliche Kündigung",
        "query": "Welche Anforderungen gelten für eine ordentliche Kündigung?",
        "must_include": ["BGB § 623", "KSchG § 4"],
        "must_not_include": ["BGB § 580a"],
        "expected_profiles": ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"],
        "max_high_audit_issues": 0,
    }
    case.update(overrides)
    return case


def _response(**overrides):
    response = {
        "answer": "Die Kündigung muss schriftlich erfolgen [1]. Die Klagefrist beträgt drei Wochen [2].",
        "citations": [
            {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623"},
            {"id": "[2]", "gesetz": "KSchG", "paragraph": "§ 4"},
        ],
        "retrieval_plan": {
            "profiles": ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"],
            "required_norms": ["BGB § 623", "KSchG § 4"],
        },
        "source_audit": {"missing_required": []},
        "answer_audit": {
            "status": "pass",
            "score": 100,
            "high_severity_count": 0,
            "issue_count": 0,
        },
        "model": "test-model",
    }
    response.update(overrides)
    return response


def test_normalize_norm_label_handles_spacing_and_case():
    assert normalize_norm_label("bgb §623") == "BGB § 623"
    assert normalize_norm_label("KSchG § 4") == "KSCHG § 4"


def test_load_kanzlei_core_eval_set_has_unique_ids():
    data = load_eval_set(Path(__file__).resolve().parents[1] / "evals" / "kanzlei_core.json")

    assert data["name"] == "kanzlei_core"
    assert len(data["cases"]) == 15
    assert len({case["id"] for case in data["cases"]}) == 15


def test_evaluate_response_passes_when_sources_profiles_and_audit_match():
    result = evaluate_response(_case(), _response(), latency_seconds=1.25)

    assert result.passed
    assert result.blocking
    assert result.failures == []
    assert result.audit_status == "pass"
    assert result.audit_score == 100
    assert result.latency_seconds == 1.25


def test_evaluate_response_fails_on_missing_must_include():
    result = evaluate_response(
        _case(),
        _response(citations=[{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623"}]),
    )

    assert not result.passed
    assert result.blocking
    assert "missing_must_include" in {failure.check for failure in result.failures}


def test_evaluate_response_fails_on_excluded_source_in_context_or_answer():
    result = evaluate_response(
        _case(),
        _response(
            answer="Die mietrechtliche Kündigungsfrist aus § 580a BGB ist relevant [3].",
            citations=[
                {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623"},
                {"id": "[2]", "gesetz": "KSchG", "paragraph": "§ 4"},
                {"id": "[3]", "gesetz": "BGB", "paragraph": "§ 580a"},
            ],
        ),
    )

    checks = {failure.check for failure in result.failures}
    assert "must_not_include_in_citations" in checks
    assert "must_not_include_in_answer" in checks


def test_known_gap_failures_are_non_blocking():
    result = evaluate_response(
        _case(group="known_gap"),
        _response(answer_audit={"status": "fail", "score": 0, "high_severity_count": 3}),
    )

    assert not result.passed
    assert not result.blocking


def test_format_eval_report_marks_known_gaps_separately():
    result = evaluate_response(
        _case(group="known_gap"),
        _response(answer_audit={"status": "fail", "score": 0, "high_severity_count": 1}),
    )
    summary = {
        "name": "kanzlei_core",
        "version": 1,
        "total": 1,
        "passed": 0,
        "failed": 1,
        "blocking_failed": 0,
        "results": [result.as_dict()],
    }

    report = format_eval_report(summary)

    assert "GAP arbeitsrecht_001" in report
    assert "too_many_high_audit_issues" in report


def test_run_eval_set_filters_single_case_before_endpoint_call(monkeypatch):
    calls = []

    def fake_call(endpoint, query, top_k=8, timeout=180):
        calls.append(query)
        return _response()

    monkeypatch.setattr(eval_runner, "call_enhanced_endpoint", fake_call)
    eval_set = {
        "name": "unit",
        "version": 1,
        "default_endpoint": "https://example.test/api",
        "cases": [
            _case(id="case_a", query="Frage A"),
            _case(id="case_b", query="Frage B"),
        ],
    }

    summary = run_eval_set(eval_set, case_id="case_b")

    assert calls == ["Frage B"]
    assert summary["total"] == 1
    assert summary["results"][0]["case_id"] == "case_b"


def test_run_eval_set_rejects_unknown_single_case_before_endpoint_call(monkeypatch):
    def fail_call(*args, **kwargs):
        raise AssertionError("endpoint should not be called")

    monkeypatch.setattr(eval_runner, "call_enhanced_endpoint", fail_call)
    eval_set = {
        "name": "unit",
        "version": 1,
        "default_endpoint": "https://example.test/api",
        "cases": [_case(id="case_a", query="Frage A")],
    }

    with pytest.raises(ValueError, match="Eval case not found"):
        run_eval_set(eval_set, case_id="missing")


def test_run_eval_set_can_skip_known_gaps(monkeypatch):
    calls = []

    def fake_call(endpoint, query, top_k=8, timeout=180):
        calls.append(query)
        return _response()

    monkeypatch.setattr(eval_runner, "call_enhanced_endpoint", fake_call)
    eval_set = {
        "name": "unit",
        "version": 1,
        "default_endpoint": "https://example.test/api",
        "cases": [
            _case(id="guard", query="Guard", group="regression_guard"),
            _case(id="gap", query="Gap", group="known_gap"),
        ],
    }

    summary = run_eval_set(eval_set, include_known_gaps=False)

    assert calls == ["Guard"]
    assert summary["total"] == 1
