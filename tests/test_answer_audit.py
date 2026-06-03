"""Regression tests for post-generation source-level answer auditing."""

from src.retrieval.answer_audit import audit_answer_sources


def test_answer_audit_passes_grounded_required_norms():
    citations = [
        {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"},
        {"id": "[2]", "gesetz": "KSchG", "paragraph": "§ 4", "titel": "Anrufung des Arbeitsgerichtes"},
    ]
    plan = {
        "profiles": ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"],
        "required_norms": ["BGB § 623", "KSchG § 4"],
    }
    answer = (
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1]. "
        "Die Kündigungsschutzklagefrist beträgt drei Wochen nach § 4 KSchG [2]."
    )

    audit = audit_answer_sources(answer, citations, retrieval_plan=plan)

    assert audit["status"] == "pass"
    assert audit["score"] == 100
    assert audit["issue_count"] == 0
    assert audit["material_claims"] == 2
    assert audit["cited_claims"] == 2


def test_answer_audit_flags_missing_citation_and_overconfidence():
    citations = [
        {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"},
    ]
    answer = (
        "Die Kündigung ist immer wirksam. "
        "Nach § 999 BGB besteht ein Anspruch [1]."
    )

    audit = audit_answer_sources(answer, citations)

    issues = {issue["issue"] for issue in audit["issues"]}
    assert audit["status"] == "fail"
    assert "missing_claim_citation" in issues
    assert "paragraph_not_in_cited_sources" in issues


def test_answer_audit_flags_required_norm_omission_and_deadline():
    citations = [
        {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"},
        {"id": "[2]", "gesetz": "KSchG", "paragraph": "§ 4", "titel": "Anrufung des Arbeitsgerichtes"},
    ]
    plan = {
        "profiles": ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"],
        "required_norms": ["BGB § 623", "KSchG § 4"],
    }
    answer = "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1]."

    audit = audit_answer_sources(answer, citations, retrieval_plan=plan)

    issues = {issue["issue"] for issue in audit["issues"]}
    assert audit["status"] == "fail"
    assert audit["missing_required_in_answer"] == ["KSchG § 4"]
    assert "missing_required_norm_in_answer" in issues
    assert "missing_deadline" in issues


def test_answer_audit_respects_required_norms_missing_from_retrieval():
    citations = [
        {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"},
    ]
    plan = {
        "profiles": ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"],
        "required_norms": ["BGB § 623", "KSchG § 4"],
    }
    source_audit = {"missing_required": ["KSchG § 4"]}
    answer = "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1]."

    audit = audit_answer_sources(
        answer,
        citations,
        retrieval_plan=plan,
        source_audit=source_audit,
    )

    assert audit["missing_required_in_answer"] == []
    assert audit["retrieval_missing_required"] == ["KSchG § 4"]


def test_answer_audit_ignores_structural_headings():
    audit = audit_answer_sources(
        "### 1. Issue (Rechtliche Fragestellung)\n"
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1].",
        [{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1


def test_answer_audit_accepts_paragraphs_referenced_inside_cited_source_preview():
    citations = [
        {
            "id": "[1]",
            "gesetz": "BGB",
            "paragraph": "§ 620",
            "titel": "Beendigung des Dienstverhältnisses",
            "text_preview": "Ist die Dauer nicht bestimmt, so kann nach den §§ 621 bis 623 gekündigt werden.",
        }
    ]
    answer = "§ 620 BGB verweist für Kündigungen auf §§ 621 bis 623 BGB [1]."

    audit = audit_answer_sources(answer, citations)

    issues = {issue["issue"] for issue in audit["issues"]}
    assert "paragraph_not_in_cited_sources" not in issues
