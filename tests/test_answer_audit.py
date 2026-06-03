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


def test_answer_audit_treats_required_norm_citation_as_answered():
    citations = [
        {"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"},
        {"id": "[2]", "gesetz": "KSchG", "paragraph": "§ 4", "titel": "Anrufung des Arbeitsgerichtes"},
    ]
    plan = {"required_norms": ["BGB § 623", "KSchG § 4"]}
    answer = (
        "Die Kündigung muss schriftlich erfolgen [1]. "
        "Der Arbeitnehmer muss die Klagefrist beim Arbeitsgericht beachten [2]."
    )

    audit = audit_answer_sources(answer, citations, retrieval_plan=plan)

    assert audit["missing_required_in_answer"] == []
    assert audit["status"] == "pass"


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


def test_answer_audit_accepts_grouped_citation_ids():
    citations = [
        {"id": "[1]", "gesetz": "KSchG", "paragraph": "§ 23", "titel": "Geltungsbereich"},
        {"id": "[3]", "gesetz": "KSchG", "paragraph": "§ 1", "titel": "Sozial ungerechtfertigte Kündigungen"},
        {"id": "[4]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform der Kündigung"},
        {"id": "[8]", "gesetz": "BGB", "paragraph": "§ 130", "titel": "Wirksamwerden der Willenserklärung"},
    ]
    answer = (
        "Die Kündigung muss schriftlich erfolgen und dem Arbeitnehmer zugehen [4, 8]. "
        "Das Kündigungsschutzgesetz ist bei Betrieben mit mehr als zehn Arbeitnehmern und "
        "nach sechs Monaten Beschäftigung relevant [1, 3]."
    )

    audit = audit_answer_sources(answer, citations)

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 2
    assert audit["cited_claims"] == 2


def test_answer_audit_ignores_irac_questions_headings_and_intake_prompts():
    audit = audit_answer_sources(
        "### 1. Issue (Rechtliche Fragestellung)\n"
        "Welche formalen Voraussetzungen müssen für eine ordentliche Kündigung erfüllt sein?\n"
        "**a) Form der Kündigung:**\n"
        "Liegt die Kündigung schriftlich und eigenhändig unterschrieben vor?\n"
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1].",
        [{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1
    assert audit["cited_claims"] == 1


def test_answer_audit_keeps_ordinal_dates_and_deadlines_together():
    audit = audit_answer_sources(
        "Bei zehn oder weniger Arbeitnehmern gelten Sonderregeln für Arbeitsverhältnisse, "
        "die nach dem 31. Dezember 2003 begonnen haben [1]. "
        "Grundsätzlich beträgt die Kündigungsfrist vier Wochen zum 15. oder zum Monatsende [2].",
        [
            {"id": "[1]", "gesetz": "KSchG", "paragraph": "§ 23", "titel": "Geltungsbereich"},
            {"id": "[2]", "gesetz": "BGB", "paragraph": "§ 622", "titel": "Kündigungsfristen"},
        ],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 2
    assert audit["cited_claims"] == 2


def test_answer_audit_ignores_summary_disclaimer_and_document_requests():
    audit = audit_answer_sources(
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1].\n"
        "**Zusammenfassende Kernaussage:**\n"
        "Eine ordentliche Kündigung muss schriftlich erfolgen und Fristen einhalten.\n"
        "Bitte stellen Sie uns den Arbeitsvertrag und das Kündigungsschreiben zur Verfügung.\n"
        "Dies ist eine allgemeine Information, keine Rechtsberatung.",
        [{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1
    assert audit["cited_claims"] == 1


def test_answer_audit_ignores_bold_intro_labels_and_numbered_checklist_items():
    audit = audit_answer_sources(
        "**Kündigungsschutzgesetz (KSchG):** Das Kündigungsschutzgesetz regelt die sozialen Anforderungen.\n"
        "* **Soziale Rechtfertigung:** Eine Kündigung darf nicht sozial ungerechtfertigt sein [1].\n"
        "4.  **Anwendbarkeit des Kündigungsschutzgesetzes:** Prüfen Sie, ob das Gesetz anwendbar ist.\n"
        "Ein Anwalt wird Ihren konkreten Fall detailliert prüfen.",
        [{"id": "[1]", "gesetz": "KSchG", "paragraph": "§ 1", "titel": "Sozial ungerechtfertigte Kündigungen"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1
    assert audit["cited_claims"] == 1


def test_answer_audit_ignores_generic_assessment_frames():
    audit = audit_answer_sources(
        "Für die Wirksamkeit einer ordentlichen Kündigung sind verschiedene gesetzliche Regelungen zu beachten.\n"
        "Um die Wirksamkeit einer ordentlichen Kündigung zu beurteilen, müssen alle genannten Aspekte geprüft werden.\n"
        "Für eine abschließende Beurteilung Ihres konkreten Falls wäre es hilfreich, wenn Sie Unterlagen einreichen.\n"
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1].",
        [{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1
    assert audit["cited_claims"] == 1


def test_answer_audit_ignores_soft_risk_and_special_protection_prompts():
    audit = audit_answer_sources(
        "Für bestimmte Arbeitnehmergruppen kann besonderer Kündigungsschutz bestehen.\n"
        "Es ist ratsam, alle diese Punkte sorgfältig zu prüfen, um rechtliche Risiken zu minimieren.\n"
        "Die Kündigung muss nach § 623 BGB schriftlich erfolgen [1].",
        [{"id": "[1]", "gesetz": "BGB", "paragraph": "§ 623", "titel": "Schriftform"}],
    )

    assert audit["status"] == "pass"
    assert audit["material_claims"] == 1
    assert audit["cited_claims"] == 1
