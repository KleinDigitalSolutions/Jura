"""Regression tests for deterministic legal retrieval quality controls."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrieval.enhanced_search import EnhancedLegalSearch
from src.retrieval.legal_quality import (
    apply_legal_quality,
    build_retrieval_plan,
    source_key,
)


def _law_doc(law: str, paragraph: str, title: str = "") -> dict:
    return {
        "typ": "gesetz",
        "abkürzung": law,
        "paragraph": paragraph,
        "paragraph_titel": title or f"{law} {paragraph}",
        "rechtsgebiet": "Arbeitsrecht" if law in {"BGB", "KSchG", "BetrVG", "SGB IX"} else "Sonstiges",
        "inhalt": f"Normtext zu {law} {paragraph}",
    }


def _fake_searcher(documents: list[dict]):
    indexer = SimpleNamespace(
        documents=documents,
        _doc_label=lambda doc: f"{doc.get('paragraph')} {doc.get('abkürzung')}",
    )
    return SimpleNamespace(indexer=indexer)


def test_retrieval_plan_detects_employee_ordinary_termination_profile():
    plan = build_retrieval_plan(
        "Welche Anforderungen gelten für eine ordentliche Kündigung eines Arbeitnehmers?",
        rechtsgebiet="Arbeitsrecht",
    )

    assert plan.profiles == ["arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"]
    assert [n.label for n in plan.required_norms] == [
        "BGB § 623",
        "BGB § 622",
        "KSchG § 1",
        "KSchG § 4",
        "KSchG § 23",
        "BetrVG § 102",
    ]
    assert "BGB § 580a" in [n.label for n in plan.excluded_norms]


def test_apply_legal_quality_injects_required_sources_and_filters_false_positive():
    documents = [
        _law_doc("BGB", "§ 623", "Schriftform der Kündigung"),
        _law_doc("BGB", "§ 130", "Wirksamwerden der Willenserklärung gegenüber Abwesenden"),
        _law_doc("BGB", "§ 622", "Kündigungsfristen bei Arbeitsverhältnissen"),
        _law_doc("KSchG", "§ 1", "Sozial ungerechtfertigte Kündigungen"),
        _law_doc("KSchG", "§ 4", "Anrufung des Arbeitsgerichtes"),
        _law_doc("KSchG", "§ 23", "Geltungsbereich"),
        _law_doc("BetrVG", "§ 102", "Mitbestimmung bei Kündigungen"),
        _law_doc("SGB IX", "§ 168", "Erfordernis der Zustimmung"),
        _law_doc("BGB", "§ 580a", "Kündigungsfristen"),
    ]
    initial_results = [
        _law_doc("BetrVG", "§ 102", "Mitbestimmung bei Kündigungen"),
        _law_doc("BGB", "§ 580a", "Kündigungsfristen"),
        _law_doc("TzBfG", "§ 16", "Folgen unwirksamer Befristung"),
        {**_law_doc("InsO", "§ 126", "Beschlußverfahren zum Kündigungsschutz"), "rechtsgebiet": "Arbeitsrecht"},
    ]

    qualified, plan, audit = apply_legal_quality(
        query="Welche Anforderungen gelten für eine ordentliche Kündigung eines Arbeitnehmers?",
        results=initial_results,
        searcher=_fake_searcher(documents),
        rechtsgebiet="Arbeitsrecht",
        top_k=8,
    )

    keys = {source_key(source) for source in qualified}
    assert ("BGB", "§ 580a") not in keys
    assert ("TZBFG", "§ 16") not in keys
    assert ("INSO", "§ 126") not in keys
    assert {
        ("BGB", "§ 130"),
        ("BGB", "§ 623"),
        ("BGB", "§ 622"),
        ("KSCHG", "§ 1"),
        ("KSCHG", "§ 4"),
        ("KSCHG", "§ 23"),
        ("BETRVG", "§ 102"),
        ("SGB IX", "§ 168"),
    }.issubset(keys)
    assert plan.has_profile
    assert "BGB § 580a" in [r["source"] for r in audit.rejected]
    assert "TZBFG § 16" in [r["source"] for r in audit.rejected]
    assert "INSO § 126" in [r["source"] for r in audit.rejected]
    assert "BGB § 623" in audit.injected
    assert "BGB § 130" in audit.injected
    assert "SGB IX § 168" in audit.injected
    assert audit.missing_required == []


def test_enhanced_search_applies_legal_quality_layer():
    documents = [
        _law_doc("BGB", "§ 623", "Schriftform der Kündigung"),
        _law_doc("BGB", "§ 130", "Wirksamwerden der Willenserklärung gegenüber Abwesenden"),
        _law_doc("BGB", "§ 622", "Kündigungsfristen bei Arbeitsverhältnissen"),
        _law_doc("KSchG", "§ 1", "Sozial ungerechtfertigte Kündigungen"),
        _law_doc("KSchG", "§ 4", "Anrufung des Arbeitsgerichtes"),
        _law_doc("KSchG", "§ 23", "Geltungsbereich"),
        _law_doc("BetrVG", "§ 102", "Mitbestimmung bei Kündigungen"),
        _law_doc("SGB IX", "§ 168", "Erfordernis der Zustimmung"),
        _law_doc("BGB", "§ 580a", "Kündigungsfristen"),
    ]
    searcher = MagicMock()
    searcher.indexer = _fake_searcher(documents).indexer
    searcher.search_multi_query.return_value = [
        _law_doc("BetrVG", "§ 102", "Mitbestimmung bei Kündigungen"),
        _law_doc("BGB", "§ 580a", "Kündigungsfristen"),
        _law_doc("TzBfG", "§ 16", "Folgen unwirksamer Befristung"),
        {**_law_doc("InsO", "§ 126", "Beschlußverfahren zum Kündigungsschutz"), "rechtsgebiet": "Arbeitsrecht"},
        {**_law_doc("BetrVG", "§ 99", "Mitbestimmung bei personellen Einzelmaßnahmen"), "context_type": "citation_kg"},
    ]
    searcher.get_related.return_value = []
    rewriter = MagicMock()
    rewriter.rewrite.return_value = ["ordentliche Kündigung Arbeitnehmer § 622 BGB"]
    classifier = MagicMock()
    classifier.classify.return_value = "Arbeitsrecht"

    result = EnhancedLegalSearch(searcher, rewriter, classifier).enhanced_search(
        "Welche Anforderungen gelten für eine ordentliche Kündigung eines Arbeitnehmers?",
        top_k=8,
    )

    keys = {source_key(source) for source in result["results"]}
    assert ("BGB", "§ 580a") not in keys
    assert ("BETRVG", "§ 99") not in keys
    assert ("INSO", "§ 126") not in keys
    assert ("TZBFG", "§ 16") not in keys
    assert ("BGB", "§ 130") in keys
    assert ("BGB", "§ 623") in keys
    assert ("KSCHG", "§ 1") in keys
    assert ("SGB IX", "§ 168") in keys
    assert result["retrieval_plan"]["profiles"] == [
        "arbeitsrecht_ordentliche_kuendigung_arbeitnehmer"
    ]
    assert result["source_audit"]["missing_required"] == []
