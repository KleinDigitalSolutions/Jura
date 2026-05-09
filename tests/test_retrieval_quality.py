"""Retrieval quality evaluation: compares search() vs search_multi_query().

Usage:
    python -m pytest tests/test_retrieval_quality.py -v -s

Requires an existing index (legal_rag_storage/documents.json).
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from src.ingestion.rag_pipeline import STORAGE_DIR, LegalRAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test cases — 17 queries covering all 10 Rechtsgebiete
# ---------------------------------------------------------------------------
EVAL_SET: list[dict[str, Any]] = [
    {
        "query": "Mein Chef hat mich gefeuert ohne Grund, ist das erlaubt",
        "expected_paragraphs": ["§ 626"],
        "expected_laws": ["BGB", "KSchG"],
        "expected_rechtsgebiet": "Arbeitsrecht",
        "description": "Außerordentliche Kündigung",
    },
    {
        "query": "Kaution nach Mietende zurück",
        "expected_paragraphs": ["§ 548", "§ 551"],
        "expected_laws": ["BGB"],
        "expected_rechtsgebiet": "Zivilrecht",
        "description": "Mietkaution Rückforderung",
    },
    {
        "query": "GmbH Gründung Stammkapital",
        "expected_paragraphs": ["§ 1", "§ 5"],
        "expected_laws": ["GmbHG"],
        "expected_rechtsgebiet": "Handelsrecht",
        "description": "Gesellschaftsgründung",
    },
    {
        "query": "Mir wurde mein Handy geklaut, was droht dem Täter",
        "expected_paragraphs": ["§ 242"],
        "expected_laws": ["StGB"],
        "expected_rechtsgebiet": "Strafrecht",
        "description": "Einfacher Diebstahl",
    },
    {
        "query": "Scheidung Kindesunterhalt",
        "expected_paragraphs": ["§ 1569", "§ 1612"],
        "expected_laws": ["BGB"],
        "expected_rechtsgebiet": "Familienrecht",
        "description": "Familienrecht",
    },
    {
        "query": "Ich will ein Haus bauen, brauche ich eine Baugenehmigung",
        "expected_paragraphs": ["§ 29", "§ 33"],
        "expected_laws": ["BauGB"],
        "expected_rechtsgebiet": "Verwaltungsrecht",
        "description": "Öffentliches Baurecht",
    },
    {
        "query": "Steuererklärung Frist versäumt",
        "expected_paragraphs": ["§ 149", "§ 152"],
        "expected_laws": ["AO"],
        "expected_rechtsgebiet": "Steuerrecht",
        "description": "Steuerfristen",
    },
    {
        "query": "Insolvenz Gläubiger Forderung anmelden",
        "expected_paragraphs": ["§ 38", "§ 174"],
        "expected_laws": ["InsO"],
        "expected_rechtsgebiet": "Insolvenzrecht",
        "description": "Insolvenzverfahren",
    },
    {
        "query": "Darf meine Firma KI mit Bildern aus dem Internet trainieren",
        "expected_paragraphs": ["§ 44b", "§ 60d"],
        "expected_laws": ["UrhG"],
        "expected_rechtsgebiet": "Urheberrecht",
        "description": "Text and Data Mining",
    },
    {
        "query": "Ich habe ein kaputtes Auto gekauft, was kann ich vom Händler verlangen",
        "expected_paragraphs": ["§ 434", "§ 437"],
        "expected_laws": ["BGB"],
        "expected_rechtsgebiet": "Zivilrecht",
        "description": "Sachmangel Gewährleistung",
    },
    {
        "query": "Abmahnung Arbeitnehmer ungerechtfertigt",
        "expected_paragraphs": ["§ 1"],
        "expected_laws": ["KSchG"],
        "expected_rechtsgebiet": "Arbeitsrecht",
        "description": "Abmahnung Kündigungsschutz",
    },
    {
        "query": "Testament Erbe Pflichtteil",
        "expected_paragraphs": ["§ 1922", "§ 2303"],
        "expected_laws": ["BGB"],
        "expected_rechtsgebiet": "Zivilrecht",
        "description": "Erbrecht Pflichtteil",
    },
    {
        "query": "Prokura erteilen Umfang",
        "expected_paragraphs": ["§ 48", "§ 49"],
        "expected_laws": ["HGB"],
        "expected_rechtsgebiet": "Handelsrecht",
        "description": "Handelsrecht Prokura",
    },
    {
        "query": "Sorgerecht Verfahren Familiengericht",
        "expected_paragraphs": ["§ 151", "§ 1671"],
        "expected_laws": ["FamFG", "BGB"],
        "expected_rechtsgebiet": "Familienrecht",
        "description": "Sorgerechtsverfahren",
    },
    {
        "query": "Betriebsrat Gründen Schwellenwert",
        "expected_paragraphs": ["§ 1"],
        "expected_laws": ["BetrVG"],
        "expected_rechtsgebiet": "Arbeitsrecht",
        "description": "Betriebsverfassung",
    },
    {
        "query": "Strafanzeige erstatten Verfahren",
        "expected_paragraphs": ["§ 152", "§ 158"],
        "expected_laws": ["StPO"],
        "expected_rechtsgebiet": "Strafrecht",
        "description": "Strafverfahrensrecht",
    },
    {
        "query": "DSGVO Verstoß melden Bußgeld",
        "expected_paragraphs": ["§ 22", "§ 43"],
        "expected_laws": ["BDSG", "TDDDG"],
        "expected_rechtsgebiet": "Datenschutzrecht",
        "description": "DSGVO/BDSG enforcement",
    },
]

# ---------------------------------------------------------------------------
# Pytest marks
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.skipif(
        not (STORAGE_DIR / "documents.json").exists(),
        reason=f"Kein Index gefunden unter {STORAGE_DIR}/documents.json — führe zuerst python main.py --run-all oder --run-gesetze aus",
    ),
    pytest.mark.slow,
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _has_expected_para(result: dict, expected_paras: list[str]) -> bool:
    """Check if a search result matches any expected paragraph number."""
    para = result.get("paragraph", "") or ""
    for ep in expected_paras:
        if para.startswith(ep) or para.startswith("§§ " + ep.lstrip("§ ")):
            return True
    return False


def _recall_at_5(results: list[dict], test_case: dict) -> float:
    """Recall@5: fraction of expected paragraphs found in top-5 results."""
    expected = test_case["expected_paragraphs"]
    if not expected:
        return 1.0
    found = sum(1 for ep in expected if any(_has_expected_para(r, [ep]) for r in results[:5]))
    return found / len(expected)


def _rechtsgebiet_accuracy(results: list[dict], test_case: dict) -> bool:
    """Check if any top-5 result has the expected Rechtsgebiet."""
    expected_rg = test_case.get("expected_rechtsgebiet")
    if not expected_rg:
        return True
    return any(r.get("rechtsgebiet") == expected_rg for r in results[:5])


def _format_table(rows: list[list[str]], header: list[str]) -> str:
    """Align columns by max width per column."""
    col_widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines: list[str] = []
    sep = "  "
    lines.append(sep.join(h.ljust(w) for h, w in zip(header, col_widths)))
    lines.append(sep.join("-" * w for w in col_widths))
    for row in rows:
        lines.append(sep.join(str(r).ljust(w) for r, w in zip(row, col_widths)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation runs
# ---------------------------------------------------------------------------
def _run_eval(
    search_func,
    test_cases: list[dict],
    top_k: int = 5,
) -> dict[str, Any]:
    """Run evaluation for a search method across all test cases."""
    recall_scores: list[float] = []
    rg_accuracies: list[bool] = []
    latencies: list[float] = []
    per_case: list[dict] = []

    for tc in test_cases:
        t0 = time.monotonic()
        try:
            results = search_func(tc["query"], top_k=top_k)
        except Exception as e:
            logger.error(f"Search failed for {tc['query']!r}: {e}")
            results = []
        latency = time.monotonic() - t0

        recall = _recall_at_5(results, tc)
        rg_ok = _rechtsgebiet_accuracy(results, tc)

        recall_scores.append(recall)
        rg_accuracies.append(rg_ok)
        latencies.append(latency)
        per_case.append({
            "query": tc["query"],
            "description": tc["description"],
            "recall_at_5": round(recall, 3),
            "rg_accurate": rg_ok,
            "n_results": len(results) if results else 0,
            "latency": round(latency, 3),
        })

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_rg = sum(rg_accuracies) / len(rg_accuracies) if rg_accuracies else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "avg_recall@5": round(avg_recall, 3),
        "avg_rg_accuracy": round(avg_rg, 3),
        "avg_latency": round(avg_latency, 3),
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Tests — use a session-scoped searcher fixture to avoid Qdrant lock conflicts
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def searcher():
    """Load the LegalRAG index once per test session.

    Gracefully handles Qdrant lock contention. If the index is locked
    (e.g. by another process), returns None and tests skip.
    """
    import portalocker

    lock_path = STORAGE_DIR / "qdrant" / ".lock"
    if not lock_path.exists():
        pytest.skip("Kein Qdrant-Index gefunden — führe zuerst python main.py --run-all aus")

    try:
        with open(lock_path) as f:
            portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
            portalocker.unlock(f)
    except portalocker.LockException:
        pytest.skip("Qdrant-Index ist von einem anderen Prozess gesperrt")

    pipeline = LegalRAGPipeline()
    assert pipeline.indexer.load(), "Index konnte nicht geladen werden"
    return pipeline.searcher


def test_index_available(searcher):
    """Verify the index can be loaded."""
    assert searcher is not None


@pytest.mark.xfail(
    reason="DSGVO/BDSG ist EU-Recht — im deutschen Gesetzesindex vermutlich unterrepräsentiert",
    strict=False,
)
def test_dsgvo_xfail(searcher):
    """DSGVO xfail marker: recognized but expected to fail."""
    tc = EVAL_SET[-1]  # DSGVO test case
    results = searcher.search(tc["query"], top_k=5)
    recall = _recall_at_5(results, tc)
    assert recall > 0, "DSGVO-paragraphen nicht gefunden (erwartet)"


def test_eval_comparison(searcher, capsys):
    """Run eval comparison: search() vs search_multi_query() and print table."""
    # Single-query baseline
    single = _run_eval(
        lambda q, top_k: searcher.search(q, top_k=top_k),
        EVAL_SET,
        top_k=5,
    )

    # Multi-query: use original query only (no LLM rewriting) so test is deterministic
    multi = _run_eval(
        lambda q, top_k: searcher.search_multi_query([q], top_k=top_k),
        EVAL_SET,
        top_k=5,
    )

    # Print comparison table
    header = ["", "Recall@5", "RG-Acc", "Latency(s)"]
    rows = [
        ["search() (single)", single["avg_recall@5"], single["avg_rg_accuracy"], single["avg_latency"]],
        ["search_multi_query()", multi["avg_recall@5"], multi["avg_rg_accuracy"], multi["avg_latency"]],
    ]

    print("\n=== Retrieval Quality Comparison ===\n")
    print(_format_table(rows, header))
    print()

    # Best/worst improvers
    deltas = []
    for s, m in zip(single["per_case"], multi["per_case"]):
        delta = m["recall_at_5"] - s["recall_at_5"]
        deltas.append((delta, s["description"], s["query"], s["recall_at_5"], m["recall_at_5"]))
    deltas.sort(key=lambda x: x[0], reverse=True)

    print("Top Improvers:\n")
    for delta, desc, query, s_recall, m_recall in deltas[:5]:
        print(f"  +{delta:+.2f}  {desc:40s}  single={s_recall:.2f}  multi={m_recall:.2f}")

    print(f"\n  ---  {len(deltas) - 5} more cases ---" if len(deltas) > 5 else "")
    print()

    # Print per-case table
    print("Per-Case Results (single | multi):\n")
    pc_header = ["Case", "Recall", "RG-Acc", "Latency"]
    pc_table = []
    for s, m in zip(single["per_case"], multi["per_case"]):
        recall_str = f"{s['recall_at_5']:.2f} | {m['recall_at_5']:.2f}"
        rg_str = f"{'✓' if s['rg_accurate'] else '✗'} | {'✓' if m['rg_accurate'] else '✗'}"
        lat_str = f"{s['latency']:.2f}s | {m['latency']:.2f}s"
        desc_short = (s["description"] + " " * 40)[:35]
        pc_table.append([desc_short, recall_str, rg_str, lat_str])
    print(_format_table(pc_table, pc_header))

    # Assertions: both should have at least minimal recall
    assert single["avg_recall@5"] >= 0.1, f"Single query recall@{5} is very low: {single['avg_recall@5']}"
