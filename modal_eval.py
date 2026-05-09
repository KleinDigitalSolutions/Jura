"""Run retrieval quality evaluation on Modal (T4 GPU for fast reranker).

Usage:
    modal run modal_eval.py

Compares search() vs search_multi_query() across 17 test cases.
"""
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

# Constants (duplicated from modal_deploy.py since it can't be imported via modal run)
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
VOLUME_PATH = Path("/legal_rag_storage")
VOLUME = modal.Volume.from_name("legal-rag-data")
DEEPSEEK_SECRET = modal.Secret.from_name("my-deepseek-secret")
ANTHROPIC_SECRET = modal.Secret.from_name("my-anthropic-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "FlagEmbedding>=1.2.0",
        "qdrant-client>=1.13.0",
        "transformers==4.57.6",
        "networkx>=3.0",
        "loguru>=0.7",
        "scikit-learn>=1.0.0",
        "python-dotenv>=1.0",
        "openai>=1.0.0",
        "anthropic>=0.30.0",
        "torch",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        f"python -c \"from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('{EMBEDDING_MODEL}', use_fp16=False)\"",
        f"python -c \"from FlagEmbedding import FlagReranker; FlagReranker('{RERANKER_MODEL}', use_fp16=False)\"",
    )
    .add_local_dir(str(Path(__file__).parent / "src"), remote_path="/src", copy=True)
)

app = modal.App("legal-rag-eval")

# Inline EVAL_SET (17 test cases, copied from tests/test_retrieval_quality.py)
EVAL_SET: list[dict[str, Any]] = [
    {"query": "Mein Chef hat mich gefeuert ohne Grund, ist das erlaubt", "expected_paragraphs": ["§ 626"], "expected_laws": ["BGB", "KSchG"], "expected_rechtsgebiet": "Arbeitsrecht", "description": "Außerordentliche Kündigung"},
    {"query": "Kaution nach Mietende zurück", "expected_paragraphs": ["§ 548", "§ 551"], "expected_laws": ["BGB"], "expected_rechtsgebiet": "Zivilrecht", "description": "Mietkaution Rückforderung"},
    {"query": "GmbH Gründung Stammkapital", "expected_paragraphs": ["§ 1", "§ 5"], "expected_laws": ["GmbHG"], "expected_rechtsgebiet": "Handelsrecht", "description": "Gesellschaftsgründung"},
    {"query": "Mir wurde mein Handy geklaut, was droht dem Täter", "expected_paragraphs": ["§ 242"], "expected_laws": ["StGB"], "expected_rechtsgebiet": "Strafrecht", "description": "Einfacher Diebstahl"},
    {"query": "Scheidung Kindesunterhalt", "expected_paragraphs": ["§ 1569", "§ 1612"], "expected_laws": ["BGB"], "expected_rechtsgebiet": "Familienrecht", "description": "Familienrecht"},
    {"query": "Ich will ein Haus bauen, brauche ich eine Baugenehmigung", "expected_paragraphs": ["§ 29", "§ 33"], "expected_laws": ["BauGB"], "expected_rechtsgebiet": "Verwaltungsrecht", "description": "Öffentliches Baurecht"},
    {"query": "Steuererklärung Frist versäumt", "expected_paragraphs": ["§ 149", "§ 152"], "expected_laws": ["AO"], "expected_rechtsgebiet": "Steuerrecht", "description": "Steuerfristen"},
    {"query": "Insolvenz Gläubiger Forderung anmelden", "expected_paragraphs": ["§ 38", "§ 174"], "expected_laws": ["InsO"], "expected_rechtsgebiet": "Insolvenzrecht", "description": "Insolvenzverfahren"},
    {"query": "Darf meine Firma KI mit Bildern aus dem Internet trainieren", "expected_paragraphs": ["§ 44b", "§ 60d"], "expected_laws": ["UrhG"], "expected_rechtsgebiet": "Urheberrecht", "description": "Text and Data Mining"},
    {"query": "Ich habe ein kaputtes Auto gekauft, was kann ich vom Händler verlangen", "expected_paragraphs": ["§ 434", "§ 437"], "expected_laws": ["BGB"], "expected_rechtsgebiet": "Zivilrecht", "description": "Sachmangel Gewährleistung"},
    {"query": "Abmahnung Arbeitnehmer ungerechtfertigt", "expected_paragraphs": ["§ 1"], "expected_laws": ["KSchG"], "expected_rechtsgebiet": "Arbeitsrecht", "description": "Abmahnung Kündigungsschutz"},
    {"query": "Testament Erbe Pflichtteil", "expected_paragraphs": ["§ 1922", "§ 2303"], "expected_laws": ["BGB"], "expected_rechtsgebiet": "Zivilrecht", "description": "Erbrecht Pflichtteil"},
    {"query": "Prokura erteilen Umfang", "expected_paragraphs": ["§ 48", "§ 49"], "expected_laws": ["HGB"], "expected_rechtsgebiet": "Handelsrecht", "description": "Handelsrecht Prokura"},
    {"query": "Sorgerecht Verfahren Familiengericht", "expected_paragraphs": ["§ 151", "§ 1671"], "expected_laws": ["FamFG", "BGB"], "expected_rechtsgebiet": "Familienrecht", "description": "Sorgerechtsverfahren"},
    {"query": "Betriebsrat Gründen Schwellenwert", "expected_paragraphs": ["§ 1"], "expected_laws": ["BetrVG"], "expected_rechtsgebiet": "Arbeitsrecht", "description": "Betriebsverfassung"},
    {"query": "Strafanzeige erstatten Verfahren", "expected_paragraphs": ["§ 152", "§ 158"], "expected_laws": ["StPO"], "expected_rechtsgebiet": "Strafrecht", "description": "Strafverfahrensrecht"},
    {"query": "DSGVO Verstoß melden Bußgeld", "expected_paragraphs": ["§ 22", "§ 43"], "expected_laws": ["BDSG", "TDDDG"], "expected_rechtsgebiet": "Datenschutzrecht", "description": "DSGVO/BDSG enforcement"},
]


def _has_expected_para(result: dict, expected_paras: list[str]) -> bool:
    para = result.get("paragraph", "") or ""
    for ep in expected_paras:
        if para.startswith(ep) or para.startswith("§§ " + ep.lstrip("§ ")):
            return True
    return False


def _recall_at_5(results: list[dict], test_case: dict) -> float:
    expected = test_case["expected_paragraphs"]
    if not expected:
        return 1.0
    found = sum(1 for ep in expected if any(_has_expected_para(r, [ep]) for r in results[:5]))
    return found / len(expected)


def _rechtsgebiet_accuracy(results: list[dict], test_case: dict) -> bool:
    expected_rg = test_case.get("expected_rechtsgebiet")
    if not expected_rg:
        return True
    return any(r.get("rechtsgebiet") == expected_rg for r in results[:5])


def _format_table(rows: list[list[str]], header: list[str]) -> str:
    col_widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines: list[str] = []
    sep = "  "
    lines.append(sep.join(h.ljust(w) for h, w in zip(header, col_widths)))
    lines.append(sep.join("-" * w for w in col_widths))
    for row in rows:
        lines.append(sep.join(str(r).ljust(w) for r, w in zip(row, col_widths)))
    return "\n".join(lines)


def _run_eval(search_func, test_cases: list[dict], top_k: int = 5) -> dict:
    recall_scores: list[float] = []
    rg_accuracies: list[bool] = []
    latencies: list[float] = []
    per_case: list[dict] = []

    for tc in test_cases:
        t0 = time.monotonic()
        try:
            results = search_func(tc["query"], top_k=top_k)
        except Exception as e:
            print(f"  SEARCH FAILED: {tc['query']!r}: {e}")
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


@app.function(
    image=image,
    volumes={VOLUME_PATH: VOLUME},
    secrets=[DEEPSEEK_SECRET, ANTHROPIC_SECRET],
    gpu="T4",
    timeout=1200,
)
def run_eval():
    """Load index from volume, run eval comparison, print table."""
    sys.path.insert(0, "/")
    os.environ["LEGAL_RAG_STORAGE"] = str(VOLUME_PATH)

    from src.ingestion.rag_pipeline import LegalEmbedder, LegalIndexer, LegalSearcher
    from src.retrieval.enhanced_search import EnhancedLegalSearch
    from src.retrieval.query_rewriter import LegalQueryRewriter
    from src.retrieval.query_classifier import LegalQueryClassifier

    print("Loading index from Volume...")
    t0 = time.monotonic()
    embedder = LegalEmbedder(model_name=EMBEDDING_MODEL)
    indexer = LegalIndexer(embedder=embedder)
    searcher = LegalSearcher(indexer)
    indexer.load()
    print(f"Loaded {indexer.total_docs} docs in {time.monotonic() - t0:.1f}s")
    print(f"GPU available: {__import__('torch').cuda.is_available()}")

    # Single-query baseline
    print("\nRunning single-query eval (17 cases)...")
    single = _run_eval(
        lambda q, top_k: searcher.search(q, top_k=top_k),
        EVAL_SET,
        top_k=5,
    )

    # Multi-query (original only, no LLM rewriting for deterministic comparison)
    print("Running multi-query eval (17 cases)...")
    multi = _run_eval(
        lambda q, top_k: searcher.search_multi_query([q], top_k=top_k),
        EVAL_SET,
        top_k=5,
    )

    # Enhanced: full pipeline with real LLM rewriting via DeepSeek
    print("Running enhanced eval (17 cases) with LLM rewriting...")
    enhanced = _run_eval(
        lambda q, top_k: EnhancedLegalSearch(
            searcher=searcher,
            rewriter=LegalQueryRewriter(),
            classifier=LegalQueryClassifier(),
        ).enhanced_search(q, top_k=top_k)["results"],
        EVAL_SET,
        top_k=5,
    )

    # Print comparison table — three columns
    header = ["", "Recall@5", "RG-Acc", "Latency(s)"]
    rows = [
        ["search() (single)", single["avg_recall@5"], single["avg_rg_accuracy"], single["avg_latency"]],
        ["search_multi_query()", multi["avg_recall@5"], multi["avg_rg_accuracy"], multi["avg_latency"]],
        ["enhanced_search()", enhanced["avg_recall@5"], enhanced["avg_rg_accuracy"], enhanced["avg_latency"]],
    ]

    print("\n=== Retrieval Quality Comparison (T4 GPU) ===\n")
    print(_format_table(rows, header))
    print()

    # Three-way per-case comparison
    print("Per-Case Results (single | multi | enhanced):\n")
    pc_header = ["Case", "Recall", "RG-Acc", "Latency"]
    pc_table = []
    for s, m, e in zip(single["per_case"], multi["per_case"], enhanced["per_case"]):
        recall_str = f"{s['recall_at_5']:.2f} | {m['recall_at_5']:.2f} | {e['recall_at_5']:.2f}"
        rg_str = f"{'✓' if s['rg_accurate'] else '✗'} | {'✓' if m['rg_accurate'] else '✗'} | {'✓' if e['rg_accurate'] else '✗'}"
        lat_str = f"{s['latency']:.2f}s | {m['latency']:.2f}s | {e['latency']:.2f}s"
        desc_short = (s["description"] + " " * 40)[:35]
        pc_table.append([desc_short, recall_str, rg_str, lat_str])
    print(_format_table(pc_table, pc_header))

    return single, multi, enhanced


@app.local_entrypoint()
def main():
    single, multi, enhanced = run_eval.remote()
    print(f"\nDone. Single: {single['avg_recall@5']} | Multi: {multi['avg_recall@5']} | Enhanced: {enhanced['avg_recall@5']}")
