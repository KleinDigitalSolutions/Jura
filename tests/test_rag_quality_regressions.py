"""Regression tests for retrieval/context bugs that affect answer quality."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx

from src.ingestion.rag_pipeline import LegalSearcher
from src.retrieval.enhanced_search import EnhancedLegalSearch


def test_get_related_returns_document_payload_not_index_wrapper():
    source = {
        "typ": "gesetz",
        "abkürzung": "BGB",
        "paragraph": "§ 242",
        "paragraph_titel": "Treu und Glauben",
        "inhalt": "Der Schuldner ist verpflichtet...",
    }
    related = {
        "typ": "gesetz",
        "abkürzung": "BGB",
        "paragraph": "§ 241",
        "paragraph_titel": "Pflichten aus dem Schuldverhältnis",
        "inhalt": "Kraft des Schuldverhältnisses...",
    }

    graph = nx.DiGraph()
    graph.add_edge("BGB||§ 242", "BGB||§ 241")
    indexer = SimpleNamespace(
        qdrant=None,
        embedder=None,
        graph=graph,
        _para_index={
            "BGB||§ 242": {"doc": source, "index": 0},
            "BGB||§ 241": {"doc": related, "index": 1},
        },
        _doc_label=lambda doc: f"{doc.get('paragraph')} {doc.get('abkürzung')}",
    )

    result = LegalSearcher(indexer).get_related(source)

    assert result[0]["paragraph"] == "§ 241"
    assert result[0]["inhalt"] == "Kraft des Schuldverhältnisses..."
    assert result[0]["doc_index"] == 1
    assert "doc" not in result[0]


def test_enhanced_search_reports_kg_expansion_count():
    primary = {
        "pid": "BGB||§ 242",
        "score": 0.9,
        "abkürzung": "BGB",
        "paragraph": "§ 242",
        "inhalt": "Treu und Glauben",
    }
    related = {
        "pid": "BGB||§ 241",
        "abkürzung": "BGB",
        "paragraph": "§ 241",
        "inhalt": "Pflichten aus dem Schuldverhältnis",
    }

    searcher = MagicMock()
    searcher.search_multi_query.return_value = [primary]
    searcher.get_related.return_value = [related]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = ["Treu und Glauben § 242 BGB"]
    classifier = MagicMock()
    classifier.classify.return_value = "Zivilrecht"

    result = EnhancedLegalSearch(searcher, rewriter, classifier).enhanced_search(
        "Was bedeutet Treu und Glauben?",
        top_k=1,
    )

    assert result["kg_expanded_count"] == 1
    assert result["results"][1]["context_type"] == "citation_kg"


def test_enhanced_search_skips_empty_kg_documents():
    primary = {
        "pid": "BGB||§ 242",
        "score": 0.9,
        "abkürzung": "BGB",
        "paragraph": "§ 242",
        "inhalt": "Treu und Glauben",
    }
    empty_related = {"pid": "BGB||§ 241"}

    searcher = MagicMock()
    searcher.search_multi_query.return_value = [primary]
    searcher.get_related.return_value = [empty_related]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = ["Treu und Glauben § 242 BGB"]
    classifier = MagicMock()
    classifier.classify.return_value = "Zivilrecht"

    result = EnhancedLegalSearch(searcher, rewriter, classifier).enhanced_search(
        "Was bedeutet Treu und Glauben?",
        top_k=1,
    )

    assert result["kg_expanded_count"] == 0
    assert len(result["results"]) == 1


def test_enhanced_search_skips_cross_area_kg_documents():
    primary = {
        "pid": "BGB||§ 242",
        "score": 0.9,
        "abkürzung": "BGB",
        "paragraph": "§ 242",
        "rechtsgebiet": "Zivilrecht",
        "inhalt": "Treu und Glauben",
    }
    unrelated = {
        "pid": "StGB||§ 245",
        "abkürzung": "StGB",
        "paragraph": "§ 245",
        "rechtsgebiet": "Strafrecht",
        "inhalt": "Führungsaufsicht",
    }

    searcher = MagicMock()
    searcher.search_multi_query.return_value = [primary]
    searcher.get_related.return_value = [unrelated]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = ["Treu und Glauben § 242 BGB"]
    classifier = MagicMock()
    classifier.classify.return_value = "Zivilrecht"

    result = EnhancedLegalSearch(searcher, rewriter, classifier).enhanced_search(
        "Was bedeutet Treu und Glauben?",
        top_k=1,
    )

    assert result["kg_expanded_count"] == 0
    assert len(result["results"]) == 1


def test_enhanced_search_limits_kg_to_explicit_query_law():
    primary = {
        "pid": "BGB||§ 242",
        "score": 0.9,
        "abkürzung": "BGB",
        "paragraph": "§ 242",
        "rechtsgebiet": "Zivilrecht",
        "inhalt": "Treu und Glauben",
    }
    cross_law = {
        "pid": "HGB||§ 166",
        "abkürzung": "HGB",
        "paragraph": "§ 166",
        "rechtsgebiet": "Zivilrecht",
        "inhalt": "Informationsrecht der Kommanditisten",
    }

    searcher = MagicMock()
    searcher.search_multi_query.return_value = [primary]
    searcher.get_related.return_value = [cross_law]
    rewriter = MagicMock()
    rewriter.rewrite.return_value = ["Treu und Glauben § 242 BGB"]
    classifier = MagicMock()
    classifier.classify.return_value = "Zivilrecht"

    result = EnhancedLegalSearch(searcher, rewriter, classifier).enhanced_search(
        "Was bedeutet Treu und Glauben nach § 242 BGB?",
        top_k=1,
    )

    assert result["kg_expanded_count"] == 0
    assert len(result["results"]) == 1
