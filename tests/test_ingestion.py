"""Tests for the current RAG ingestion/indexing helpers."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingestion import rag_pipeline
from src.ingestion.rag_pipeline import LegalIndexer, LegalRAGPipeline, _doc_para_id


def test_doc_para_id_for_gesetz():
    doc = {"typ": "gesetz", "abkürzung": "BGB", "paragraph": "§ 242"}
    assert _doc_para_id(doc) == "BGB||§ 242"


def test_doc_para_id_for_urteil():
    doc = {"typ": "urteil", "gericht": "BGH", "aktenzeichen": "VI ZR 1/24"}
    assert _doc_para_id(doc) == "BGH||VI ZR 1/24"


def test_index_skips_empty_and_repealed_documents(tmp_path, monkeypatch):
    embedder = SimpleNamespace(dim=1024, embed=MagicMock(return_value=([], [])))
    qdrant = MagicMock()
    monkeypatch.setattr(rag_pipeline, "QdrantClient", lambda path: qdrant)
    indexer = LegalIndexer(storage_dir=tmp_path, embedder=embedder)
    indexer._ensure_collection = MagicMock()
    monkeypatch.setattr(rag_pipeline, "DOCS_PATH", str(tmp_path / "documents.json"))
    monkeypatch.setattr(rag_pipeline, "GRAPH_PATH", str(tmp_path / "legal_graph.graphml"))

    stats = indexer.index(
        [
            {"abkürzung": "BGB", "paragraph": "§ 1", "inhalt": ""},
            {"abkürzung": "BGB", "paragraph": "§ 2", "inhalt": "(weggefallen)"},
        ],
        incremental=False,
    )

    assert stats == {"indexed": 0, "skipped": 2}
    embedder.embed.assert_not_called()
    qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_insert_empty_documents_returns_initial_stats(monkeypatch):
    pipeline = object.__new__(LegalRAGPipeline)
    pipeline._stats = {"inserted": 0, "failed": 0, "skipped": 0}

    stats = await LegalRAGPipeline.insert_documents(pipeline, [])

    assert stats == {"inserted": 0, "failed": 0, "skipped": 0}
