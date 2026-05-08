"""Tests for RAG ingestion pipeline."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.rag_pipeline import make_doc_id, LegalRAGPipeline


class TestMakeDocId:
    def test_gesetz_doc_id(self):
        doc = {
            "quelle": "gesetze-im-internet.de",
            "typ": "gesetz",
            "abkürzung": "BGB",
            "paragraph": "§ 242",
        }
        doc_id = make_doc_id(doc)
        assert "gesetze-im-internet" in doc_id
        assert "gesetz" in doc_id
        assert "bgb" in doc_id
        assert "242" in doc_id

    def test_urteil_doc_id(self):
        doc = {
            "quelle": "rechtsprechung-im-internet.de",
            "typ": "urteil",
            "gericht": "BGH",
            "aktenzeichen": "VI ZR 1/24",
        }
        doc_id = make_doc_id(doc)
        assert "rechtsprechung-im-internet" in doc_id
        assert "urteil" in doc_id
        assert "bgh" in doc_id
        assert "vi_zr_1-24" in doc_id

    def test_eu_doc_id(self):
        doc = {
            "quelle": "eur-lex.europa.eu",
            "typ": "eu_verordnung",
            "abkürzung": "32016R0679",
            "paragraph": "Art. 1",
        }
        doc_id = make_doc_id(doc)
        assert "eur-lex" in doc_id
        assert "eu_verordnung" in doc_id
        assert "32016r0679" in doc_id
        assert "art-_1" in doc_id

    def test_deterministic(self):
        doc = {"quelle": "test.de", "typ": "gesetz", "abkürzung": "ABC", "paragraph": "§ 1"}
        assert make_doc_id(doc) == make_doc_id(doc)

    def test_missing_fields(self):
        doc = {}
        doc_id = make_doc_id(doc)
        assert "unknown" in doc_id
        assert "doc" in doc_id


class TestLegalRAGPipeline:
    @pytest.mark.asyncio
    async def test_init(self):
        pipeline = LegalRAGPipeline()
        assert pipeline.rag is None
        assert pipeline.stats == {"inserted": 0, "failed": 0, "skipped": 0}

    @pytest.mark.asyncio
    async def test_insert_empty_documents(self):
        pipeline = LegalRAGPipeline()
        pipeline.rag = MagicMock()
        stats = await pipeline.insert_documents([])
        assert stats["inserted"] == 0

    @pytest.mark.asyncio
    async def test_insert_one_skips_empty_content(self):
        pipeline = LegalRAGPipeline()
        pipeline.rag = MagicMock()
        pipeline.rag.insert_content_list = AsyncMock()
        stats = await pipeline.insert_documents([{"inhalt": "", "quelle": "test"}])
        assert stats["skipped"] == 1
        pipeline.rag.insert_content_list.assert_not_called()

    @pytest.mark.asyncio
    async def test_insert_one_inserts_content(self):
        pipeline = LegalRAGPipeline()
        pipeline.rag = MagicMock()
        pipeline.rag.insert_content_list = AsyncMock()
        doc = {"inhalt": "Testinhalt", "quelle": "test.de", "typ": "gesetz", "abkürzung": "ABC", "paragraph": "§ 1"}
        stats = await pipeline.insert_documents([doc])
        assert stats["inserted"] == 1
        pipeline.rag.insert_content_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_one_handles_error(self):
        pipeline = LegalRAGPipeline()
        pipeline.rag = MagicMock()
        pipeline.rag.insert_content_list = AsyncMock(side_effect=RuntimeError("Insert failed"))
        doc = {"inhalt": "Testinhalt", "quelle": "test.de", "abkürzung": "ABC"}
        stats = await pipeline.insert_documents([doc])
        assert stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_insert_batch_progress(self):
        pipeline = LegalRAGPipeline()
        pipeline.rag = MagicMock()
        pipeline.rag.insert_content_list = AsyncMock()
        docs = [{"inhalt": f"Doc {i}", "quelle": "test.de"} for i in range(5)]
        stats = await pipeline.insert_documents(docs)
        assert stats["inserted"] == 5
        assert pipeline.rag.insert_content_list.call_count == 5
