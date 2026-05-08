"""Tests for legal document processors."""
import pytest

from src.processors.cleaner import clean, clean_html, clean_legal_text
from src.processors.metadata_extractor import (
    extract_rechtsgebiet,
    extract_paragraph_references,
    normalize_date,
    extract,
)
from src.processors.chunker import (
    estimate_tokens,
    chunk_gesetz,
    chunk_urteil,
    chunk_document,
    _build_meta_prefix,
)


# ── Cleaner ──────────────────────────────────────────────

class TestCleaner:
    def test_clean_html_strips_tags(self):
        result = clean_html("<p>Hello <b>World</b></p>")
        assert "Hello" in result
        assert "World" in result
        assert "<b>" not in result
        assert "<p>" not in result

    def test_clean_html_strips_nav_header_footer(self):
        html = "<html><body><nav>Skip</nav><header>Head</header><main>Content</main><footer>Foot</footer></body></html>"
        result = clean_html(html)
        assert "Content" in result
        assert "Skip" not in result
        assert "Head" not in result
        assert "Foot" not in result

    def test_clean_legal_text_preserves_paragraph(self):
        result = clean_legal_text("§242 BGB regelt Treu und Glauben.")
        assert "§ 242" in result
        assert "BGB" in result

    def test_clean_legal_text_normalizes_quotes(self):
        result = clean_legal_text('Der „Vertrag" wurde „angenommen"')
        assert "Vertrag" in result
        assert "angenommen" in result

    def test_clean_pipeline(self):
        result = clean("<p>§1 BGB — Die Rechtsfähigkeit.</p>")
        assert "§ 1" in result
        assert "Rechtsfähigkeit" in result


# ── Metadata Extractor ──────────────────────────────────

class TestMetadataExtractor:
    def test_extract_rechtsgebiet_zivilrecht(self):
        text = "Schadensersatz nach § 823 BGB wegen unerlaubter Handlung"
        result = extract_rechtsgebiet(text)
        assert result == "Zivilrecht"

    def test_extract_rechtsgebiet_strafrecht(self):
        text = "Der Angeklagte wurde wegen Diebstahls nach § 242 StGB verurteilt"
        result = extract_rechtsgebiet(text)
        assert result == "Strafrecht"

    def test_extract_rechtsgebiet_datenschutz(self):
        text = "Verarbeitung personenbezogener Daten gemäß DSGVO"
        result = extract_rechtsgebiet(text)
        assert result == "Datenschutzrecht"

    def test_extract_rechtsgebiet_empty_returns_sonstiges(self):
        result = extract_rechtsgebiet("")
        assert result == "Sonstiges"

    def test_extract_paragraph_references_single(self):
        refs = extract_paragraph_references("Gemäß § 242 BGB gilt Treu und Glauben.")
        assert len(refs) >= 1
        assert any(r["paragraph"] == "§ 242" for r in refs)

    def test_extract_paragraph_references_double(self):
        refs = extract_paragraph_references("Siehe §§ 242, 259 BGB.")
        assert len(refs) >= 1

    def test_extract_paragraph_references_none(self):
        refs = extract_paragraph_references("Keine Paragraphen hier.")
        assert refs == []

    def test_normalize_date_german(self):
        result = normalize_date("24.07.2024")
        assert result == "2024-07-24"

    def test_normalize_date_iso(self):
        result = normalize_date("2024-07-24")
        assert result == "2024-07-24"

    def test_normalize_date_eurlex(self):
        result = normalize_date("2024/07/24")
        assert result == "2024-07-24"

    def test_normalize_date_none(self):
        result = normalize_date(None)
        assert len(result) == 10  # YYYY-MM-DD

    def test_extract_adds_fields(self):
        doc = {"inhalt": "Schadensersatz § 823 BGB", "titel": "Test"}
        result = extract(doc)
        assert "rechtsgebiet" in result
        assert "paragraph_refs" in result
        assert "stand" in result


# ── Chunker ─────────────────────────────────────────────

class TestEstimateTokens:
    def test_german_text(self):
        tokens = estimate_tokens("Die Rechtsfähigkeit des Menschen beginnt mit der Vollendung der Geburt.")
        assert tokens > 0
        assert tokens < 100


class TestBuildMetaPrefix:
    def test_gesetz_prefix(self):
        doc = {"typ": "gesetz", "abkürzung": "BGB", "paragraph": "§ 1"}
        prefix = _build_meta_prefix(doc)
        assert "Typ: gesetz" in prefix
        assert "Gesetz: BGB" in prefix
        assert "Paragraph: § 1" in prefix

    def test_urteil_prefix(self):
        doc = {"typ": "urteil", "gericht": "BGH", "aktenzeichen": "VI ZR 1/24"}
        prefix = _build_meta_prefix(doc)
        assert "Gericht: BGH" in prefix
        assert "Aktenzeichen: VI ZR 1/24" in prefix

    def test_empty_doc(self):
        prefix = _build_meta_prefix({})
        assert prefix == ""


class TestChunkGesetz:
    def test_short_paragraph_single_chunk(self):
        doc = {"inhalt": "Kurzer Text.", "typ": "gesetz", "abkürzung": "BGB", "paragraph": "§ 1"}
        chunks = chunk_gesetz(doc)
        assert len(chunks) == 1
        assert "Typ:" in chunks[0]["inhalt"]

    def test_empty_inhalt_still_chunks(self):
        doc = {"inhalt": "", "typ": "gesetz", "abkürzung": "BGB", "paragraph": "§ 1"}
        chunks = chunk_gesetz(doc)
        assert len(chunks) == 1


class TestChunkUrteil:
    def test_leitsatz_own_chunk(self):
        doc = {
            "leitsatz": "Wichtiger Leitsatz.",
            "volltext": "Tatbestand\nSachverhalt...\nEntscheidungsgründe\nBegründung...",
            "typ": "urteil", "gericht": "BGH", "datum": "2024-01-01",
        }
        chunks = chunk_urteil(doc)
        leitsatz_chunks = [c for c in chunks if "LEITSATZ" in c["inhalt"]]
        assert len(leitsatz_chunks) == 1

    def test_no_leitsatz_no_leitsatz_chunk(self):
        doc = {"leitsatz": "", "volltext": "Nur Volltext.", "typ": "urteil"}
        chunks = chunk_urteil(doc)
        leitsatz_chunks = [c for c in chunks if "LEITSATZ" in c["inhalt"]]
        assert len(leitsatz_chunks) == 0


class TestChunkDocument:
    def test_routes_gesetz(self):
        doc = {"typ": "gesetz", "abkürzung": "BGB", "paragraph": "§ 1", "inhalt": "Test"}
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_routes_urteil(self):
        doc = {"typ": "urteil", "gericht": "BGH", "leitsatz": "Test.", "volltext": "Text."}
        chunks = chunk_document(doc)
        assert len(chunks) >= 1

    def test_routes_unknown(self):
        doc = {"typ": "unknown", "inhalt": "Test"}
        chunks = chunk_document(doc)
        assert len(chunks) == 1
