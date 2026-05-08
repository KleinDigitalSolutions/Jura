"""Tests for legal document scrapers."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scrapers.base_scraper import RateLimiter


class TestRateLimiter:
    def test_wait_first_request_no_delay(self):
        rl = RateLimiter()
        result = asyncio.run(rl.wait("example.com"))
        assert result is None

    def test_rate_limit_enforced(self):
        rl = RateLimiter()
        asyncio.run(rl.wait("example.com"))

        # Second call immediately — rate limiter should delay ~1s
        elapsed = asyncio.run(self._measure_wait(rl, "example.com"))
        assert elapsed >= 0.9  # 1 req/s limit enforced

    @staticmethod
    async def _measure_wait(rl, domain):
        import time
        start = time.monotonic()
        await rl.wait(domain)
        return time.monotonic() - start


class TestGesetzeScraper:
    def test_init(self):
        from src.scrapers.gesetze_scraper import GesetzeScraper
        scraper = GesetzeScraper()
        assert scraper is not None

    @pytest.mark.asyncio
    async def test_discover_laws_returns_list(self):
        from src.scrapers.gesetze_scraper import GesetzeScraper
        scraper = GesetzeScraper()
        with patch.object(scraper, 'fetch', AsyncMock(return_value='<html><body></body></html>')):
            laws = await scraper._discover_laws()
            assert isinstance(laws, list)

    @pytest.mark.asyncio
    async def test_scrape_returns_list(self):
        from src.scrapers.gesetze_scraper import GesetzeScraper
        scraper = GesetzeScraper()
        with patch.object(scraper, '_discover_laws', AsyncMock(return_value=[
            {"url_slug": "bgb", "abkuerzung": "BGB", "title": "Bürgerliches Gesetzbuch"}
        ])):
            with patch.object(scraper, '_download_xml_zip', AsyncMock(return_value=None)):
                with patch.object(scraper, '_scrape_law_html', AsyncMock(return_value=[{
                    "typ": "gesetz", "abkürzung": "BGB", "titel": "Test",
                    "paragraph": "§ 1", "inhalt": "Testinhalt",
                    "url": "https://example.com", "stand": "2024-01-01",
                    "quelle": "gesetze-im-internet.de"
                }])):
                    docs = await scraper.scrape()
                    assert isinstance(docs, list)
                    if len(docs) > 0:
                        assert docs[0]["abkürzung"] == "BGB"


class TestEurLexScraper:
    def test_init(self):
        from src.scrapers.eurlex_scraper import EurLexScraper
        scraper = EurLexScraper()
        assert scraper is not None

    @pytest.mark.asyncio
    async def test_scrape_returns_list_on_sparql_failure(self):
        from src.scrapers.eurlex_scraper import EurLexScraper
        scraper = EurLexScraper()
        with patch.object(scraper, '_sparql_query', AsyncMock(return_value=[])):
            docs = await scraper.scrape()
            assert isinstance(docs, list)
            assert len(docs) == 0


class TestUrteileScraper:
    def test_init(self):
        from src.scrapers.urteile_scraper import UrteileScraper
        scraper = UrteileScraper()
        assert scraper is not None

    def test_deduce_rechtsgebiet(self):
        from src.scrapers.urteile_scraper import UrteileScraper
        scraper = UrteileScraper()
        result = scraper._deduce_rechtsgebiet("Schadensersatz nach § 823 BGB")
        assert result == "Zivilrecht"

        result = scraper._deduce_rechtsgebiet("Diebstahl gemäß § 242 StGB")
        assert result == "Strafrecht"

        result = scraper._deduce_rechtsgebiet("Unbekannter Rechtstext")
        assert result == "Sonstiges"

    @pytest.mark.asyncio
    async def test_scrape_returns_list(self):
        from src.scrapers.urteile_scraper import UrteileScraper
        scraper = UrteileScraper()
        with patch.object(scraper, '_fetch_court_page', AsyncMock(return_value='<html></html>')):
            docs = await scraper.scrape()
            assert isinstance(docs, list)
