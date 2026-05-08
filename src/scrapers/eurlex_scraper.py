"""Scraper for EUR-Lex EU law via SPARQL endpoint."""
import asyncio
import re
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from loguru import logger

from src.config import EURLEX_SPARQL_URL, PRIORITY_EU_REGULATIONS
from src.scrapers.base_scraper import BaseScraper


class EurLexScraper(BaseScraper):
    """Query EUR-Lex via SPARQL for structured EU legal documents in German."""

    source_name = "eurlex"

    SPARQL_QUERY_TEMPLATE = """
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?celex ?title ?date ?type ?html_url ?pdf_url
    WHERE {{
        ?work cdm:resource_legal_id_celex ?celex .
        ?work cdm:work_date_document ?date .
        ?work cdm:work_has_resource_type ?type .
        OPTIONAL {{ ?expr cdm:expression_belongs_to_work ?work .
                   ?expr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/DEU> .
                   ?expr cdm:expression_title ?title .
        }}
        OPTIONAL {{ ?manif cdm:manifestation_belongs_to_work ?work .
                   ?manif cdm:manifestation_type ?mtype .
                   OPTIONAL {{ ?manif cdm:manifestation_has_legal_format "html"^^xsd:string ;
                                        cdm:manifestation_internet_url ?html_url }}
                   OPTIONAL {{ ?manif cdm:manifestation_has_legal_format "pdf"^^xsd:string ;
                                        cdm:manifestation_internet_url ?pdf_url }}
        }}
        FILTER (?celex IN ({celex_values}))
        FILTER (LANG(?title) = "de" || LANG(?title) = "")
    }}
    ORDER BY ?celex
    """

    async def _sparql_query(self, query: str) -> list[dict[str, str]]:
        """Execute a SPARQL query against the EUR-Lex endpoint."""
        url = f"{EURLEX_SPARQL_URL}?query={quote(query)}"
        try:
            text = await self.fetch(url, domain="publications.europa.eu")
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return []

        rows: list[dict[str, str]] = []
        # Parse SPARQL XML result
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(text)
            ns = {"sp": "http://www.w3.org/2005/sparql-results#"}
            for result in root.findall(".//sp:result", ns):
                row: dict[str, str] = {}
                for binding in result.findall("sp:binding", ns):
                    name = binding.get("name", "")
                    value_elem = binding.find("sp:literal", ns) or binding.find("sp:uri", ns)
                    row[name] = value_elem.text if value_elem is not None else ""
                if row:
                    rows.append(row)
        except ET.ParseError as e:
            logger.error(f"SPARQL result parse error: {e}")
        return rows

    async def _fetch_document_text(self, celex: str) -> Optional[str]:
        """Fetch the full text of a EUR-Lex document via the Cellar REST API."""
        # EUR-Lex content negotiation: request HTML DE version
        url = f"https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:{celex}"
        try:
            html = await self.fetch(url, domain="eur-lex.europa.eu")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            # Remove navigation elements
            for nav in soup.find_all(["nav", "header", "footer", "script", "style"]):
                nav.decompose()
            body = soup.find("body")
            return body.get_text("\n", strip=True) if body else html
        except Exception as e:
            logger.warning(f"Failed to fetch text for {celex}: {e}")
            return None

    async def scrape(self) -> list[dict]:
        """Scrape priority EU regulations."""
        all_docs: list[dict] = []

        celex_list = ", ".join(f'"{c}"' for c in PRIORITY_EU_REGULATIONS)
        query = self.SPARQL_QUERY_TEMPLATE.format(celex_values=celex_list)

        metadata_rows = await self._sparql_query(query)
        logger.info(f"EUR-Lex SPARQL returned {len(metadata_rows)} results")

        for row in metadata_rows:
            celex = row.get("celex", "")
            title = row.get("title", celex)
            date_val = row.get("date", "")

            # Normalize date
            try:
                date_iso = datetime.strptime(date_val, "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                date_iso = date_val or datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching full text for {celex}: {title[:80]}...")
            full_text = await self._fetch_document_text(celex)

            if not full_text:
                continue

            # Attempt to split into articles/paragraphs
            articles = self._split_into_articles(full_text, celex)

            for i, article in enumerate(articles):
                all_docs.append({
                    "typ": "eu_verordnung",
                    "abkürzung": celex,
                    "titel": title,
                    "paragraph": f"Art. {i + 1}" if len(articles) > 1 else "",
                    "paragraph_titel": "",
                    "inhalt": article.strip(),
                    "url": f"https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:{celex}",
                    "stand": date_iso,
                    "quelle": "eur-lex.europa.eu",
                })

            logger.info(f"  → {len(articles)} articles extracted")
            # Small delay between documents
            await asyncio.sleep(0.5)

        return all_docs

    def _split_into_articles(self, text: str, celex: str) -> list[str]:
        """Split regulation text into individual articles."""
        # Try to find article markers: "Artikel 1", "Art. 1", etc.
        article_pattern = re.compile(r'(?:^|\n)\s*(?:Artikel|Art\.)\s+\d+', re.IGNORECASE)
        splits = list(article_pattern.finditer(text))

        if len(splits) < 2:
            # No article structure found, return as single chunk
            return [text]

        articles: list[str] = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            articles.append(text[start:end].strip())

        return articles
