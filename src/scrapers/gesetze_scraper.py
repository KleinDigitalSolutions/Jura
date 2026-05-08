"""Scraper for gesetze-im-internet.de — German federal laws."""
import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from typing import Optional
from urllib.parse import urljoin
from zipfile import ZipFile

from bs4 import BeautifulSoup
from loguru import logger

from src.config import GESETZE_BASE_URL, GESETZE_INDEX_URL, PRIORITY_LAWS
from src.scrapers.base_scraper import BaseScraper


LAW_SLUG_MAP: dict[str, str] = {
    # Verfassung
    "GG": "gg",
    # Zivilrecht
    "BGB": "bgb",
    "ZPO": "zpo",
    "ZVG": "zvg",
    "FamFG": "famfg",
    "GVG": "gvg",
    "ProdHaftG": "prodhaftg",
    # Strafrecht
    "StGB": "stgb",
    "StPO": "stpo",
    "OWiG": "owig_1968",
    "JGG": "jgg",
    # Handels- & Gesellschaftsrecht
    "HGB": "hgb",
    "GmbHG": "gmbhg",
    "AktG": "aktg",
    "PartGG": "partgg",
    "GenG": "geng",
    # Arbeitsrecht
    "KSchG": "kschg",
    "BetrVG": "betrvg",
    "TzBfG": "tzbfg",
    "ArbGG": "arbgg",
    "AGG": "agg",
    "MiLoG": "milog",
    "ArbZG": "arbzg",
    # Verwaltungsrecht
    "VwVfG": "vwvfg",
    "VwGO": "vwgo",
    "BImSchG": "bimschg",
    # Steuerrecht
    "AO": "ao_1977",
    "EStG": "estg",
    "GewStG": "gewstg",
    # Sozialrecht
    "SGB I": "sgb_1",
    "SGB II": "sgb_2",
    "SGB III": "sgb_3",
    "SGB IV": "sgb_4",
    "SGB V": "sgb_5",
    "SGB VI": "sgb_6",
    "SGB VII": "sgb_7",
    "SGB VIII": "sgb_8",
    "SGB IX": "sgb_9_2018",
    "SGB X": "sgb_10",
    "SGB XI": "sgb_11",
    "SGB XII": "sgb_12",
    # IP / IT / Medien
    "UrhG": "urhg",
    "MarkenG": "markeng",
    "PatG": "patg",
    "GebrMG": "gebrmg",
    "TTDSG": "ttdsg",
    "BDSG": "bdsg_2018",
    # Insolvenz
    "InsO": "inso",
    # Verkehr
    "StVG": "stvg",
    # EGBGB, WEG, BGB-InfoV, UmwG, HRefG, AÜG, BauGB, UStG, KStG,
    # DesignG, TKG, TMG, FZV — slug auto-discovered from Teilliste index
}


class GesetzeScraper(BaseScraper):
    """Scrape gesetze-im-internet.de for structured German federal law."""

    source_name = "gesetze-im-internet"

    async def _discover_laws(self) -> list[dict[str, str]]:
        """Parse Teilliste index pages to discover all available laws.
        Returns list of {abkuerzung, titel, url_slug}."""
        laws: list[dict[str, str]] = []

        html = await self.fetch(GESETZE_INDEX_URL, domain="www.gesetze-im-internet.de")
        soup = BeautifulSoup(html, "lxml")

        teilliste_links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "Teilliste_" in href:
                teilliste_links.append(urljoin(GESETZE_BASE_URL, href))

        logger.info(f"Found {len(teilliste_links)} Teilliste pages")

        for tl_url in teilliste_links:
            try:
                tl_html = await self.fetch(tl_url, domain="www.gesetze-im-internet.de")
                tl_soup = BeautifulSoup(tl_html, "lxml")

                for a in tl_soup.find_all("a", href=True):
                    href = a["href"].strip()
                    text = a.get_text(strip=True)
                    if href.endswith("/index.html") and text:
                        slug = href.rstrip("/index.html").rstrip("/")
                        if slug.startswith("./"):
                            slug = slug[2:]
                        laws.append({
                            "abkuerzung": text.split("(")[0].strip() if "(" in text else text,
                            "titel": text,
                            "url_slug": slug,
                        })
            except Exception as e:
                logger.warning(f"Failed to parse Teilliste {tl_url}: {e}")

        logger.info(f"Discovered {len(laws)} laws total")
        return laws

    async def _download_xml_zip(self, law_slug: str) -> Optional[str]:
        """Download and extract XML content for a law from its xml.zip."""
        zip_url = f"{GESETZE_BASE_URL}/{law_slug}/xml.zip"
        try:
            data = await self.fetch_bytes(zip_url, domain="www.gesetze-im-internet.de")
            with ZipFile(BytesIO(data)) as zf:
                xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
                if not xml_files:
                    logger.warning(f"No XML found in {zip_url}")
                    return None
                return zf.read(xml_files[0]).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to download XML for {law_slug}: {e}")
            return None

    def _get_law_stand_date(self, soup: BeautifulSoup, law_slug: str) -> str:
        """Extract the 'Stand' date from the law's index page."""
        try:
            text = soup.get_text()
            m = re.search(r'Stand:\s*(?:Zuletzt geändert durch[^)]+\)\s*)?(\d{2}\.\d{2}\.\d{4})', text)
            if m:
                return datetime.strptime(m.group(1), "%d.%m.%Y").strftime("%Y-%m-%d")
        except Exception:
            pass
        return datetime.now().strftime("%Y-%m-%d")

    async def _scrape_law(self, law_slug: str) -> list[dict]:
        """Scrape a single law — download XML, extract paragraphs."""
        documents: list[dict] = []

        # Get index page for metadata
        index_url = f"{GESETZE_BASE_URL}/{law_slug}/index.html"
        try:
            index_html = await self.fetch(index_url, domain="www.gesetze-im-internet.de")
            index_soup = BeautifulSoup(index_html, "lxml")
        except Exception:
            logger.warning(f"Cannot fetch index for {law_slug}, skipping")
            return documents

        law_title_raw = index_soup.find("title")
        law_title = law_title_raw.get_text(strip=True).replace(" - dejure.org", "") if law_title_raw else law_slug
        stand = self._get_law_stand_date(index_soup, law_slug)

        # Parse abbreviation from slug map
        abk = law_slug.upper()
        for short, slug in LAW_SLUG_MAP.items():
            if slug == law_slug:
                abk = short
                break

        # Attempt XML download (preferred)
        xml_content = await self._download_xml_zip(law_slug)
        if xml_content:
            documents = self._parse_xml(xml_content, abk, law_title, law_slug, stand)

        # Fallback: HTML paragraph pages
        if not documents:
            logger.info(f"XML not available for {law_slug}, falling back to HTML paragraph pages")
            try:
                documents = await self._scrape_law_html(index_soup, law_slug, abk, law_title, stand)
            except Exception as e:
                logger.error(f"HTML fallback also failed for {law_slug}: {e}")

        return documents

    @staticmethod
    def _local_tag(elem) -> str:
        """Get tag name without namespace."""
        return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    def _parse_xml(self, xml_content: str, abk: str, law_title: str, law_slug: str, stand: str) -> list[dict]:
        """Parse XML content into paragraph-level documents.

        gesetze-im-internet XML structure:
          <norm doknr="...">
            <metadaten>
              <jurabk>BGB</jurabk>
              <enbez>§ 242</enbez>
              <titel>Leistung nach Treu und Glauben</titel>
            </metadaten>
            <textdaten>
              <text><P>...</P></text>
              <fussnoten>...</fussnoten>
            </textdaten>
          </norm>
        """
        documents: list[dict] = []
        try:
            root = ET.fromstring(xml_content)

            for norm in root.iter():
                if self._local_tag(norm) not in ("norm", "paragraf", "paragraph", "article"):
                    continue

                # Extract from metadaten child (enbez, titel)
                para_id = ""
                para_title = ""
                for child in norm:
                    if self._local_tag(child) == "metadaten":
                        for meta in child:
                            tag = self._local_tag(meta)
                            if tag == "enbez" and meta.text:
                                para_id = meta.text.strip()
                            elif tag == "titel" and meta.text:
                                para_title = meta.text.strip()
                        break

                # Fallback: use doknr attribute for norms without enbez (preambles etc.)
                if not para_id:
                    para_id = norm.get("doknr", "")

                # Collect text from textdaten child only (skip metadaten noise)
                text_parts: list[str] = []
                for child in norm:
                    if self._local_tag(child) == "textdaten":
                        for elem in child.iter():
                            if elem.text:
                                t = elem.text.strip()
                                if t:
                                    text_parts.append(t)
                        break

                inhalt = "\n".join(text_parts)

                if inhalt:
                    documents.append({
                        "typ": "gesetz",
                        "abkürzung": abk,
                        "titel": law_title,
                        "paragraph": para_id,
                        "paragraph_titel": para_title,
                        "inhalt": inhalt,
                        "url": f"{GESETZE_BASE_URL}/{law_slug}/",
                        "stand": stand,
                        "quelle": "gesetze-im-internet.de",
                    })
        except ET.ParseError as e:
            logger.error(f"XML parse error for {abk}: {e}")
        except Exception as e:
            logger.error(f"XML processing error for {abk}: {e}")

        return documents

    async def _scrape_law_html(
        self, index_soup: BeautifulSoup, law_slug: str, abk: str, law_title: str, stand: str
    ) -> list[dict]:
        """Fallback: scrape individual paragraph HTML pages."""
        documents: list[dict] = []
        para_links: set[str] = set()

        for a in index_soup.find_all("a", href=True):
            href = a["href"]
            if re.match(r"__\d+\.html", href):
                para_links.add(urljoin(f"{GESETZE_BASE_URL}/{law_slug}/", href))

        if not para_links:
            logger.warning(f"No paragraph links found for {law_slug}")
            return documents

        # Limit concurrency to 5 to avoid server overload
        semaphore = asyncio.Semaphore(5)

        async def fetch_paragraph(url: str) -> Optional[dict]:
            async with semaphore:
                try:
                    html = await self.fetch(url, domain="www.gesetze-im-internet.de")
                    soup = BeautifulSoup(html, "lxml")

                    para_num = ""
                    h = soup.find("h1") or soup.find("h2") or soup.find("h3")
                    if h:
                        m = re.match(r"§\s*\d+[a-z]?", h.get_text(strip=True))
                        if m:
                            para_num = m.group(0)

                    para_title = h.get_text(strip=True).replace(para_num, "").strip() if h else ""
                    content_div = soup.find("div", {"id": "content"}) or soup.find("main") or soup.find("body")
                    inhalt = content_div.get_text("\n", strip=True) if content_div else ""
                    inhalt = re.sub(r'\n{3,}', '\n\n', inhalt)

                    return {
                        "typ": "gesetz",
                        "abkürzung": abk,
                        "titel": law_title,
                        "paragraph": para_num,
                        "paragraph_titel": para_title,
                        "inhalt": inhalt,
                        "url": url,
                        "stand": stand,
                        "quelle": "gesetze-im-internet.de",
                    }
                except Exception as e:
                    logger.warning(f"Failed to fetch paragraph {url}: {e}")
                    return None

        tasks = [fetch_paragraph(url) for url in sorted(para_links)]
        results = await asyncio.gather(*tasks)
        documents = [r for r in results if r is not None]
        return documents

    async def scrape(self) -> list[dict]:
        """Main entry point: scrape all priority laws."""
        all_docs: list[dict] = []

        laws = await self._discover_laws()

        # Build slug lookup: LAW_SLUG_MAP has priority, discovered laws fill gaps
        law_by_abk: dict[str, str] = {}

        # First: seed from LAW_SLUG_MAP (authoritative)
        for short, slug in LAW_SLUG_MAP.items():
            law_by_abk[short.upper()] = slug

        # Second: supplement from discovered laws (skip translations)
        for law in laws:
            slug = law["url_slug"].lower()
            if "englisch_" in slug or "translations" in slug:
                continue
            abk = law["abkuerzung"].upper()
            if abk not in law_by_abk:
                law_by_abk[abk] = slug

        # Scrape priority laws
        for priority_abk in PRIORITY_LAWS:
            slug = law_by_abk.get(priority_abk.upper())
            if not slug:
                logger.warning(f"No slug found for {priority_abk}, skipping")
                continue

            logger.info(f"Scraping {priority_abk} ({slug})...")
            try:
                docs = await self._scrape_law(slug)
                all_docs.extend(docs)
                logger.info(f"  → {len(docs)} paragraphs extracted")
            except Exception as e:
                logger.error(f"Failed to scrape {priority_abk}: {e}")

        return all_docs
