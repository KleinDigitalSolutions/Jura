"""Scraper for German court rulings from rechtsprechung-im-internet.de.

Uses RSS feeds for discovery + XML ZIP downloads for full text.
BGH Zivilsenat prioritized, then BGH Strafsenat, then BVerfG.
"""

import asyncio
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from typing import Optional
from urllib.parse import urljoin
from zipfile import ZipFile

from loguru import logger

from src.config import RECHTSPRECHUNG_BASE_URL
from src.scrapers.base_scraper import BaseScraper

# ---------------------------------------------------------------------------
# Court RSS feed configuration
# ---------------------------------------------------------------------------
COURT_FEEDS: dict[str, str] = {
    "BGH": "bsjrs-bgh",
    "BVerfG": "bsjrs-bverfg",
    "BVerwG": "bsjrs-bverwg",
    "BFH": "bsjrs-bfh",
    "BAG": "bsjrs-bag",
    "BSG": "bsjrs-bsg",
    "BPatG": "bsjrs-bpatg",
}

FEED_URL = f"{RECHTSPRECHUNG_BASE_URL}/jportal/docs/feed/{{feed_id}}.xml"
XML_ZIP_URL = f"{RECHTSPRECHUNG_BASE_URL}/jportal/docs/bsjrs/{{doc_id}}.zip"

# Priority senate ordering (Zivilsenat first per user request)
SENAT_PRIORITY: list[str] = [
    "Zivilsenat",
    "Strafsenat",
    "Anwaltssachen",
    "Kartellsenat",
    "Dienstgericht",
    "Senat",
]

# Rechtsgebiet from § references in <norm> field
NORM_RECHTSGEBIET: dict[str, str] = {
    "BGB": "Zivilrecht",
    "ZPO": "Zivilrecht",
    "HGB": "Handelsrecht",
    "GmbHG": "Handelsrecht",
    "AktG": "Handelsrecht",
    "WEG": "Zivilrecht",
    "StGB": "Strafrecht",
    "StPO": "Strafrecht",
    "OWiG": "Strafrecht",
    "BtmG": "Strafrecht",
    "VwGO": "Verwaltungsrecht",
    "VwVfG": "Verwaltungsrecht",
    "BauGB": "Verwaltungsrecht",
    "AO": "Steuerrecht",
    "EStG": "Steuerrecht",
    "UStG": "Steuerrecht",
    "KSchG": "Arbeitsrecht",
    "BetrVG": "Arbeitsrecht",
    "GG": "Verfassungsrecht",
    "InsO": "Insolvenzrecht",
    "FamFG": "Familienrecht",
    "UrhG": "Urheberrecht",
    "MarkenG": "Markenrecht",
    "PatG": "Patentrecht",
}


class UrteileScraper(BaseScraper):
    """Scrape court rulings via RSS feeds + XML ZIP downloads."""

    source_name = "rechtsprechung"

    def __init__(self, courts=None, max_per_court=0, zivilsenat_only=False):
        """
        Args:
            courts: list of court keys (e.g. ["BGH", "BVerfG"]). Default: BGH + BVerfG.
            max_per_court: max rulings per court (0 = unlimited).
            zivilsenat_only: if True, only BGH Zivilsenat rulings.
        """
        super().__init__()
        self.courts = courts or ["BGH", "BVerfG"]
        self.max_per_court = max_per_court
        self.zivilsenat_only = zivilsenat_only

    # ------------------------------------------------------------------
    # RSS Feed
    # ------------------------------------------------------------------
    async def _fetch_rss_feed(self, court: str) -> list[dict]:
        """Parse RSS feed for a court. Returns list of ruling metadata."""
        feed_id = COURT_FEEDS.get(court)
        if not feed_id:
            logger.warning(f"No feed configured for {court}")
            return []

        url = FEED_URL.format(feed_id=feed_id)
        logger.info(f"Fetching RSS feed: {url}")

        try:
            xml_text = await self.fetch(url, domain="www.rechtsprechung-im-internet.de")
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed for {court}: {e}")
            return []

        return self._parse_rss(xml_text, court)

    def _parse_rss(self, xml_text: str, court: str) -> list[dict]:
        """Parse RSS 2.0 XML into ruling metadata list."""
        items: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"RSS parse error for {court}: {e}")
            return items

        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description")
            pubdate_el = item.find("pubDate")
            guid_el = item.find("guid")

            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            link = link_el.text.strip() if link_el is not None and link_el.text else ""
            description = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
            guid = guid_el.text.strip() if guid_el is not None and guid_el.text else ""

            if not guid:
                continue

            # Parse title: "BGH 2. Zivilsenat, Urteil vom 05.05.2026, II ZR 2/25"
            senate = ""
            doktyp = ""
            aktenzeichen = ""
            entsch_datum = ""

            # Extract senate + decision type
            m = re.match(
                r"(?:BGH|BVerfG|BVerwG|BFH|BAG|BSG|BPatG)\s+(.+?),\s*(Urteil|Beschluss|Gerichtsbescheid)\s+vom\s+(\d{2}\.\d{2}\.\d{4}),\s*(.+)",
                title,
            )
            if m:
                senate = m.group(1).strip()
                doktyp = m.group(2).strip()
                raw_date = m.group(3).strip()
                aktenzeichen = m.group(4).strip()
                try:
                    entsch_datum = datetime.strptime(raw_date, "%d.%m.%Y").strftime("%Y-%m-%d")
                except ValueError:
                    entsch_datum = raw_date

            items.append({
                "title": title,
                "link": link,
                "description": description,
                "doc_id": guid,
                "court": court,
                "senate": senate,
                "doktyp": doktyp,
                "aktenzeichen": aktenzeichen,
                "entsch_datum": entsch_datum,
            })

        logger.info(f"  RSS feed for {court}: {len(items)} rulings")
        return items

    # ------------------------------------------------------------------
    # XML ZIP download + parse
    # ------------------------------------------------------------------
    async def _download_xml_zip(self, doc_id: str) -> Optional[str]:
        """Download and extract XML content from a ruling ZIP."""
        url = XML_ZIP_URL.format(doc_id=doc_id)
        try:
            data = await self.fetch_bytes(url, domain="www.rechtsprechung-im-internet.de")
            with ZipFile(BytesIO(data)) as zf:
                xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
                if not xml_files:
                    logger.warning(f"No XML in {doc_id}.zip")
                    return None
                return zf.read(xml_files[0]).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to download XML for {doc_id}: {e}")
            return None

    def _parse_ruling_xml(self, xml_content: str, feed_item: dict) -> Optional[dict]:
        """Parse ruling XML into structured document dict."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.warning(f"XML parse error for {feed_item.get('doc_id')}: {e}")
            return None

        def _text(tag: str) -> str:
            el = root.find(tag)
            if el is not None and el.text:
                return el.text.strip()
            return ""

        def _html_text(tag: str) -> str:
            """Extract plain text from an element that may contain HTML (leitsatz, tenor, etc.)."""
            el = root.find(tag)
            if el is None:
                return ""
            # Use ET's itertext to get all text nodes, skip <a> anchor names
            parts: list[str] = []
            for elem in el.iter():
                if elem.tag == "a":
                    continue
                if elem.text:
                    t = elem.text.strip()
                    if t:
                        parts.append(t)
                if elem.tail:
                    t = elem.tail.strip()
                    if t:
                        parts.append(t)
            return "\n".join(parts)

        # --- Metadata from XML (more reliable than RSS title parsing) ---
        court = _text("gertyp") or feed_item.get("court", "")
        senate = _text("spruchkoerper") or feed_item.get("senate", "")
        aktenzeichen = _text("aktenzeichen") or feed_item.get("aktenzeichen", "")
        doktyp = _text("doktyp") or feed_item.get("doktyp", "")
        ecli = _text("ecli")
        norm = _text("norm")  # e.g. "§ 37 Abs 2 GmbHG, § 51 Abs 1 GmbHG"

        raw_date = _text("entsch-datum")  # YYYYMMDD
        entsch_datum = feed_item.get("entsch_datum", "")
        if raw_date and len(raw_date) == 8:
            try:
                entsch_datum = datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pass

        # --- Text sections ---
        leitsatz = _html_text("leitsatz")
        tenor = _html_text("tenor")
        tatbestand = _html_text("tatbestand")
        entscheidungsgruende = _html_text("entscheidungsgruende") or _html_text("gruende")
        sonstosatz = _html_text("sonstosatz")

        # Fallback: RSS <description> if XML leitsatz is empty
        if not leitsatz:
            rss_desc = feed_item.get("description", "")
            # Filter out boilerplate "Hinweis der Dokumentationsstelle…"
            if rss_desc and not rss_desc.startswith("Hinweis der Dokumentationsstelle"):
                leitsatz = rss_desc

        # Build volltext: all sections concatenated
        volltext_parts: list[str] = []
        if leitsatz:
            volltext_parts.append(f"LEITSATZ:\n{leitsatz}")
        if tenor:
            volltext_parts.append(f"TENOR:\n{tenor}")
        if tatbestand:
            volltext_parts.append(f"TATBESTAND:\n{tatbestand}")
        if entscheidungsgruende:
            volltext_parts.append(f"ENTSCHEIDUNGSGRÜNDE:\n{entscheidungsgruende}")
        if sonstosatz:
            volltext_parts.append(sonstosatz)
        volltext = "\n\n".join(volltext_parts)

        if not volltext.strip():
            logger.warning(f"Empty volltext for {doc_id}")
            return None

        # --- Rechtsgebiet ---
        rechtsgebiet = self._deduce_rechtsgebiet(norm, leitsatz + " " + volltext[:2000], senate)

        doc_id = feed_item.get("doc_id", "")
        vorinstanz = _html_text("vorinstanz")

        return {
            "typ": "urteil",
            "gericht": court,
            "senate": senate,
            "aktenzeichen": aktenzeichen,
            "doktyp": doktyp,
            "ecli": ecli,
            "datum": entsch_datum,
            "norm": norm,
            "vorinstanz": vorinstanz,
            "rechtsgebiet": rechtsgebiet,
            "leitsatz": leitsatz,
            "volltext": volltext,
            "url": feed_item.get("link", ""),
            "quelle": "rechtsprechung-im-internet.de",
            "doc_id": doc_id,
        }

    # ------------------------------------------------------------------
    # Rechtsgebiet
    # ------------------------------------------------------------------
    def _deduce_rechtsgebiet(self, norm_str: str, text: str, senate: str = "") -> str:
        """Deduce legal area from <norm> references, senate name, then text keywords."""
        # 1) From norm references (most reliable)
        if norm_str:
            for abk, gebiet in NORM_RECHTSGEBIET.items():
                if abk in norm_str:
                    return gebiet

        # 2) From senate name
        if senate:
            s = senate.lower()
            if "zivil" in s:
                return "Zivilrecht"
            if "straf" in s:
                return "Strafrecht"
            if "anwalt" in s:
                return "Berufsrecht"
            if "kartell" in s:
                return "Kartellrecht"

        # 3) Text keyword fallback
        keywords: dict[str, list[str]] = {
            "Zivilrecht": ["Kaufvertrag", "Schadensersatz", "Mietrecht", "BGB", "Zivil",
                          "Werkvertrag", "Darlehen", "Bürgschaft", "Eigentum", "Besitz",
                          "Zivilsenat", "Mangel", "Nacherfüllung", "Gewährleistung"],
            "Strafrecht": ["StGB", "Straftat", "Angeklagte", "Verurteilung", "Diebstahl",
                          "Betrug", "Körperverletzung", "Strafsenat", "Strafkammer"],
            "Arbeitsrecht": ["Arbeitnehmer", "Arbeitgeber", "Kündigung", "Betriebsrat",
                            "Arbeitsvertrag", "Tarifvertrag", "KSchG", "Arbeitsgericht"],
            "Verwaltungsrecht": ["Verwaltungsakt", "Behörde", "Baugenehmigung", "VwVfG",
                                "VwGO", "BImSchG"],
            "Steuerrecht": ["Steuer", "Finanzamt", "Einkommensteuer", "Umsatzsteuer", "AO"],
            "Handelsrecht": ["Handelsgesellschaft", "GmbH", "Aktiengesellschaft", "HGB",
                            "Prokura", "GmbHG", "AktG"],
            "Familienrecht": ["Ehe", "Scheidung", "Unterhalt", "Sorgerecht", "Umgangsrecht",
                             "FamFG", "Familien"],
            "Verfassungsrecht": ["Verfassungsbeschwerde", "Grundgesetz", "GG", "BVerfG",
                                "Grundrecht", "Art. "],
        }

        text_lower = text.lower()
        for gebiet, words in keywords.items():
            if any(w.lower() in text_lower for w in words):
                return gebiet

        return "Sonstiges"

    # ------------------------------------------------------------------
    # Senate priority sort
    # ------------------------------------------------------------------
    def _senate_rank(self, senate: str) -> int:
        """Lower = higher priority. Zivilsenat = 0, Strafsenat = 1, etc."""
        for i, keyword in enumerate(SENAT_PRIORITY):
            if keyword.lower() in senate.lower():
                return i
        return len(SENAT_PRIORITY)

    # ------------------------------------------------------------------
    # Main scrape
    # ------------------------------------------------------------------
    async def scrape(self) -> list[dict]:
        """Scrape rulings from configured courts via RSS + XML ZIP."""
        all_rulings: list[dict] = []

        for court in self.courts:
            logger.info(f"Scraping {court} rulings...")

            # 1) Fetch RSS feed
            feed_items = await self._fetch_rss_feed(court)
            if not feed_items:
                logger.warning(f"No feed items for {court}")
                continue

            # 2) Filter + sort: Zivilsenat first
            if self.zivilsenat_only:
                feed_items = [f for f in feed_items if "zivilsenat" in f.get("senate", "").lower()]
                logger.info(f"  Filtered to {len(feed_items)} Zivilsenat rulings")

            feed_items.sort(key=lambda f: self._senate_rank(f.get("senate", "")))

            # 3) Limit
            if self.max_per_court > 0:
                feed_items = feed_items[:self.max_per_court]

            logger.info(f"  Processing {len(feed_items)} rulings for {court}")

            # 4) Download + parse each ruling
            for i, item in enumerate(feed_items):
                doc_id = item["doc_id"]
                logger.info(f"  [{i+1}/{len(feed_items)}] {item.get('aktenzeichen')} — {item.get('senate')} ({doc_id})")

                xml_content = await self._download_xml_zip(doc_id)
                if not xml_content:
                    continue

                ruling = self._parse_ruling_xml(xml_content, item)
                if ruling:
                    all_rulings.append(ruling)
                    logger.info(f"    → {len(ruling['volltext'])} chars, {ruling['rechtsgebiet']}")

                await asyncio.sleep(1.0)  # rate limit

        logger.info(f"Total rulings scraped: {len(all_rulings)}")
        return all_rulings
