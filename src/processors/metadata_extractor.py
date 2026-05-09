"""Metadata extraction from legal documents."""
import re
from datetime import datetime
from typing import Optional


RECHTSGEBIET_KEYWORDS: dict[str, list[str]] = {
    "Zivilrecht": [
        "Bürgerliches Gesetzbuch", "BGB", "Kaufvertrag", "Schadensersatz", "Schadenersatz",
        "Mietrecht", "Werkvertrag", "Darlehen", "Bürgschaft", "Eigentum", "Besitz",
        "unerlaubte Handlung", "Vertrag", "Schuldverhältnis", "Sachenrecht", "Erbrecht",
        "Kaution", "Vermieter", "Mieter", "Miete", "Mietsache",
    ],
    "Strafrecht": [
        "Strafgesetzbuch", "StGB", "Straftat", "Angeklagte", "Verurteilung", "Strafe",
        "Diebstahl", "Betrug", "Körperverletzung", "Mord", "Totschlag", "Freiheitsstrafe",
        "Geldstrafe", "Strafprozessordnung", "StPO",
    ],
    "Handelsrecht": [
        "Handelsgesetzbuch", "HGB", "Handelsgesellschaft", "Prokura", "Handelsregister",
        "GmbH", "Aktiengesellschaft", "AktG", "GmbHG", "Kommanditgesellschaft",
    ],
    "Arbeitsrecht": [
        "Arbeitnehmer", "Arbeitgeber", "Kündigung", "Betriebsrat", "Arbeitsvertrag",
        "Tarifvertrag", "KSchG", "Kündigungsschutz", "Betriebsverfassungsgesetz",
    ],
    "Familienrecht": [
        "Ehe", "Scheidung", "Unterhalt", "Sorgerecht", "Umgangsrecht", "Vormundschaft",
    ],
    "Verwaltungsrecht": [
        "Verwaltungsakt", "Behörde", "Baugenehmigung", "VwVfG", "VwGO", "Verwaltungsverfahren",
    ],
    "Steuerrecht": [
        "Steuer", "Finanzamt", "Einkommensteuer", "Umsatzsteuer", "Abgabenordnung",
        "AO", "EStG", "UStG",
    ],
    "Datenschutzrecht": [
        "Datenschutz", "DSGVO", "BDSG", "personenbezogene Daten", "Verarbeitung",
        "Auftragsverarbeiter", "Einwilligung",
    ],
    "Insolvenzrecht": [
        "Insolvenz", "InsO", "Insolvenzverfahren", "Insolvenzverwalter", "Gläubiger",
    ],
    "Urheberrecht": [
        "Urheber", "UrhG", "Urheberrecht", "Lizenz", "Werknutzung",
    ],
}


def extract_rechtsgebiet(text: str) -> str:
    """Determine legal area from text content using keyword matching."""
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for gebiet, keywords in RECHTSGEBIET_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[gebiet] = score

    if not scores:
        return "Sonstiges"

    return max(scores, key=lambda k: scores[k])


def extract_paragraph_references(text: str) -> list[dict[str, str]]:
    """Extract structured paragraph references from text.
    e.g. "§ 242 BGB" or "§§ 242 ff. BGB" → [{"paragraph": "§ 242", "gesetz": "BGB"}]
    """
    refs: list[dict[str, str]] = []

    # Pattern: § 242 BGB, §§ 242, 243 BGB, § 242 Abs. 1 BGB
    pattern = r'(§§?\s*\d+[a-z]?(?:\s*(?:,\s*)?\d+[a-z]?)*(?:\s*(?:Abs\.|Absatz)\s*\d+)?(?:\s*(?:ff\.|f\.))?)\s*([A-Z][A-Za-züöäÜÖÄ]+(?:\s*-\s*[A-Z][A-Za-züöäÜÖÄ]+)?)?'
    matches = re.finditer(pattern, text)

    for match in matches:
        para = match.group(1).strip()
        gesetz = match.group(2) or ""
        refs.append({"paragraph": para, "gesetz": gesetz.strip()})

    return refs


def normalize_date(date_str: Optional[str]) -> str:
    """Normalize a date string to ISO 8601 (YYYY-MM-DD)."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    formats = [
        "%Y-%m-%d",      # ISO
        "%d.%m.%Y",      # German
        "%Y/%m/%d",      # EUR-Lex
        "%d-%m-%Y",      # Alt
        "%Y%m%d",        # Compact
        "%d. %B %Y",     # German long
        "%B %d, %Y",     # English long
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try to extract date components from arbitrary string
    m = re.search(r'(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', date_str)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # Fallback
    return datetime.now().strftime("%Y-%m-%d")


def extract(source_dict: dict) -> dict:
    """Extract and normalize metadata from a scraped document dict.
    Adds: rechtsgebiet, paragraph_refs, normalized dates.
    """
    text_for_analysis = (
        source_dict.get("inhalt", "")
        + " "
        + source_dict.get("titel", "")
        + " "
        + source_dict.get("leitsatz", "")
    )

    result = dict(source_dict)
    result["rechtsgebiet"] = source_dict.get("rechtsgebiet") or extract_rechtsgebiet(text_for_analysis)
    result["paragraph_refs"] = extract_paragraph_references(text_for_analysis)
    result["stand"] = normalize_date(source_dict.get("stand"))

    if "datum" in result:
        result["datum"] = normalize_date(result.get("datum"))

    return result
