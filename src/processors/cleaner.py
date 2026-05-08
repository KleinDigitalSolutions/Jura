"""Text cleaning utilities for legal documents."""
import re

from bs4 import BeautifulSoup


def clean_html(raw_html: str) -> str:
    """Clean HTML to plain text suitable for RAG ingestion.
    - Strips HTML tags
    - Removes navigation, header, footer elements
    - Normalizes whitespace
    - Preserves § symbols and paragraph structure
    - Output encoding: UTF-8
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Remove hidden elements
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
        tag.decompose()

    # Remove common navigation/UI text patterns
    for tag in soup.find_all(string=re.compile(
        r"^(Navigation|Menü|Suche|Impressum|Datenschutz|Kontakt|Startseite|Zurück|Weiter)$",
        re.I
    )):
        tag.extract()

    text = soup.get_text("\n", strip=False)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)

    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Strip trailing/leading whitespace per line while preserving structure
    lines = [l.strip() for l in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def clean_legal_text(text: str) -> str:
    """Additional cleaning specific to legal documents.
    - Preserves § symbols
    - Normalizes legal abbreviations
    - Fixes common OCR/HTML artifacts in legal texts
    """
    # Fix common encoding issues
    text = text.replace("&sect;", "§")
    text = text.replace("&nbsp;", " ")
    text = text.replace("\xa0", " ")

    # Fix broken German quotes
    text = text.replace("â", "„")
    text = text.replace("â", "“")

    # Normalize legal reference patterns
    # "§ 242 BGB" keeps its space, "§242" gets normalized
    text = re.sub(r'§(\d)', r'§ \1', text)

    # Normalize "Abs." / "Absatz"
    text = re.sub(r'\bAbs\.\s*(\d)', r'Absatz \1', text)

    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)

    # Remove consecutive empty lines again
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean(raw_html: str) -> str:
    """Full cleaning pipeline: HTML → clean legal text."""
    text = clean_html(raw_html)
    text = clean_legal_text(text)
    return text
