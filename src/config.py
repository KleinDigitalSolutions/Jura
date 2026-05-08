"""Central configuration for legal RAG ingestion bot."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# RAG-Anything storage (shared)
RAG_STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", str(PROJECT_ROOT.parent / "RAG-Anything" / "rag_storage"))
RAG_ACTIVATE_SCRIPT = str(PROJECT_ROOT.parent / "RAG-Anything" / "activate.sh")

# Scraping
RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "1.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF: float = float(os.getenv("RETRY_BACKOFF", "2.0"))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

USER_AGENTS: list[str] = [
    "Mozilla/5.0 (compatible; LegalBot/1.0; +https://github.com/legal-rag-ingestion)",
    "Mozilla/5.0 (compatible; LegalRAG/1.0; de-DE)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
]

# Sources
GESETZE_BASE_URL: str = "https://www.gesetze-im-internet.de"
GESETZE_INDEX_URL: str = f"{GESETZE_BASE_URL}/aktuell.html"
GESETZE_RSS_URL: str = f"{GESETZE_BASE_URL}/aktuDienst-rss-feed.xml"

EURLEX_SPARQL_URL: str = "https://publications.europa.eu/webapi/rdf/sparql"

RECHTSPRECHUNG_BASE_URL: str = "https://www.rechtsprechung-im-internet.de"

# Priority laws to always fetch
PRIORITY_LAWS: list[str] = [
    # Verfassungsrecht
    "GG", "BVerfGG",
    # Zivilrecht
    "BGB", "EGBGB", "ZPO", "ZVG", "FamFG", "GVG",
    "WEG", "ProdHaftG",
    # Strafrecht
    "StGB", "StPO", "OWiG", "JGG",
    # Handels- & Gesellschaftsrecht
    "HGB", "GmbHG", "AktG", "UmwG", "PartGG", "GenG",
    "WpHG",
    # Arbeitsrecht
    "KSchG", "BetrVG", "TzBfG", "ArbGG", "AGG", "AÜG",
    "MiLoG", "ArbZG",
    # Verwaltungsrecht
    "VwVfG", "VwGO", "BauGB", "BauNVO", "BImSchG",
    # Steuerrecht
    "AO", "EStG", "UStG", "KStG", "GewStG", "ErbStG",
    # Sozialrecht
    "SGB I", "SGB II", "SGB III", "SGB IV", "SGB V",
    "SGB VI", "SGB VII", "SGB VIII", "SGB IX", "SGB X",
    "SGB XI", "SGB XII", "WoGG",
    # IP / IT / Medien
    "UrhG", "MarkenG", "PatG", "DesignG", "GebrMG",
    "BDSG", "TKG", "TTDSG",
    # Insolvenz
    "InsO",
    # Verkehr
    "StVG", "FZV",
    # Weitere wichtige Gesetze
    "GNotKG", "BNotO", "BRAO", "VermG",
    "LPartG", "VersAusglG", "GewO", "HaftPflG",
]

# Priority EU regulations
PRIORITY_EU_REGULATIONS: list[str] = [
    "32016R0679",  # DSGVO
    "32024R1689",  # AI Act
    "32011R1007",  # Textilkennzeichnung (example)
]

# Batch
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))

# Scheduling
CRON_DAY: str = os.getenv("CRON_DAY", "saturday")
CRON_TIME: str = os.getenv("CRON_TIME", "02:00")

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
