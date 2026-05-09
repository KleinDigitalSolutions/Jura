"""Query classification into Rechtsgebiete using keyword matching."""
import logging
import re
from typing import Optional

from src.processors.metadata_extractor import RECHTSGEBIET_KEYWORDS

logger = logging.getLogger(__name__)

# Domains that share ambiguous keywords (e.g. "Kündigung" in both)
AMBIGUOUS_DOMAINS = {
    "Arbeitsrecht", "Zivilrecht",  # Kündigung, Vertrag
}

# Keywords that appear in multiple domains — used for disambiguation scoring.
# When a query matches keywords from competing domains, the domain with
# more *unique* (non-shared) keyword hits wins.
SHARED_KEYWORDS: dict[str, set[str]] = {
    "Kündigung": {"Arbeitsrecht", "Zivilrecht"},
    "Vertrag": {"Arbeitsrecht", "Zivilrecht", "Handelsrecht"},
    "Schadensersatz": {"Zivilrecht", "Strafrecht"},
    "Frist": {"Zivilrecht", "Arbeitsrecht", "Steuerrecht", "Strafrecht"},
}


class LegalQueryClassifier:
    """Classifies a user query into a Rechtsgebiet using fast keyword matching.

    Reuses RECHTSGEBIET_KEYWORDS from metadata_extractor.py so the same
    vocabulary is used for document-side and query-side classification.

    Uses word-boundary matching (\\b) to prevent subsumption errors
    (e.g. "Miete" matching "Mietrecht" but also "unbefristete Miete").
    Disambiguates multi-domain queries by counting unique (non-shared) hits.
    """

    # Pre-compile regex patterns for each keyword for performance
    _keyword_patterns: dict[str, dict[str, re.Pattern]] = {}

    @classmethod
    def _get_patterns(cls) -> dict[str, dict[str, re.Pattern]]:
        """Build and cache word-boundary regex patterns per domain."""
        if not cls._keyword_patterns:
            for gebiet, keywords in RECHTSGEBIET_KEYWORDS.items():
                cls._keyword_patterns[gebiet] = {}
                for kw in keywords:
                    # Use word boundary matching to prevent partial matches
                    # e.g. "Miete" should not match inside "Mietrechtserhaltungsanspruch"
                    escaped = re.escape(kw)
                    cls._keyword_patterns[gebiet][kw] = re.compile(
                        rf'\b{escaped}\b', re.IGNORECASE
                    )
        return cls._keyword_patterns

    def classify(self, query: str) -> Optional[str]:
        """Determine the most likely Rechtsgebiet for a query string.

        Args:
            query: User query in natural language.

        Returns:
            Rechtsgebiet name (e.g. "Arbeitsrecht") or None if no keywords match.
        """
        if not query or not query.strip():
            return None

        patterns = self._get_patterns()
        scores: dict[str, int] = {}
        unique_scores: dict[str, int] = {}  # Non-shared keyword hits

        for gebiet, kw_patterns in patterns.items():
            keyword_hits = 0
            unique_hits = 0
            for kw, pattern in kw_patterns.items():
                if pattern.search(query):
                    keyword_hits += 1
                    # Count as unique if this keyword is NOT shared with other domains
                    shared_domains = SHARED_KEYWORDS.get(kw)
                    if not shared_domains or shared_domains == {gebiet}:
                        unique_hits += 1
            if keyword_hits > 0:
                scores[gebiet] = keyword_hits
                unique_scores[gebiet] = unique_hits

        if not scores:
            return None

        # If top domains are tied or share ambiguous keywords, use unique_hits as tiebreaker
        max_score = max(scores.values())
        top_domains = [g for g, s in scores.items() if s == max_score]

        if len(top_domains) == 1:
            return top_domains[0]

        # Tiebreaker: domain with more unique (non-shared) keyword hits wins
        best = max(top_domains, key=lambda g: unique_scores.get(g, 0))
        return best
