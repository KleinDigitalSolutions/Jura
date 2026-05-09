"""Query rewriting using LLM — transforms user queries into legal-search-friendly variants.

Uses the same provider pattern as modal_deploy.py (deepseek default, anthropic fallback)
but reads API keys from environment variables directly — no Modal dependency.
"""
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """Du bist ein deutscher Jurist. Schreibe die folgende Frage in 3 juristische
Varianten um, die in deutschen Gesetzestexten vorkommen würden.

Jede Variante soll mindestens einen juristischen Fachbegriff oder
Paragraphenverweis enthalten, der die originale Frage präzisiert.

FORMAT: Gib EXAKT 3 Zeilen zurück, eine pro Variante. Keine Nummerierung, kein Prefix.

BEISPIELE:
Frage: Kann mein Chef mich feuern?
Antwort:
ordentliche Kündigung außerordentliche Kündigung § 1 KSchG § 626 BGB soziale Rechtfertigung
Kündigung Arbeitsverhältnis Kündigungsschutzklage KSchG
fristlose Kündigung Arbeitgeber verhaltensbedingte Kündigung personenbedingte Kündigung

Frage: Kann mein Vermieter die Kaution einbehalten?
Antwort:
Mietkaution § 551 BGB § 548 BGB Sicherungseinbehalt Verjährung Ersatzansprüche
Mietsicherheit Rückzahlung Mietende § 548 BGB Verjährung
Mietkaution Abrechnung Frist Schadensersatz Vermieter

Frage: Unfall Schaden wer bezahlt?
Antwort:
Schadensersatz § 823 BGB § 249 BGB unerlaubte Handlung Haftung
Deliktsrecht Schadensersatzpflicht Verursacher Gefährdungshaftung
Kfz-Haftpflicht Schadensregulierung § 7 StVG"""

REWRITE_USER_PROMPT = """Frage: {query}
Antwort:"""


class LegalQueryRewriter:
    """Rewrites a user query into 3 legal-search-friendly variants using an LLM.

    Supports deepseek (default) and anthropic providers with failover.
    On error or timeout, falls back to [original_query] so retrieval never
    breaks when the LLM is unavailable.
    """

    def __init__(
        self,
        default_provider: str = "deepseek",
        timeout: Optional[float] = None,
    ):
        self.default_provider = default_provider
        self.timeout = timeout if timeout is not None else float(
            os.getenv("REWRITER_TIMEOUT", "5.0")
        )

    def rewrite(self, query: str, rechtsgebiet: Optional[str] = None) -> list[str]:
        """Rewrite query into up to 3 legal-search-friendly variants.

        Args:
            query: User query in natural language.
            rechtsgebiet: Detected legal area to guide rewriting (e.g. "Familienrecht").

        Returns:
            List of rewritten query variants. [query] on failure or timeout.
        """
        if not query or not query.strip():
            return [query]

        # Build prompt with optional rechtsgebiet context
        system_prompt = REWRITE_SYSTEM_PROMPT
        if rechtsgebiet:
            system_prompt += (
                f"\n\nACHTUNG: Die Frage betrifft das Rechtsgebiet **{rechtsgebiet}**."
                f"\nStelle sicher, dass alle 3 Varianten spezifische Paragraphen und"
                f"\nFachbegriffe aus diesem Rechtsgebiet enthalten."
            )

        prompt = REWRITE_USER_PROMPT.format(query=query.strip())

        # Try primary provider first, then failover
        providers = [self.default_provider]
        if self.default_provider == "deepseek":
            providers.append("anthropic")
        else:
            providers.append("deepseek")

        last_error = ""
        for provider in providers:
            try:
                if provider == "anthropic":
                    text = self._call_anthropic(prompt, system_prompt)
                else:
                    text = self._call_deepseek(prompt, system_prompt)

                if text is None:
                    continue

                variants = self._parse_response(text)
                if variants:
                    return variants
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Query rewrite failed with {provider}: {e}")
                continue

        if last_error:
            logger.warning(f"All rewrite providers failed: {last_error}")
        return [query]

    def _call_deepseek(self, prompt: str, system_prompt: str = REWRITE_SYSTEM_PROMPT) -> Optional[str]:
        """Call DeepSeek via OpenAI-compatible API. Returns None on failure."""
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        from openai import OpenAI, Timeout as OpenAITimeout

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=self.timeout,
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=500,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str, system_prompt: str = REWRITE_SYSTEM_PROMPT) -> Optional[str]:
        """Call Anthropic Claude. Returns None on failure."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        import httpx
        from anthropic import Anthropic

        http_client = httpx.Client(timeout=self.timeout)
        client = Anthropic(api_key=api_key, http_client=http_client)
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=500,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _parse_response(self, text: str) -> list[str]:
        """Parse LLM response into clean variant list."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Deduplicate by lowered content, preserve order
        seen: set[str] = set()
        variants: list[str] = []
        for line in lines:
            lowered = line.lower()
            if lowered not in seen and len(line) >= 3:
                seen.add(lowered)
                variants.append(line)
        return variants[:3]
