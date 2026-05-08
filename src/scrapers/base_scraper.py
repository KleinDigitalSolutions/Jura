"""Abstract base class for legal document scrapers."""
import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import RATE_LIMIT_SECONDS, MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT, USER_AGENTS


class RateLimiter:
    """Domain-based rate limiter: 1 request per `interval` seconds per domain."""

    def __init__(self, interval: float = RATE_LIMIT_SECONDS):
        self.interval = interval
        self._last_request: dict[str, float] = {}

    async def wait(self, domain: str) -> None:
        now = time.monotonic()
        last = self._last_request.get(domain, 0)
        wait_time = self.interval - (now - last)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self._last_request[domain] = time.monotonic()


rate_limiter = RateLimiter()


class BaseScraper(ABC):
    """Abstract scraper with retry logic, rate limiting, and user-agent rotation."""

    source_name: str = "base"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session: Optional[aiohttp.ClientSession] = session
        self._owns_session: bool = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                headers={"User-Agent": random.choice(USER_AGENTS)},
            )
            self._owns_session = True
        return self._session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_BACKOFF, min=1, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def fetch(self, url: str, *, domain: Optional[str] = None) -> str:
        """Fetch URL with rate limiting and retries."""
        if domain is None:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc

        await rate_limiter.wait(domain)
        session = await self._get_session()
        start = time.monotonic()
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                text = self._decode_response(data, resp)
                elapsed = time.monotonic() - start
                logger.info(f"[{self.source_name}] GET {url} → {resp.status} ({elapsed:.1f}s)")
                return text
        except Exception:
            elapsed = time.monotonic() - start
            logger.error(f"[{self.source_name}] GET {url} → FAIL ({elapsed:.1f}s)")
            raise

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_BACKOFF, min=1, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def fetch_bytes(self, url: str, *, domain: Optional[str] = None) -> bytes:
        """Fetch binary content with rate limiting and retries."""
        if domain is None:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc

        await rate_limiter.wait(domain)
        session = await self._get_session()
        start = time.monotonic()
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                elapsed = time.monotonic() - start
                logger.info(f"[{self.source_name}] GET {url} → {resp.status} {len(data)}B ({elapsed:.1f}s)")
                return data
        except Exception:
            elapsed = time.monotonic() - start
            logger.error(f"[{self.source_name}] GET {url} → FAIL ({elapsed:.1f}s)")
            raise

    @staticmethod
    def _decode_response(data: bytes, resp) -> str:
        """Decode response bytes with correct encoding. Tries Content-Type charset,
        then common German encodings."""
        content_type = resp.headers.get("Content-Type", "")
        # Try charset from header
        import re
        m = re.search(r'charset=([^\s;]+)', content_type)
        if m:
            try:
                return data.decode(m.group(1))
            except (LookupError, UnicodeDecodeError):
                pass
        # Fallback: common German encodings
        for enc in ["utf-8", "iso-8859-1", "windows-1252", "latin-1"]:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @abstractmethod
    async def scrape(self) -> list[dict]:
        """Run the scraper. Returns list of structured document dicts."""
        ...
