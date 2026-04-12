"""
agents/signals/sources/newsapi.py

NewsAPI client for fetching company and domain news signals.
Free tier: 100 requests/day, 1 month historical data.

API docs: https://newsapi.org/docs
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

NEWSAPI_BASE    = "https://newsapi.org/v2"
DEFAULT_TIMEOUT = 15.0
LOOKBACK_DAYS   = 30   # how far back to search for signals


def _get_api_key() -> str | None:
    return os.getenv("NEWSAPI_API_KEY")


def _from_date() -> str:
    """ISO date string for LOOKBACK_DAYS ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    return dt.strftime("%Y-%m-%d")


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=False,
)
async def fetch_company_news(
    company:     str,
    max_results: int = 5,
) -> list[dict]:
    """
    Fetch recent news articles mentioning a specific company.

    Args:
        company:     company name e.g. "Adobe", "Siemens"
        max_results: max articles to return

    Returns:
        List of article dicts with keys: title, description,
        url, publishedAt, source.name
        Empty list on any error — never raises.
    """
    key = _get_api_key()
    if not key:
        logger.debug("[newsapi] NEWSAPI_API_KEY not set — skipping")
        return []

    # Build query — company name + hiring intent keywords
    query = f'"{company}" AND (hiring OR funding OR expansion OR layoff OR growth)'

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{NEWSAPI_BASE}/everything",
                params={
                    "q":          query,
                    "from":       _from_date(),
                    "sortBy":     "relevancy",
                    "pageSize":   max_results,
                    "language":   "en",
                    "apiKey":     key,
                },
            )
            response.raise_for_status()
            data = response.json()

        articles = data.get("articles", [])
        logger.info(
            f"[newsapi] '{company}': {len(articles)} articles found"
        )
        return articles

    except httpx.HTTPStatusError as e:
        logger.warning(
            f"[newsapi] HTTP {e.response.status_code} for '{company}'"
        )
        return []
    except Exception as e:
        logger.warning(f"[newsapi] Error fetching '{company}': {e}")
        return []


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    reraise=False,
)
async def fetch_domain_news(
    domain_keywords: list[str],
    max_results:     int = 10,
) -> list[dict]:
    """
    Fetch recent news about a domain/industry for watch list discovery.
    Used to find fast-growing companies not yet in ranked results.

    Args:
        domain_keywords: e.g. ["AML", "fintech", "machine learning India"]
        max_results:     max articles to return

    Returns:
        List of article dicts. Empty list on any error.
    """
    key = _get_api_key()
    if not key:
        return []

    query = " OR ".join(f'"{kw}"' for kw in domain_keywords[:3])
    query += " AND (funding OR expansion OR hiring OR startup)"

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                f"{NEWSAPI_BASE}/everything",
                params={
                    "q":        query,
                    "from":     _from_date(),
                    "sortBy":   "publishedAt",
                    "pageSize": max_results,
                    "language": "en",
                    "apiKey":   key,
                },
            )
            response.raise_for_status()
            data = response.json()

        articles = data.get("articles", [])
        logger.info(
            f"[newsapi] Domain scan: {len(articles)} articles found"
        )
        return articles

    except Exception as e:
        logger.warning(f"[newsapi] Domain scan error: {e}")
        return []
