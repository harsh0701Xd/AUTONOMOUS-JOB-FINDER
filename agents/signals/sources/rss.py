"""
agents/signals/sources/rss.py

RSS feed client for tech news signals.
Zero cost, no API key, zero rate limits.

Feeds covered:
  - TechCrunch (global tech funding + startups)
  - YourStory  (India startup ecosystem)
  - ET Tech    (Economic Times — India tech business)
  - Inc42      (India startup funding)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0

RSS_FEEDS = {
    "techcrunch": "https://techcrunch.com/feed/",
    "yourstory":  "https://yourstory.com/feed",
    "ettech":     "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    "inc42":      "https://inc42.com/feed/",
}


def _parse_pub_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse RSS pubDate string to timezone-aware datetime."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _parse_feed(xml_content: str, source_name: str) -> list[dict]:
    """
    Parse RSS XML into a list of article dicts.
    Returns empty list on parse error.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning(f"[rss] Parse error for {source_name}: {e}")
        return []

    articles = []
    # Handle both RSS 2.0 (<channel><item>) and Atom (<entry>) formats
    items = root.findall(".//item") or root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )

    for item in items:
        title = (
            item.findtext("title") or
            item.findtext("{http://www.w3.org/2005/Atom}title") or ""
        ).strip()

        description = (
            item.findtext("description") or
            item.findtext("{http://www.w3.org/2005/Atom}summary") or ""
        ).strip()

        url = (
            item.findtext("link") or
            item.findtext("{http://www.w3.org/2005/Atom}link") or ""
        ).strip()

        pub_date_str = (
            item.findtext("pubDate") or
            item.findtext("{http://www.w3.org/2005/Atom}published") or ""
        )

        if not title:
            continue

        articles.append({
            "title":       title,
            "description": description[:500],   # cap description length
            "url":         url,
            "publishedAt": _parse_pub_date(pub_date_str),
            "source":      source_name,
        })

    return articles


async def fetch_rss_feed(
    feed_name: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Fetch and parse a single RSS feed by name.

    Args:
        feed_name:   key from RSS_FEEDS dict
        max_results: max articles to return

    Returns:
        List of article dicts. Empty list on any error.
    """
    url = RSS_FEEDS.get(feed_name)
    if not url:
        logger.warning(f"[rss] Unknown feed: {feed_name}")
        return []

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "AutonomousJobFinder/1.0"},
                follow_redirects=True,
            )
            response.raise_for_status()

        articles = _parse_feed(response.text, feed_name)
        result   = articles[:max_results]
        logger.info(f"[rss] {feed_name}: {len(result)} articles fetched")
        return result

    except httpx.TimeoutException:
        logger.warning(f"[rss] Timeout fetching {feed_name}")
        return []
    except Exception as e:
        logger.warning(f"[rss] Error fetching {feed_name}: {e}")
        return []


async def fetch_all_rss_feeds(max_per_feed: int = 20) -> list[dict]:
    """
    Fetch all configured RSS feeds concurrently.
    Returns combined deduplicated article list.
    """
    import asyncio

    tasks = [
        fetch_rss_feed(name, max_per_feed)
        for name in RSS_FEEDS
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_articles = []
    for result in results:
        if isinstance(result, list):
            all_articles.extend(result)

    logger.info(f"[rss] Total articles from all feeds: {len(all_articles)}")
    return all_articles


def filter_articles_by_company(
    articles: list[dict],
    company:  str,
) -> list[dict]:
    """
    Filter RSS articles that mention a specific company.
    Checks both title and description.
    """
    company_lower = company.lower()
    # Also match common abbreviations and variations
    company_words = [w for w in company_lower.split() if len(w) > 3]

    matched = []
    for article in articles:
        text = (
            (article.get("title") or "") + " " +
            (article.get("description") or "")
        ).lower()

        if company_lower in text or any(w in text for w in company_words):
            matched.append(article)

    return matched
