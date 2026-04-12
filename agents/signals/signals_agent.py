"""
agents/signals/signals_agent.py

Agent 5 — Hiring Signals Agent

Responsibility:
  - Extract unique companies from ranked_jobs[]
  - Extract domain keywords from candidate_profile
  - Fetch news signals from NewsAPI + RSS feeds in parallel
  - Classify each signal using classifier.py
  - Cross-reference signals back to user's matched jobs
  - Produce hiring_signals[] (anchored to results) and
    watch_list[] (proactive discovery)

Input  (from SessionState): ranked_jobs[], candidate_profile, confirmed_profiles
Output (to SessionState)  : hiring_signals[], watch_list[]

Design note:
  Signal classification is intentionally isolated in classifier.py.
  Swapping keyword → LLM classification requires changing only one
  function in that file. This agent is unaffected.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from agents.signals.classifier import SignalClassification, classify_signal
from agents.signals.sources.newsapi import fetch_company_news, fetch_domain_news
from agents.signals.sources.rss import (
    fetch_all_rss_feeds,
    filter_articles_by_company,
)
from core.state.session_state import (
    CandidateProfile,
    HiringSignal,
    RankedJob,
    SessionState,
    SuggestedProfile,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_COMPANIES_TO_SCAN  = 10   # top N companies from ranked results
MAX_SIGNALS_PER_COMPANY = 2   # best 2 signals per company
MIN_SIGNAL_STRENGTH    = "low"  # filter out noise
WATCH_LIST_MAX         = 5    # max proactive watch list entries


# ── Target extraction ─────────────────────────────────────────────────────────

def _extract_companies(ranked_jobs: list[RankedJob]) -> list[str]:
    """
    Extract unique company names from ranked jobs.
    Ordered by fit score (highest first), capped at MAX_COMPANIES_TO_SCAN.
    """
    seen = set()
    companies = []
    for job in ranked_jobs:
        company = (job.company or "").strip()
        if company and company not in seen:
            seen.add(company)
            companies.append(company)
        if len(companies) >= MAX_COMPANIES_TO_SCAN:
            break
    return companies


def _extract_domain_keywords(
    profile:   Optional[CandidateProfile],
    confirmed: list[SuggestedProfile],
) -> list[str]:
    """
    Extract domain keywords for proactive watch list scanning.
    Combines domain expertise, profile titles, and key skills.
    """
    keywords = []

    if profile:
        keywords.extend(profile.domain_expertise or [])
        if profile.current_title:
            keywords.append(profile.current_title)

    for p in confirmed[:2]:
        keywords.append(p.title)

    # Deduplicate preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result[:5]


def _jobs_matched_count(company: str, ranked_jobs: list[RankedJob]) -> int:
    """Count how many ranked jobs are from this company."""
    company_lower = company.lower()
    return sum(
        1 for job in ranked_jobs
        if (job.company or "").lower() == company_lower
    )


def _relevant_profiles(
    company:   str,
    ranked_jobs: list[RankedJob],
) -> list[str]:
    """Get the profile titles that matched jobs at this company."""
    company_lower = company.lower()
    profiles = set()
    for job in ranked_jobs:
        if (job.company or "").lower() == company_lower:
            for profile in job.matched_via:
                profiles.add(profile.strip())
    return list(profiles)


# ── URL validation ────────────────────────────────────────────────────────────

_BAD_URL_PATTERNS = [
    "consent.yahoo.com",
    "consent.",
    "login.",
    "signin.",
    "accounts.google",
    "javascript:",
]

def _is_valid_url(url: str) -> bool:
    """Filter out consent walls, login pages, and non-article URLs."""
    if not url or not url.startswith("http"):
        return False
    return not any(pattern in url.lower() for pattern in _BAD_URL_PATTERNS)

def _article_to_text(article: dict) -> str:
    """Combine title + description into a single text for classification."""
    title = article.get("title") or ""
    desc  = article.get("description") or ""
    return f"{title}. {desc}".strip()


def _article_date(article: dict) -> Optional[datetime]:
    """Extract datetime from article, handling both str and datetime."""
    val = article.get("publishedAt")
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        val = val.replace("Z", "+00:00")
        return datetime.fromisoformat(val)
    except (ValueError, AttributeError):
        return None


# ── Per-company signal processing ─────────────────────────────────────────────

async def _process_company(
    company:     str,
    rss_articles: list[dict],
    ranked_jobs:  list[RankedJob],
) -> list[HiringSignal]:
    """
    Fetch and classify all signals for one company.
    Combines NewsAPI results with pre-fetched RSS articles.
    Returns list of HiringSignal objects (may be empty).
    """
    # Fetch from NewsAPI (async)
    newsapi_articles = await fetch_company_news(company, max_results=5)

    # Filter RSS articles for this company (already fetched)
    rss_matched = filter_articles_by_company(rss_articles, company)

    # Combine all articles
    all_articles = newsapi_articles + rss_matched

    if not all_articles:
        logger.debug(f"[signals] No articles found for '{company}'")
        return []

    # Classify each article
    signals: list[HiringSignal] = []
    for article in all_articles[:5]:
        text = _article_to_text(article)
        if len(text) < 20:
            continue

        # Skip articles that don't actually mention the company
        if company.lower() not in text.lower():
            logger.debug(f"[signals] Skipping off-topic article for '{company}'")
            continue

        # Skip invalid URLs (consent walls, login pages)
        url = article.get("url") or ""
        if not _is_valid_url(url):
            logger.debug(f"[signals] Skipping bad URL for '{company}': {url[:60]}")
            continue

        classification: SignalClassification = classify_signal(text, company)

        # Skip neutral / low-confidence signals
        if classification.signal_type == "neutral":
            continue

        # Build signal — extract source name safely (NewsAPI returns a dict)
        src = article.get("source") or {}
        source_name = (
            src.get("name") if isinstance(src, dict)
            else str(src)
        ) or "unknown"

        try:
            signal = HiringSignal(
                company              = company,
                signal_type          = classification.signal_type,
                signal_strength      = classification.signal_strength,
                summary              = classification.summary,
                is_positive          = classification.is_positive,
                confidence           = classification.confidence,
                source_url           = article.get("url") or "",
                source_date          = _article_date(article),
                source_name          = source_name,
                jobs_you_matched     = _jobs_matched_count(company, ranked_jobs),
                relevant_to_profiles = _relevant_profiles(company, ranked_jobs),
            )
            signals.append(signal)
        except Exception as e:
            logger.warning(f"[signals] Skipping article for '{company}': {e}")

    # Return best signals per company (highest confidence first)
    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals[:MAX_SIGNALS_PER_COMPANY]


# ── Watch list discovery ──────────────────────────────────────────────────────

async def _build_watch_list(
    domain_keywords: list[str],
    existing_companies: set[str],
    rss_articles: list[dict],
) -> list[HiringSignal]:
    """
    Discover companies not in ranked results but showing growth
    in the user's domain — proactive watch list.
    """
    # Fetch domain-level news
    domain_articles = await fetch_domain_news(domain_keywords, max_results=10)
    all_articles    = domain_articles + rss_articles

    watch_companies: dict[str, list[dict]] = {}

    for article in all_articles:
        text = _article_to_text(article)
        classification = classify_signal(text, "")

        # Only strong positive signals qualify for watch list
        if not classification.is_positive:
            continue
        if classification.signal_strength == "low":
            continue

        # Try to extract a company name from the title
        title = (article.get("title") or "").strip()
        if not title:
            continue

        # Heuristic: first capitalized sequence before a verb
        # e.g. "Razorpay raises $100M" → "Razorpay"
        import re
        match = re.match(r"^([A-Z][A-Za-z0-9\s&\-\.]{2,30}?)(?:\s+(?:raises|secures|launches|opens|expands|acquires|announces|hires|grows))", title)
        if not match:
            continue

        company_name = match.group(1).strip()

        # Skip if already in ranked results
        if company_name.lower() in {c.lower() for c in existing_companies}:
            continue

        if company_name not in watch_companies:
            watch_companies[company_name] = []
        watch_companies[company_name].append(article)

    # Build watch list signals
    watch_signals: list[HiringSignal] = []
    for company, articles in list(watch_companies.items())[:WATCH_LIST_MAX]:
        best = articles[0]
        text = _article_to_text(best)
        classification = classify_signal(text, company)

        src = best.get("source") or {}
        source_name = (
            src.get("name") if isinstance(src, dict)
            else str(src)
        ) or "unknown"

        try:
            watch_signals.append(HiringSignal(
                company              = company,
                signal_type          = classification.signal_type,
                signal_strength      = classification.signal_strength,
                summary              = f"Watch list: {classification.summary}",
                is_positive          = True,
                confidence           = classification.confidence,
                source_url           = best.get("url") or "",
                source_date          = _article_date(best),
                source_name          = source_name,
                jobs_you_matched     = 0,
                relevant_to_profiles = [],
            ))
        except Exception as e:
            logger.warning(f"[signals] Skipping watch list entry for '{company}': {e}")

    return watch_signals


# ── Main agent function ───────────────────────────────────────────────────────

async def run_signals_agent(state: SessionState) -> SessionState:
    """
    Agent 5 — Hiring Signals Agent.

    Fetches and classifies hiring signals for companies in
    the user's ranked job results plus proactive domain discovery.

    Returns updated SessionState with hiring_signals[] and watch_list[].
    """
    state.current_agent = "signals"
    logger.info(
        f"[signals] Starting — session_id={state.session_id}, "
        f"ranked_jobs={len(state.ranked_jobs)}"
    )

    if not state.ranked_jobs:
        logger.warning("[signals] No ranked jobs — skipping signal scan")
        return state

    # Step 1: Extract targets
    companies       = _extract_companies(state.ranked_jobs)
    domain_keywords = _extract_domain_keywords(
        state.candidate_profile, state.confirmed_profiles
    )

    logger.info(
        f"[signals] Scanning {len(companies)} companies, "
        f"keywords: {domain_keywords[:3]}"
    )

    # Step 2: Pre-fetch all RSS feeds once (shared across all companies)
    rss_articles = await fetch_all_rss_feeds(max_per_feed=20)

    # Step 3: Process all companies in parallel
    company_tasks = [
        _process_company(company, rss_articles, state.ranked_jobs)
        for company in companies
    ]

    # Also run watch list discovery in parallel
    watch_task = _build_watch_list(
        domain_keywords   = domain_keywords,
        existing_companies = set(companies),
        rss_articles      = rss_articles,
    )

    all_results = await asyncio.gather(
        *company_tasks, watch_task,
        return_exceptions=True
    )

    # Step 4: Collect results
    hiring_signals: list[HiringSignal] = []
    watch_list:     list[HiringSignal] = []

    for i, result in enumerate(all_results[:-1]):   # all except last (watch list)
        if isinstance(result, Exception):
            logger.warning(
                f"[signals] Error processing '{companies[i]}': {result}"
            )
            continue
        hiring_signals.extend(result)

    watch_result = all_results[-1]
    if isinstance(watch_result, list):
        watch_list = watch_result

    # Sort by confidence desc
    hiring_signals.sort(key=lambda s: s.confidence, reverse=True)

    state.hiring_signals = hiring_signals
    state.watch_list     = watch_list
    state.error          = None

    logger.info(
        f"[signals] Complete — "
        f"{len(hiring_signals)} hiring signals, "
        f"{len(watch_list)} watch list entries"
    )
    return state


# ── LangGraph node wrapper ────────────────────────────────────────────────────

async def node_signals(state: dict) -> dict:
    """
    LangGraph async node wrapper for the signals agent.
    Defined as async — graph must use ainvoke.
    """
    session = SessionState(**state)

    if not session.ranked_jobs:
        logger.warning("[graph] Skipping signals — no ranked jobs")
        return state

    updated = await run_signals_agent(session)
    return updated.model_dump()
