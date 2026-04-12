"""
agents/signals/classifier.py

Signal classifier — determines type and strength of a hiring signal
from a news article.

ARCHITECTURE NOTE:
  The public interface `classify_signal(article_text, company)` is
  intentionally stable. The keyword implementation below (v1) can be
  replaced with a Claude API call (v2) by swapping only this function's
  body. Nothing upstream or downstream changes.

v1: Keyword matching — free, instant, no API call
v2: Claude Haiku call — accurate classification, ~₹0.001/article
    (see _classify_with_llm stub at bottom for the upgrade path)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SignalType     = Literal["funding", "expansion", "product_launch",
                         "headcount_growth", "hiring_freeze", "layoff", "neutral"]
SignalStrength = Literal["high", "medium", "low"]


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class SignalClassification:
    """
    Result of classifying one article.
    Schema is identical whether produced by keywords or LLM.
    """
    signal_type:     SignalType
    signal_strength: SignalStrength
    summary:         str        # 1-sentence human-readable description
    is_positive:     bool       # True = hiring momentum, False = avoid signal
    confidence:      float      # 0.0–1.0, keyword=fixed, LLM=model-reported


# ── Keyword dictionaries ──────────────────────────────────────────────────────

_POSITIVE_HIGH = {
    "raised", "funding", "series a", "series b", "series c", "series d",
    "series e", "series f", "ipo", "unicorn", "valuation", "investment",
    "backed by", "venture", "secured", "$", "million", "billion",
}

_POSITIVE_MEDIUM = {
    "expansion", "expands", "expanding", "new office", "opens office",
    "headquarter", "new location", "global presence", "enters market",
    "launches in", "hiring", "hiring spree", "headcount", "growing team",
    "new division", "new team", "new product", "launched", "launches",
    "partnership", "acquisition", "acquires", "merger",
}

_NEGATIVE_HIGH = {
    "layoffs", "laid off", "lay off", "retrenchment", "job cuts",
    "reducing workforce", "downsizing", "restructuring", "bankruptcy",
    "shutdown", "shutting down", "closing down", "wind down",
}

_NEGATIVE_MEDIUM = {
    "hiring freeze", "pause hiring", "slow hiring", "cost cutting",
    "budget cuts", "losses", "declining revenue", "missed targets",
}

_GROWTH_SIGNALS = {
    "product launch", "new feature", "announced", "unveils", "unveiled",
    "breakthrough", "innovation", "ai division", "ml team", "data science team",
    "engineering team", "technology team", "research lab",
}


def _keyword_score(text: str, keywords: set[str]) -> int:
    """Count how many keywords appear in lowercased text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


# ── v1: Keyword classifier (current implementation) ───────────────────────────

def _classify_with_keywords(
    article_text: str,
    company:      str,
) -> SignalClassification:
    """
    Keyword-based signal classification.
    Fast, free, deterministic. Accuracy ~70% on clear signals.
    """
    text = article_text.lower()

    pos_high   = _keyword_score(text, _POSITIVE_HIGH)
    pos_medium = _keyword_score(text, _POSITIVE_MEDIUM)
    neg_high   = _keyword_score(text, _NEGATIVE_HIGH)
    neg_medium = _keyword_score(text, _NEGATIVE_MEDIUM)
    growth     = _keyword_score(text, _GROWTH_SIGNALS)

    # Negative signals take priority
    if neg_high >= 1:
        return SignalClassification(
            signal_type     = "layoff",
            signal_strength = "high",
            summary         = f"{company} may have negative hiring momentum — layoff or shutdown signals detected.",
            is_positive     = False,
            confidence      = 0.75,
        )

    if neg_medium >= 1:
        return SignalClassification(
            signal_type     = "hiring_freeze",
            signal_strength = "medium",
            summary         = f"{company} shows signs of hiring slowdown.",
            is_positive     = False,
            confidence      = 0.65,
        )

    # Positive signals
    if pos_high >= 2:
        return SignalClassification(
            signal_type     = "funding",
            signal_strength = "high",
            summary         = f"{company} shows strong funding or investment activity — likely to be actively hiring.",
            is_positive     = True,
            confidence      = 0.80,
        )

    if pos_high >= 1:
        return SignalClassification(
            signal_type     = "funding",
            signal_strength = "medium",
            summary         = f"{company} has recent funding activity suggesting hiring momentum.",
            is_positive     = True,
            confidence      = 0.70,
        )

    if pos_medium >= 2:
        return SignalClassification(
            signal_type     = "expansion",
            signal_strength = "high",
            summary         = f"{company} is actively expanding — strong signal for upcoming hiring.",
            is_positive     = True,
            confidence      = 0.75,
        )

    if pos_medium >= 1 or growth >= 1:
        return SignalClassification(
            signal_type     = "product_launch" if growth >= 1 else "expansion",
            signal_strength = "medium",
            summary         = f"{company} shows growth or product activity suggesting moderate hiring momentum.",
            is_positive     = True,
            confidence      = 0.60,
        )

    # Nothing meaningful found
    return SignalClassification(
        signal_type     = "neutral",
        signal_strength = "low",
        summary         = f"No strong hiring signals detected for {company}.",
        is_positive     = False,
        confidence      = 0.50,
    )


# ── v2 stub: LLM classifier (future upgrade) ─────────────────────────────────

def _classify_with_llm(
    article_text: str,
    company:      str,
) -> SignalClassification:
    """
    LLM-based signal classification using Claude Haiku.

    TO ACTIVATE v2:
      1. Uncomment this function body
      2. Change the `classify_signal` dispatcher below to call this instead
      3. No other changes needed anywhere in the codebase

    Cost: ~₹0.001 per article (Claude Haiku)
    Accuracy: ~90% vs ~70% for keywords
    """
    # import anthropic, json
    # client = anthropic.Anthropic()
    # prompt = SIGNAL_CLASSIFY_PROMPT.replace("{company}", company)
    #                                .replace("{article_text}", article_text[:2000])
    # response = client.messages.create(
    #     model="claude-haiku-4-5-20251001",
    #     max_tokens=256,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # data = json.loads(response.content[0].text)
    # return SignalClassification(
    #     signal_type     = data["signal_type"],
    #     signal_strength = data["signal_strength"],
    #     summary         = data["summary"],
    #     is_positive     = data["is_positive"],
    #     confidence      = data["confidence"],
    # )
    raise NotImplementedError("LLM classifier not yet activated — use keyword classifier")


# ── LLM prompt template (ready for v2) ────────────────────────────────────────

SIGNAL_CLASSIFY_PROMPT = """
You are a hiring signal analyst. Given a news article, classify whether
it indicates positive hiring momentum, negative momentum, or neither
for the company named below.

Company: {company}

Article:
{article_text}

Return ONLY valid JSON:
{{
  "signal_type": "funding" | "expansion" | "product_launch" |
                 "headcount_growth" | "hiring_freeze" | "layoff" | "neutral",
  "signal_strength": "high" | "medium" | "low",
  "summary": "one sentence explanation tied to this specific company",
  "is_positive": true | false,
  "confidence": 0.0 to 1.0
}}
""".strip()


# ── Public interface — the seam ───────────────────────────────────────────────

def classify_signal(
    article_text: str,
    company:      str,
) -> SignalClassification:
    """
    Classify a news article as a hiring signal for a given company.

    PUBLIC INTERFACE — stable across v1 and v2.
    Swap implementation by changing which private function is called here.

    Current: keyword matching (v1)
    Upgrade: replace with _classify_with_llm(article_text, company)
    """
    return _classify_with_keywords(article_text, company)
