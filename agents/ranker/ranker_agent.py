"""
agents/ranker/ranker_agent.py

Agent 4 — Ranker + Deduplication Agent

Responsibility:
  - Deduplicate raw_jobs[] across sources and profiles
  - Score each job against candidate_profile using semantic similarity
  - Apply recency decay and seniority alignment scoring
  - Output unified ranked_jobs[] list with fit scores and profile badges

Scoring model (composite):
  fit_score = (
      0.50 * semantic_score    +   # skill + JD text similarity
      0.25 * seniority_score   +   # seniority level alignment
      0.25 * recency_score         # posting freshness
  )

Input  (from SessionState): raw_jobs[], candidate_profile
Output (to SessionState)  : ranked_jobs[]
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from core.state.session_state import (
    CandidateProfile,
    RankedJob,
    RawJob,
    SessionState,
    SuggestedProfile,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_FINAL_RESULTS  = 25    # hard cap on ranked results shown to user
RECENCY_DECAY_DAYS = 30    # jobs older than this score 0 on recency
WEIGHTS = {
    "semantic":  0.50,
    "seniority": 0.25,
    "recency":   0.25,
}

SENIORITY_ORDER = [
    "intern", "junior", "mid", "senior", "lead", "principal", "executive"
]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _fingerprint(job: RawJob) -> str:
    """
    Create a dedup fingerprint from company + title (normalised).
    Jobs with the same company and similar title are duplicates
    regardless of which source or profile found them.
    """
    company = re.sub(r"\s+", " ", (job.company or "").lower().strip())
    title   = re.sub(r"\s+", " ", (job.title or "").lower().strip())
    # Strip seniority prefixes for better matching
    # "Sr Data Scientist" and "Senior Data Scientist" → same job
    title = re.sub(
        r"^(sr\.?|senior|junior|jr\.?|lead|principal|staff)\s+",
        "", title
    )
    return f"{company}::{title}"


def deduplicate(jobs: list[RawJob]) -> list[RawJob]:
    """
    Remove duplicate jobs across sources and profiles.
    When duplicates exist, keep the one with the most complete data
    and merge the matched_profile tags from all copies.

    Returns deduplicated list with merged matched_via tracking.
    """
    seen: dict[str, RawJob]         = {}
    matched_via: dict[str, set[str]] = {}

    for job in jobs:
        fp = _fingerprint(job)

        if fp not in seen:
            seen[fp]        = job
            matched_via[fp] = {job.matched_profile}
        else:
            # Keep the copy with more data (longer JD text)
            existing = seen[fp]
            if len(job.jd_text) > len(existing.jd_text):
                seen[fp] = job
            # Always merge profile tags
            matched_via[fp].add(job.matched_profile)

    # Attach merged matched_via list to each surviving job
    result = []
    for fp, job in seen.items():
        # RawJob is immutable-ish — build a RankedJob shell with merged tags
        job.matched_profile = " | ".join(sorted(matched_via[fp]))
        result.append(job)

    logger.info(
        f"[ranker] Dedup: {len(jobs)} → {len(result)} unique jobs"
    )
    return result


# ── Semantic scoring (no embeddings — keyword overlap) ────────────────────────

def _build_candidate_keywords(profile: CandidateProfile) -> set[str]:
    """
    Extract a keyword set from the candidate profile for matching.
    Uses skills, tools, domain expertise, and job titles.
    """
    keywords: set[str] = set()

    for skill in profile.skills.technical:
        keywords.update(skill.lower().split())
    for tool in profile.skills.tools:
        keywords.update(tool.lower().split())
    for domain in profile.domain_expertise:
        keywords.update(domain.lower().split())
    if profile.current_title:
        keywords.update(profile.current_title.lower().split())
    for exp in profile.work_experience:
        keywords.update(exp.title.lower().split())

    # Remove common stop words that add noise
    stopwords = {
        "and", "or", "the", "a", "an", "of", "in", "at", "for",
        "with", "to", "from", "by", "on", "is", "are", "was",
        "experience", "skills", "using", "based",
    }
    return keywords - stopwords


def _semantic_score(job: RawJob, candidate_keywords: set[str]) -> float:
    """
    Keyword overlap score between job description and candidate profile.

    In v1 we use TF-based keyword matching — no external embedding model
    needed, zero cost, works offline. Agent 4 can be upgraded to use
    sentence-transformers later without changing the interface.

    Returns float 0.0 – 1.0
    """
    if not job.jd_text or not candidate_keywords:
        return 0.3   # neutral score when no data

    jd_words = set(re.findall(r"\b[a-z][a-z0-9+#\-\.]{1,}\b",
                               job.jd_text.lower()))

    overlap  = candidate_keywords & jd_words
    score    = len(overlap) / max(len(candidate_keywords), 1)

    # Normalize to 0–1, cap at 1.0
    return min(score * 2.5, 1.0)   # multiply to make scores more spread out


# ── Seniority scoring ─────────────────────────────────────────────────────────

def _seniority_score(
    job:       RawJob,
    profile:   CandidateProfile,
    confirmed: list[SuggestedProfile],
) -> float:
    """
    Score how well the job's apparent seniority matches
    the candidate's target seniority level.

    Returns float 0.0 – 1.0
    """
    # Infer job seniority from title keywords
    title_lower = job.title.lower()

    job_seniority = None
    if any(w in title_lower for w in ["principal", "staff", "vp", "director"]):
        job_seniority = "principal"
    elif any(w in title_lower for w in ["lead", "head", "manager"]):
        job_seniority = "lead"
    elif any(w in title_lower for w in ["senior", "sr.", "sr "]):
        job_seniority = "senior"
    elif any(w in title_lower for w in ["junior", "jr.", "associate", "entry"]):
        job_seniority = "junior"
    elif any(w in title_lower for w in ["intern", "graduate", "trainee"]):
        job_seniority = "intern"
    else:
        job_seniority = "mid"   # default assumption

    # Get candidate's target seniority from confirmed profiles
    target_seniority = None
    for p in confirmed:
        if p.seniority_target:
            target_seniority = p.seniority_target
            break

    if not target_seniority:
        target_seniority = profile.seniority_level or "mid"

    try:
        job_idx    = SENIORITY_ORDER.index(job_seniority)
        target_idx = SENIORITY_ORDER.index(target_seniority)
        diff       = abs(job_idx - target_idx)

        # Perfect match = 1.0, one level off = 0.7, two = 0.4, three+ = 0.1
        scores = {0: 1.0, 1: 0.7, 2: 0.4}
        return scores.get(diff, 0.1)

    except ValueError:
        return 0.5   # unknown seniority — neutral


# ── Recency scoring ───────────────────────────────────────────────────────────

def _recency_score(job: RawJob) -> float:
    """
    Score job freshness — newer postings score higher.
    Jobs older than RECENCY_DECAY_DAYS score 0.

    Returns float 0.0 – 1.0
    """
    if not job.posted_date:
        return 0.5   # unknown date — neutral score

    now = datetime.now(timezone.utc)
    # Ensure posted_date is timezone-aware
    posted = job.posted_date
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)

    age_days = (now - posted).days

    if age_days <= 0:
        return 1.0
    if age_days >= RECENCY_DECAY_DAYS:
        return 0.0

    # Linear decay from 1.0 → 0.0 over RECENCY_DECAY_DAYS
    return 1.0 - (age_days / RECENCY_DECAY_DAYS)


# ── Recommended action ────────────────────────────────────────────────────────

def _recommended_action(fit_score: float) -> str:
    """Map fit score to a recommended action for the user."""
    if fit_score >= 0.70:
        return "apply_now"
    if fit_score >= 0.50:
        return "apply_with_note"
    if fit_score >= 0.30:
        return "monitor"
    return "skip"


# ── Gap analysis ──────────────────────────────────────────────────────────────

def _gap_skills(
    job:                RawJob,
    candidate_keywords: set[str],
) -> list[str]:
    """
    Identify skills mentioned in the JD that the candidate doesn't have.
    Returns top 5 gap skills as a list of strings.
    """
    if not job.jd_text:
        return []

    # Common tech skills to watch for in JDs
    TECH_SIGNALS = {
        "kubernetes", "docker", "spark", "kafka", "airflow", "dbt",
        "mlflow", "kubeflow", "pytorch", "tensorflow", "transformers",
        "langchain", "fastapi", "react", "typescript", "rust", "scala",
        "snowflake", "databricks", "redshift", "bigquery", "aws", "gcp",
        "azure", "terraform", "pyspark", "flink", "elasticsearch",
        "mongodb", "cassandra", "redis", "graphql", "grpc",
    }

    jd_lower   = job.jd_text.lower()
    gaps       = []

    for skill in TECH_SIGNALS:
        if skill in jd_lower and skill not in candidate_keywords:
            gaps.append(skill)

    return gaps[:5]


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_jobs(
    jobs:      list[RawJob],
    profile:   CandidateProfile,
    confirmed: list[SuggestedProfile],
) -> list[RankedJob]:
    """
    Score and rank a list of deduplicated RawJob objects.

    Args:
        jobs:      deduplicated raw jobs from Agent 3
        profile:   parsed candidate profile from Agent 1
        confirmed: confirmed job profiles from Agent 2

    Returns:
        List of RankedJob objects sorted by fit_score descending,
        capped at MAX_FINAL_RESULTS.
    """
    candidate_keywords = _build_candidate_keywords(profile)
    ranked: list[RankedJob] = []

    for job in jobs:
        sem_score = _semantic_score(job, candidate_keywords)
        sen_score = _seniority_score(job, profile, confirmed)
        rec_score = _recency_score(job)

        fit_score = (
            WEIGHTS["semantic"]  * sem_score +
            WEIGHTS["seniority"] * sen_score +
            WEIGHTS["recency"]   * rec_score
        )

        # Build matched_via badge list from merged profile tag
        matched_via = [
            p.strip()
            for p in job.matched_profile.split("|")
            if p.strip()
        ]

        ranked.append(RankedJob(
            job_id             = job.job_id,
            title              = job.title,
            company            = job.company,
            location           = job.location,
            work_type          = job.work_type,
            jd_text            = job.jd_text,
            apply_url          = job.apply_url,
            source             = job.source,
            posted_date        = job.posted_date,
            salary_min         = job.salary_min,
            salary_max         = job.salary_max,
            matched_via        = matched_via,
            fit_score          = round(fit_score, 3),
            semantic_score     = round(sem_score, 3),
            seniority_score    = round(sen_score, 3),
            recency_score      = round(rec_score, 3),
            gap_skills         = _gap_skills(job, candidate_keywords),
            recommended_action = _recommended_action(fit_score),
        ))

    # Sort by fit_score descending
    ranked.sort(key=lambda j: j.fit_score, reverse=True)

    result = ranked[:MAX_FINAL_RESULTS]
    logger.info(
        f"[ranker] Ranked {len(jobs)} jobs → "
        f"top {len(result)} results "
        f"(scores: {result[0].fit_score:.2f}–{result[-1].fit_score:.2f})"
        if result else "[ranker] No jobs to rank"
    )
    return result


# ── Main agent function ───────────────────────────────────────────────────────

def run_ranker_agent(state: SessionState) -> SessionState:
    """
    Agent 4 — Ranker + Deduplication Agent.

    Reads raw_jobs[] and candidate_profile from state.
    Deduplicates, scores, and ranks all jobs.
    Writes ranked_jobs[] to state.

    Returns updated SessionState.
    """
    state.current_agent = "ranker"
    logger.info(
        f"[ranker] Starting — session_id={state.session_id}, "
        f"raw_jobs={len(state.raw_jobs)}"
    )

    if not state.raw_jobs:
        logger.warning("[ranker] No raw jobs to rank")
        state.ranked_jobs   = []
        state.results_ready = True
        return state

    if not state.candidate_profile:
        logger.error("[ranker] No candidate profile — cannot rank")
        state.error = "Cannot rank jobs: candidate profile missing."
        return state

    # Step 1: Deduplicate
    unique_jobs = deduplicate(state.raw_jobs)

    # Step 2: Score and rank
    state.ranked_jobs = rank_jobs(
        jobs      = unique_jobs,
        profile   = state.candidate_profile,
        confirmed = state.confirmed_profiles,
    )

    state.results_ready = True
    state.error         = None

    logger.info(
        f"[ranker] Complete — "
        f"{len(state.ranked_jobs)} ranked jobs ready"
    )
    return state


# ── LangGraph node wrapper ────────────────────────────────────────────────────

def node_ranker(state: dict) -> dict:
    """LangGraph node wrapper for the ranker agent."""
    session = SessionState(**state)

    if not session.raw_jobs:
        logger.warning("[graph] Skipping ranker — no raw jobs")
        return state

    updated = run_ranker_agent(session)
    return updated.model_dump()
