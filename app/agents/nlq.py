"""
Natural Language Query (NLQ) Agent — Phase 5+, Read-Only CFO Assistant.

================================================================================
PURPOSE
================================================================================
Lets a CFO type plain English ("Why is SG&A up 15% at TechForge?") and get
an answer grounded in the database. Read-only, deterministic tools.

REUSE-FIRST DESIGN (critical)
-----------------------------
The Variance and Trial Balance agents already contain the exact deterministic
math a CFO question needs. Rather than duplicate that logic, the NLQ agent
exposes those SAME functions as tools. This guarantees:

    - Consistency: the CFO sees the same numbers the pipeline produced.
    - Zero drift:  fix a rule in variance.py → NLQ answers update automatically.
    - No new math: the LLM can only call what already exists.

Only TWO thin DB-lookup tools are NEW (not duplication):
    - `list_companies()`             — maps "TechForge" → "techforge_saas"
    - `list_accounts_for_company()`  — maps "SG&A" → actual account codes
                                       (charts of accounts vary per entity)

SAFETY
------
- All tools are SELECT-only — no INSERT/UPDATE/DELETE anywhere.
- The LLM is instructed to NEVER invent numbers; every figure must come
  from a tool return value.
- Agent answers are free-form text (not JSON) — this is a conversational
  assistant, not a structured pipeline step.
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Company, TrialBalance
# ---- Reuse existing deterministic agents (no duplicate math) --------------
from app.agents.variance import analyze_variances
from app.agents.validator import analyze_trial_balance

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL 1 — Company lookup (thin DB query)
# =============================================================================

def list_companies() -> list[dict[str, Any]]:
    """
    Return the list of portfolio companies as {id, name, industry, revenue_annual}.

    The LLM calls this first when a question references a company by name
    (e.g., "TechForge"), because the DB primary key is a slug like
    "techforge_saas" — not a human-readable name.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(select(Company).order_by(Company.name)).all()
        return [
            {
                "id": c.id,
                "name": c.name,
                "industry": c.industry,
                "revenue_annual": float(c.revenue_annual or 0),
            }
            for c in rows
        ]
    finally:
        db.close()


# =============================================================================
# TOOL 2 — Account discovery for a company (thin DB query)
# =============================================================================

def list_accounts_for_company(company_id: str, period: str | None = None) -> list[dict[str, Any]]:
    """
    Return the chart of accounts for a company, optionally narrowed to a period.

    Why this exists:
    Charts of accounts vary per industry. SaaS companies have "Subscription
    Revenue" (4200); manufacturers have "Product Sales" (4000). The CFO says
    "SG&A" or "R&D" — the DB stores numeric account codes and full names. This
    tool bridges the gap so the LLM can find the right accounts.

    Args:
        company_id: The company slug (e.g., "techforge_saas").
        period:     Optional 'YYYY-MM' filter. If None, returns the union
                    across all periods.
    """
    db = SessionLocal()
    try:
        stmt = select(
            TrialBalance.account_code,
            TrialBalance.account_name,
            TrialBalance.account_type,
        ).where(TrialBalance.company_id == company_id)
        if period:
            stmt = stmt.where(TrialBalance.period == period)

        rows = db.execute(stmt.distinct()).all()
        seen: dict[str, dict[str, Any]] = {}
        for code, name, acct_type in rows:
            if code not in seen:
                seen[code] = {
                    "account_code": code,
                    "account_name": name,
                    "account_type": acct_type,
                }
        return sorted(seen.values(), key=lambda x: x["account_code"])
    finally:
        db.close()


# =============================================================================
# REUSED TOOLS — aliased so Agno sees clear, CFO-friendly names
# =============================================================================

def get_trial_balance_analysis(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic trial balance analysis for one company+period.

    Wraps app.agents.validator.analyze_trial_balance — the exact same
    function the pipeline's Phase 1 TB agent uses. Returns totals, balance
    status, and per-account issues.
    """
    return analyze_trial_balance(company_id, period)


def get_variance_analysis(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic actual-vs-budget variance analysis for one company+period.

    Wraps app.agents.variance.analyze_variances — the exact same function
    the pipeline's Phase 1 Variance agent uses. Returns flagged accounts
    with actual, budget, variance, variance_pct, severity.
    """
    return analyze_variances(company_id, period)


# =============================================================================
# AGENT DEFINITION
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are the CFO Assistant for a Private Equity month-end close system.",
    "You answer financial questions about 8 portfolio companies.",
    "",
    "TOOLS YOU HAVE:",
    "  • list_companies()                     — get company IDs and names",
    "  • list_accounts_for_company(...)       — get the chart of accounts",
    "  • get_trial_balance_analysis(...)      — TB totals, balance status, issues",
    "  • get_variance_analysis(...)           — actual vs budget, flagged variances",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. NEVER invent numbers. Every dollar figure, percentage, or count MUST",
    "   come from a tool return value. If a tool didn't return it, don't say it.",
    "2. ALWAYS resolve the company first: call list_companies() and match the",
    "   CFO's named entity (e.g., 'TechForge') to the correct company_id slug",
    "   (e.g., 'techforge_saas'). Never guess the slug.",
    "3. If the question involves a specific account category (SG&A, R&D,",
    "   COGS, rent, etc.), call list_accounts_for_company() to find the",
    "   matching account codes first. Account naming varies by industry.",
    "4. If you need actual vs budget numbers, call get_variance_analysis().",
    "   If you need raw balances or TB-level integrity, call",
    "   get_trial_balance_analysis().",
    "5. If the CFO did not specify a period, default to '2026-01' and say so.",
    "6. If the data doesn't answer the question, say so honestly. Do NOT",
    "   speculate, extrapolate, or fill gaps with plausible-sounding numbers.",
    "7. Keep answers to 3-6 sentences. Lead with the direct answer, then",
    "   supporting figures, then a one-line next action if relevant.",
    "",
    "Tone: concise, numerate, board-ready. No markdown headers, no bullet",
    "spam — plain prose with figures inline.",
]


def _build_agent() -> Agent:
    """Build a fresh Agent per call — no shared state across concurrent users."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot initialise NLQ agent.")

    return Agent(
        name="CFO Assistant",
        model=Gemini(id="gemini-3.5-flash-lite", api_key=settings.gemini_api_key),
        tools=[
            list_companies,
            list_accounts_for_company,
            get_trial_balance_analysis,
            get_variance_analysis,
        ],
        description=(
            "Read-only financial assistant. Answers CFO questions by calling "
            "deterministic query tools. Never invents numbers."
        ),
        instructions=AGENT_INSTRUCTIONS,
        markdown=True,   # chat responses render better with light markdown
    )


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================

def ask_financial_question(query: str) -> str:
    """
    Answer a free-form CFO question using the NLQ agent.

    Args:
        query: Plain-English question, e.g. "Why is SG&A up 15% at TechForge?"

    Returns:
        The agent's answer as plain text (with light markdown). On failure,
        returns a user-friendly error string — never raises, so the Streamlit
        chat UI stays responsive.
    """
    if not query or not query.strip():
        return "Please enter a question."

    try:
        agent = _build_agent()
        response = agent.run(query.strip())
        content = response.content

        # Agno returns a string for text-only agents; guard for safety.
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return content.get("summary") or str(content)
        return str(content)

    except Exception as exc:  # noqa: BLE001
        logger.exception("NLQ agent failed for query %r", query)
        return (
            f"I couldn't complete that request right now ({type(exc).__name__}). "
            f"Please try again, or check the deterministic reports in the dashboard above."
        )