"""
Consolidation Agent — Phase 4, Final Consolidation Group.

================================================================================
BUSINESS PROBLEM
================================================================================
After intercompany elimination (Phase 3), the PE fund needs a SINGLE set of
group financials that a CFO / LP can read. This agent rolls up the trial
balances of all 8 portfolio companies into a group P&L, computes EBITDA, and
adjusts for the intercompany asymmetries surfaced in Phase 3.

WHY ADJUSTED EBITDA (the "PE view")
-----------------------------------
Group EBITDA computed by simply summing each company's EBITDA is MISLEADING
when intercompany activity hasn't reconciled. If Company A booked $500K of
intercompany revenue but Company B only booked $480K on the mirror side,
there's a $20K phantom profit that doesn't exist at group level. PE firms
therefore report:

    Adjusted Group EBITDA = Raw Group EBITDA - Phase 3 elimination asymmetry

This is a conservative haircut that makes the consolidated number defensible
to auditors and LPs, and directly ties the Phase 3 findings into the group
financials.

HYBRID DESIGN (Python Math + LLM Reasoning)
-------------------------------------------
- Python owns ALL arithmetic:
    - Account-type-based bucketing (Revenue, COGS, OpEx)
    - Sign normalization (credit-normal accounts are flipped)
    - EBITDA math
    - Asymmetry adjustment
- Gemini owns ONLY the executive narrative: 2-4 sentences a partner can read
  in 15 seconds. It cites numbers but never produces them.
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Company, TrialBalance

logger = logging.getLogger(__name__)

# Account types that map to P&L buckets. Case-insensitive matching.
REVENUE_TYPES = {"revenue"}
COGS_TYPES = {"cogs"}
OPEX_TYPES = {"operating expense", "expense"}


# =============================================================================
# 1. STRUCTURED OUTPUT SCHEMA
# =============================================================================

class ConsolidationResult(BaseModel):
    """Final structured group financials returned by the agent."""

    period: str
    status: str = Field(..., description="'COMPLETE' if all entities contributed, else 'INCOMPLETE'.")
    entity_count: int = Field(..., description="Number of companies included in the rollup.")
    gross_revenue: float = Field(..., description="Consolidated revenue (sign-normalized, positive).")
    total_cogs: float = Field(..., description="Consolidated cost of goods sold (positive).")
    gross_profit: float = Field(..., description="gross_revenue - total_cogs.")
    total_opex: float = Field(..., description="Consolidated operating expenses (positive).")
    raw_ebitda: float = Field(..., description="gross_profit - total_opex (pre-elimination haircut).")
    elimination_asymmetry: float = Field(
        ..., description="Phase 3 intercompany asymmetry total, used as a conservative EBITDA haircut."
    )
    adjusted_group_ebitda: float = Field(
        ..., description="raw_ebitda - elimination_asymmetry. The number reported to LPs."
    )
    executive_summary: str = Field(..., description="2-4 sentence board-ready narrative.")


# =============================================================================
# 2. DETERMINISTIC ANALYSIS TOOL
# =============================================================================

def _get_elimination_asymmetry(period: str, run_id: str | None) -> Decimal:
    """
    Return the Phase 3 elimination asymmetry for `period`.

    Prefers the value persisted by Phase 3 under `close:{run_id}:phase3:result`
    (fast, reflects the actual run). If unavailable, recomputes deterministically
    from the IC ledger via the Phase 3 tool.
    """
    # ---- Preferred path: read from Redis (fast, exact run result) --------
    if run_id:
        try:
            import redis as _redis
            r = _redis.from_url(settings.redis_url, decode_responses=True)
            raw = r.get(f"close:{run_id}:phase3:result")
            if raw:
                data = json.loads(raw)
                return Decimal(str(data.get("total_asymmetry_usd", 0)))
        except Exception:
            logger.warning("Could not read phase3 result from Redis — recomputing from IC ledger.")

    # ---- Fallback: recompute via the Phase 3 tool ------------------------
    from app.agents.elimination import verify_intercompany_eliminations
    facts = verify_intercompany_eliminations(period)
    return Decimal(str(facts.get("total_asymmetry_usd", 0)))


def generate_consolidated_financials(period: str, run_id: str | None = None) -> dict[str, Any]:
    """
    Deterministic group consolidation for one period.

    ALGORITHM
    ---------
    1. Load all TrialBalance rows for `period` across ALL companies.
    2. Bucket by account_type:
         Revenue          → gross_revenue (flip sign; credit-normal → positive)
         COGS             → total_cogs    (already positive; debit-normal)
         Operating Expense / Expense → total_opex (already positive)
       Assets / Liabilities / Equity are ignored — this is a P&L consolidation.
    3. Compute:
         gross_profit = gross_revenue - total_cogs
         raw_ebitda   = gross_profit - total_opex
    4. Fetch the Phase 3 elimination asymmetry (see _get_elimination_asymmetry).
    5. adjusted_group_ebitda = raw_ebitda - asymmetry.

    This function is exposed to the Agno agent as a tool. It returns raw
    facts only — all narrative is left to the LLM layer.

    Args:
        period: 'YYYY-MM'.
        run_id: optional — used to fetch the exact Phase 3 result from Redis.

    Returns:
        JSON-serialisable dict. Never raises for missing data.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(TrialBalance.period == period)
        ).all()

        if not rows:
            return {
                "period": period, "found": False,
                "error": f"No trial balance rows for period {period}.",
                "status": "INCOMPLETE",
                "entity_count": 0,
                "gross_revenue": 0.0, "total_cogs": 0.0, "gross_profit": 0.0,
                "total_opex": 0.0, "raw_ebitda": 0.0,
                "elimination_asymmetry": 0.0, "adjusted_group_ebitda": 0.0,
            }

        # ---- Bucket by account_type (case-insensitive) -------------------
        gross_revenue = Decimal("0")
        total_cogs = Decimal("0")
        total_opex = Decimal("0")
        entity_ids: set[str] = set()

        for r in rows:
            acct_type = (r.account_type or "").strip().lower()
            balance = r.balance or Decimal("0")
            entity_ids.add(r.company_id)

            if acct_type in REVENUE_TYPES:
                # Credit-normal: our schema stores revenue as negative balance.
                gross_revenue += -balance
            elif acct_type in COGS_TYPES:
                total_cogs += balance
            elif acct_type in OPEX_TYPES:
                total_opex += balance
            # Assets/Liabilities/Equity intentionally ignored — P&L view only.

        gross_profit = gross_revenue - total_cogs
        raw_ebitda = gross_profit - total_opex

        # ---- Phase 3 asymmetry as PE-style EBITDA haircut ----------------
        asymmetry = _get_elimination_asymmetry(period, run_id)
        adjusted_ebitda = raw_ebitda - asymmetry

        # ---- Entity coverage check ---------------------------------------
        all_companies = db.scalar(select(Company.id).limit(1))
        total_companies = len(db.scalars(select(Company.id)).all())
        complete = len(entity_ids) == total_companies and total_companies > 0

        return {
            "period": period,
            "found": True,
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "entity_count": len(entity_ids),
            "total_companies": total_companies,
            "gross_revenue": float(gross_revenue),
            "total_cogs": float(total_cogs),
            "gross_profit": float(gross_profit),
            "total_opex": float(total_opex),
            "raw_ebitda": float(raw_ebitda),
            "elimination_asymmetry": float(asymmetry),
            "adjusted_group_ebitda": float(adjusted_ebitda),
            "note": (
                f"Consolidated {len(entity_ids)}/{total_companies} entities. "
                f"Raw EBITDA ${raw_ebitda:,.2f}, adjusted for "
                f"${asymmetry:,.2f} in elimination asymmetries."
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_consolidated_financials failed for %s", period)
        return {
            "period": period, "found": False,
            "error": str(exc), "status": "INCOMPLETE",
            "entity_count": 0,
            "gross_revenue": 0.0, "total_cogs": 0.0, "gross_profit": 0.0,
            "total_opex": 0.0, "raw_ebitda": 0.0,
            "elimination_asymmetry": 0.0, "adjusted_group_ebitda": 0.0,
        }
    finally:
        db.close()


def _latest_period_globally() -> str | None:
    """Latest period present in trial_balances across all companies."""
    db = SessionLocal()
    try:
        return db.scalar(
            select(TrialBalance.period)
            .order_by(TrialBalance.period.desc())
            .limit(1)
        )
    finally:
        db.close()


# =============================================================================
# 3. AGNO AGENT DEFINITION
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are a Consolidation Agent for a Private Equity month-end close system.",
    "You produce the FINAL group-level financials the fund reports to LPs.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. ALWAYS call `generate_consolidated_financials` FIRST. Never invent numbers.",
    "2. NEVER compute, re-sum, or adjust any dollar figure. The tool is the source of truth.",
    "3. Copy every numeric field verbatim from the tool output into your result.",
    "4. Set status = tool's status ('COMPLETE' or 'INCOMPLETE').",
    "5. Write `executive_summary` as 2-4 sentences for a board deck:",
    "     - Lead with adjusted_group_ebitda (the PE-reported number).",
    "     - Mention gross revenue, gross profit, and raw EBITDA briefly.",
    "     - Explicitly state the elimination asymmetry haircut and why it was applied.",
    "     - If status='INCOMPLETE', warn that not all entities contributed and the",
    "       number should be treated as preliminary.",
    "6. If the tool returns found=false, set status='INCOMPLETE' and explain the",
    "   missing-data reason in `executive_summary`.",
    "",
    "Output ONLY the structured JSON schema. No prose outside the schema.",
]


def _build_agent() -> Agent:
    """Build a fresh Agent per call — no shared state across Celery workers."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot initialise LLM.")

    return Agent(
        name="Consolidation Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[generate_consolidated_financials],
        description=(
            "Produces consolidated group financials for a period: revenue, COGS, "
            "OpEx, raw EBITDA, and intercompany-adjusted group EBITDA."
        ),
        instructions=AGENT_INSTRUCTIONS,
        output_schema=ConsolidationResult,
        markdown=False,
    )


# =============================================================================
# 4. PUBLIC ENTRYPOINT
# =============================================================================

def _safe_parse(content: Any) -> ConsolidationResult:
    """Parse agent response; detect Gemini error payloads cleanly."""
    if isinstance(content, ConsolidationResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return ConsolidationResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return ConsolidationResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_consolidation(
    period: str | None = None,
    run_id: str | None = None,
) -> ConsolidationResult:
    """
    Run the Consolidation Agent for one period.

    Args:
        period: 'YYYY-MM'. If None, resolves to the latest period in the ledger.
        run_id: optional — used to fetch Phase 3's exact asymmetry from Redis.

    Returns:
        ConsolidationResult. Falls back to a Python-only verdict if the LLM is
        unreachable — the orchestrator never blocks on a Gemini rate-limit.
    """
    if period is None:
        period = _latest_period_globally()
        if period is None:
            return ConsolidationResult(
                period="UNKNOWN", status="INCOMPLETE", entity_count=0,
                gross_revenue=0.0, total_cogs=0.0, gross_profit=0.0,
                total_opex=0.0, raw_ebitda=0.0,
                elimination_asymmetry=0.0, adjusted_group_ebitda=0.0,
                executive_summary="No trial balance data available for consolidation.",
            )

    prompt = (
        f"Generate consolidated group financials for period={period!r}. "
        f"Call generate_consolidated_financials with period={period!r} and "
        f"run_id={run_id!r}. Then produce the structured ConsolidationResult."
    )

    # ---- Try LLM path up to 3 times on transient errors ------------------
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            agent = _build_agent()
            response = agent.run(prompt)
            return _safe_parse(response.content)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if any(k in msg for k in ("503", "unavailable", "429", "quota", "timeout", "deadline")) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                logger.warning("Transient error on consolidation attempt %d — retrying.", attempt + 1)
                continue
            break

    # ---- Deterministic fallback ------------------------------------------
    logger.warning(
        "LLM path failed for consolidation %s (%s) — falling back.",
        period, type(last_exc).__name__,
    )
    facts = generate_consolidated_financials(period, run_id=run_id)

    return ConsolidationResult(
        period=period,
        status=facts.get("status", "INCOMPLETE"),
        entity_count=facts.get("entity_count", 0),
        gross_revenue=facts.get("gross_revenue", 0.0),
        total_cogs=facts.get("total_cogs", 0.0),
        gross_profit=facts.get("gross_profit", 0.0),
        total_opex=facts.get("total_opex", 0.0),
        raw_ebitda=facts.get("raw_ebitda", 0.0),
        elimination_asymmetry=facts.get("elimination_asymmetry", 0.0),
        adjusted_group_ebitda=facts.get("adjusted_group_ebitda", 0.0),
        executive_summary=(
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: Group revenue ${facts.get('gross_revenue', 0.0):,.2f}, "
            f"gross profit ${facts.get('gross_profit', 0.0):,.2f}, "
            f"raw EBITDA ${facts.get('raw_ebitda', 0.0):,.2f}, "
            f"adjusted for ${facts.get('elimination_asymmetry', 0.0):,.2f} in "
            f"intercompany asymmetries → adjusted group EBITDA "
            f"${facts.get('adjusted_group_ebitda', 0.0):,.2f}. "
            f"Entities contributing: {facts.get('entity_count', 0)}."
        ),
    )