"""
Consolidation Agent — Phase 4, Final Consolidation Group.

================================================================================
BUSINESS PROBLEM
================================================================================
After intercompany elimination (Phase 3), the PE fund needs a SINGLE set of
group financials that a CFO / LP can read. This agent rolls up the trial
balances of all 8 portfolio companies into a group P&L, computes EBITDA, and
applies GAAP consolidation rules — including NETTING intercompany revenue
and intercompany expense so they don't double-count at group level.

================================================================================
GAAP CONSOLIDATION — the "matched IC" elimination
================================================================================
If Company A sells $500K to Company B and B records the mirror $500K, then
at group level this is ZERO economic activity — money never left the group.
GAAP requires that BOTH the $500K of intercompany revenue (on A's books) AND
the $500K of intercompany expense/purchases (on B's books) be eliminated.

We compute the matched portion as sum(min(flow_ab, flow_ba)) for each pair.
This is subtracted from group revenue, and an equivalent amount is subtracted
from group COGS + OpEx (split proportionally).

Unmatched flow (asymmetry) is NOT netted — it represents a real disagreement
between the two entities' books and is booked as a conservative EBITDA haircut.

================================================================================
HYBRID DESIGN (Python Math + LLM Reasoning)
================================================================================
- Python owns ALL arithmetic:
    - Account-type bucketing
    - Intercompany netting (matched portion only)
    - Asymmetry haircut
    - EBITDA math
- Gemini owns ONLY the executive narrative.
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
import os
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Company, TrialBalance

_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")

logger = logging.getLogger(__name__)

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
    gross_revenue: float = Field(..., description="Consolidated revenue (post-IC elimination, positive).")
    total_cogs: float = Field(..., description="Consolidated COGS (post-IC elimination, positive).")
    gross_profit: float = Field(..., description="gross_revenue - total_cogs.")
    total_opex: float = Field(..., description="Consolidated OpEx (post-IC elimination, positive).")
    raw_ebitda: float = Field(..., description="gross_profit - total_opex (pre-asymmetry haircut).")
    # NEW: matched intercompany flow that was netted from revenue AND expense
    eliminated_intercompany_usd: float = Field(
        0.0,
        description="Matched intercompany flow netted from group revenue and expense.",
    )
    elimination_asymmetry: float = Field(
        ..., description="Unmatched IC asymmetry, used as a conservative EBITDA haircut."
    )
    adjusted_group_ebitda: float = Field(
        ..., description="raw_ebitda - elimination_asymmetry. The number reported to LPs."
    )
    executive_summary: str = Field(..., description="2-4 sentence board-ready narrative.")


# =============================================================================
# 2. DETERMINISTIC ANALYSIS TOOL
# =============================================================================

def _get_phase3_totals(period: str, run_id: str | None) -> tuple[Decimal, Decimal]:
    """
    Return (asymmetry_usd, matched_intercompany_usd) for the period.

    Prefers the value persisted by Phase 3 under `close:{run_id}:phase3:result`.
    Falls back to recomputing from the IC ledger via the Phase 3 tool.
    """
    # ---- Preferred path: read from Redis (fast, exact run result) --------
    if run_id:
        try:
            import redis as _redis
            r = _redis.from_url(settings.redis_url, decode_responses=True)
            raw = r.get(f"close:{run_id}:phase3:result")
            if raw:
                data = json.loads(raw)
                return (
                    Decimal(str(data.get("total_asymmetry_usd", 0))),
                    Decimal(str(data.get("matched_intercompany_usd", 0))),
                )
        except Exception:
            logger.warning(
                "Could not read phase3 result from Redis — recomputing from IC ledger."
            )

    # ---- Fallback: recompute via the Phase 3 tool ------------------------
    from app.agents.elimination import verify_intercompany_eliminations
    facts = verify_intercompany_eliminations(period)
    return (
        Decimal(str(facts.get("total_asymmetry_usd", 0))),
        Decimal(str(facts.get("matched_intercompany_usd", 0))),
    )


def generate_consolidated_financials(period: str, run_id: str | None = None) -> dict[str, Any]:
    """
    Deterministic group consolidation for one period.

    ALGORITHM
    ---------
    1. Load all TrialBalance rows for `period` across ALL companies.
    2. Bucket by account_type:
         Revenue                     → gross_revenue (flip sign)
         COGS                        → total_cogs
         Operating Expense / Expense → total_opex
       Assets / Liabilities / Equity are ignored — P&L consolidation only.
    3. Eliminate matched intercompany flow:
         gross_revenue -= matched_ic
         split matched_ic across COGS + OpEx proportionally and subtract.
    4. Compute gross_profit and raw_ebitda from the post-elimination numbers.
    5. Fetch the Phase 3 asymmetry and apply as a conservative EBITDA haircut.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(TrialBalance.period == period)
        ).all()

        if run_id:
            try:
                import redis as _redis
                _rr = _redis.from_url(settings.redis_url, decode_responses=True)
                _companies_raw = _rr.get(f"close:{run_id}:companies")
                if _companies_raw:
                    _allowed = set(json.loads(_companies_raw))
                    rows = [r for r in rows if r.company_id in _allowed]
            except Exception:
                pass

        if not rows:
            return {
                "period": period, "found": False,
                "error": f"No trial balance rows for period {period}.",
                "status": "INCOMPLETE",
                "entity_count": 0,
                "gross_revenue": 0.0, "total_cogs": 0.0, "gross_profit": 0.0,
                "total_opex": 0.0, "raw_ebitda": 0.0,
                "eliminated_intercompany_usd": 0.0,
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

        # ---- Fetch Phase 3 totals: asymmetry + matched IC -----------------
        asymmetry, matched_ic = _get_phase3_totals(period, run_id)

        # ---- Apply GAAP intercompany elimination --------------------------
        # Subtract matched IC flow from revenue, and an equivalent amount
        # from the expense side (split proportionally between COGS and OpEx).
        gross_revenue -= matched_ic

        combined_expense = total_cogs + total_opex
        if combined_expense > 0:
            cogs_share = matched_ic * (total_cogs / combined_expense)
            total_cogs -= cogs_share
            total_opex -= (matched_ic - cogs_share)
        else:
            # No expense to offset (rare) — put it all against OpEx
            total_opex -= matched_ic

        # ---- Compute post-elimination P&L --------------------------------
        gross_profit = gross_revenue - total_cogs
        raw_ebitda = gross_profit - total_opex

        # ---- Apply asymmetry haircut (conservative LP view) --------------
        adjusted_ebitda = raw_ebitda - asymmetry

        # ---- Entity coverage check ---------------------------------------
                # Match the run's filter — if we're demoing 4 companies, expect 4.
        if run_id:
            try:
                import redis as _redis
                _rr = _redis.from_url(settings.redis_url, decode_responses=True)
                _companies_raw = _rr.get(f"close:{run_id}:companies")
                if _companies_raw:
                    total_companies = len(json.loads(_companies_raw))
                else:
                    total_companies = len(db.scalars(select(Company.id)).all())
            except Exception:
                total_companies = len(db.scalars(select(Company.id)).all())
        else:
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
            "eliminated_intercompany_usd": float(matched_ic),
            "elimination_asymmetry": float(asymmetry),
            "adjusted_group_ebitda": float(adjusted_ebitda),
            "note": (
                f"Consolidated {len(entity_ids)}/{total_companies} entities. "
                f"Eliminated ${matched_ic:,.2f} of matched intercompany flow; "
                f"applied ${asymmetry:,.2f} asymmetry haircut. "
                f"Adjusted group EBITDA ${adjusted_ebitda:,.2f}."
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
            "eliminated_intercompany_usd": 0.0,
            "elimination_asymmetry": 0.0, "adjusted_group_ebitda": 0.0,
        }
    finally:
        db.close()

# =============================================================================
# TOOL 2 — get_entity_breakdown
# =============================================================================

def get_entity_breakdown(period: str) -> list[dict[str, Any]]:
    """
    Return per-entity revenue/COGS/OpEx totals for the period.

    Use this to see which entity contributes most to group revenue or
    drives the biggest expense lines. Helps explain the group number.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(TrialBalance.period == period)
        ).all()

        by_entity: dict[str, dict[str, Decimal]] = {}
        for r in rows:
            eid = r.company_id
            if eid not in by_entity:
                by_entity[eid] = {"revenue": Decimal("0"),
                                  "cogs": Decimal("0"),
                                  "opex": Decimal("0")}
            acct_type = (r.account_type or "").strip().lower()
            bal = r.balance or Decimal("0")
            if acct_type == "revenue":
                by_entity[eid]["revenue"] += -bal
            elif acct_type == "cogs":
                by_entity[eid]["cogs"] += bal
            elif acct_type in ("operating expense", "expense"):
                by_entity[eid]["opex"] += bal

        out = []
        for eid, t in by_entity.items():
            gp = t["revenue"] - t["cogs"]
            out.append({
                "entity_id": eid,
                "gross_revenue": float(t["revenue"]),
                "total_cogs": float(t["cogs"]),
                "gross_profit": float(gp),
                "total_opex": float(t["opex"]),
                "raw_ebitda": float(gp - t["opex"]),
            })
        out.sort(key=lambda x: x["gross_revenue"], reverse=True)
        return out
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_ic_eliminations
# =============================================================================

def get_ic_eliminations(period: str, run_id: str | None = None) -> dict[str, Any]:
    """
    Return the intercompany amounts that will be netted in consolidation.

    Use this to explain to the LLM what the elimination actually does —
    the matched IC flow is subtracted from both group revenue and group
    expense; the asymmetry is booked as an EBITDA haircut.
    """
    asymmetry, matched = _get_phase3_totals(period, run_id)
    return {
        "matched_intercompany_usd": float(matched),
        "elimination_asymmetry_usd": float(asymmetry),
        "note": (f"Consolidation will net ${matched:,.2f} from both group "
                 f"revenue and group expense, and apply a ${asymmetry:,.2f} "
                 f"EBITDA haircut for the unmatched asymmetry."),
    }

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
    "You are a Consolidation Agent for a PE month-end close system.",
    "You have THREE tools. Reason about which to call.",
    "",
    "TOOLS:",
    "  1. generate_consolidated_financials(period, run_id) — ALWAYS first.",
    "  2. get_entity_breakdown(period) — per-entity P&L. Use to explain",
    "     which entities dominate group revenue.",
    "  3. get_ic_eliminations(period, run_id) — the IC amounts being netted.",
    "",
    "WORKFLOW:",
    "  - Step 1: generate_consolidated_financials.",
    "  - Step 2: get_ic_eliminations to see the netting amounts.",
    "  - Step 3: Optionally get_entity_breakdown for context.",
    "  - Step 4: STOP after 2-4 tool calls.",
    "",
    "STRICT RULES:",
    "1. NEVER invent numbers.",
    "2. Copy every numeric field verbatim from generate_consolidated_financials.",
    "3. `executive_summary` = 2-4 sentences leading with adjusted_group_ebitda.",
    "4. Mention the IC netting AND the asymmetry haircut explicitly.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    """Build the Consolidation Agent."""
    return Agent(
        name="Consolidation Agent",
        model=get_model(),
        tools=[
            generate_consolidated_financials,
            get_entity_breakdown,
            get_ic_eliminations,
        ],
        description="Produces consolidated group financials with GAAP IC elimination.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=ConsolidationResult,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
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

    Falls back to a Python-only verdict if the LLM is unreachable — the
    orchestrator never blocks on a Gemini rate-limit.
    """
    if period is None:
        period = _latest_period_globally()
        if period is None:
            return ConsolidationResult(
                period="UNKNOWN", status="INCOMPLETE", entity_count=0,
                gross_revenue=0.0, total_cogs=0.0, gross_profit=0.0,
                total_opex=0.0, raw_ebitda=0.0,
                eliminated_intercompany_usd=0.0,
                elimination_asymmetry=0.0, adjusted_group_ebitda=0.0,
                executive_summary="No trial balance data available for consolidation.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = generate_consolidated_financials(period, run_id=run_id)
    if (
        precheck.get("found")
        and precheck.get("status") == "COMPLETE"
        and precheck.get("elimination_asymmetry", 0) == 0
    ):
        logger.info("[Consolidation] %s clean — skipping LLM.", period)
        return ConsolidationResult(
            period=period, status="COMPLETE",
            entity_count=precheck["entity_count"],
            gross_revenue=precheck["gross_revenue"],
            total_cogs=precheck["total_cogs"],
            gross_profit=precheck["gross_profit"],
            total_opex=precheck["total_opex"],
            raw_ebitda=precheck["raw_ebitda"],
            eliminated_intercompany_usd=precheck["eliminated_intercompany_usd"],
            elimination_asymmetry=precheck["elimination_asymmetry"],
            adjusted_group_ebitda=precheck["adjusted_group_ebitda"],
            executive_summary=(
                f"Group consolidated across {precheck['entity_count']} entities. "
                f"Gross revenue ${precheck['gross_revenue']:,.2f}, gross profit "
                f"${precheck['gross_profit']:,.2f}, raw EBITDA "
                f"${precheck['raw_ebitda']:,.2f}. Eliminated "
                f"${precheck['eliminated_intercompany_usd']:,.2f} of matched IC. "
                f"Adjusted group EBITDA ${precheck['adjusted_group_ebitda']:,.2f}. "
                f"LLM narrative skipped — no issues to explain."
            ),
        )

    prompt = (
    f"Generate consolidated group financials for period={period!r}. "
    f"Use the tools to understand the IC elimination being applied, then "
    f"produce the structured ConsolidationResult."
)

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
                import random as _rnd; time.sleep((5 + _rnd.random() * 3) * (attempt + 1))
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
        eliminated_intercompany_usd=facts.get("eliminated_intercompany_usd", 0.0),
        elimination_asymmetry=facts.get("elimination_asymmetry", 0.0),
        adjusted_group_ebitda=facts.get("adjusted_group_ebitda", 0.0),
        executive_summary=(
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: Group revenue ${facts.get('gross_revenue', 0.0):,.2f}, "
            f"gross profit ${facts.get('gross_profit', 0.0):,.2f}, "
            f"raw EBITDA ${facts.get('raw_ebitda', 0.0):,.2f}. "
            f"Eliminated ${facts.get('eliminated_intercompany_usd', 0.0):,.2f} of "
            f"matched intercompany flow, plus a "
            f"${facts.get('elimination_asymmetry', 0.0):,.2f} asymmetry haircut "
            f"→ adjusted group EBITDA "
            f"${facts.get('adjusted_group_ebitda', 0.0):,.2f}. "
            f"Entities contributing: {facts.get('entity_count', 0)}."
        ),
    )