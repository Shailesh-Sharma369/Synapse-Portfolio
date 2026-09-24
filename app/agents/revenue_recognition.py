"""
Revenue Recognition Agent — Phase 2, Sequential Group (Step 2 of 3).

================================================================================
REACT PATTERN
================================================================================
Three tools, LLM decides which to call:

    1. verify_revenue_recognition   — full audit + day-based monthly proration
    2. get_contract_detail          — one contract, full detail (drill-down)
    3. compute_contract_month_rev   — single-contract proration check

LLM's job: reason about allocation errors, distinguish stale vs active
contracts, and explain the day-based proration in narrative.

================================================================================
DAY-BASED PRORATION (Trap 1 answer)
================================================================================
A $120,000 annual contract beginning Jan 17 must recognise ~$4,931 in Jan
(14 days of Jan at $120,000/365). Flat monthly ($10,000) is wrong.

For each ratable obligation:
    overlap_days = max(0, min(end_date, period_end) - max(start_date, period_start))
    recognised = value × overlap_days / (end_date - start_date)
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from calendar import monthrange
from datetime import date, timedelta
from app.core.llm import get_model
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import RevenueContract, TrialBalance

logger = logging.getLogger(__name__)

ALLOCATION_TOLERANCE_USD = Decimal("100")
VALID_METHODS = {"ratable", "milestone", "point_in_time"}
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class RevenueContractIssue(BaseModel):
    contract_id: str
    customer: str
    issue_type: str = Field(
        ..., description="ALLOCATION_ERROR, STALE_MILESTONE, EXPIRED_ACTIVE, UNKNOWN_METHOD."
    )
    total_contract_value: float
    detail: str


class MonthlyRecognition(BaseModel):
    contract_id: str
    customer: str
    period: str
    recognized_revenue: float
    days_in_month: int


class RevenueRecognitionResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if no issues, else 'FAILED'.")
    contracts_examined: int
    contracts_active_in_period: int
    flagged_count: int
    flagged_value: float
    total_month_revenue_recognized: float = Field(0.0)
    monthly_recognitions: list[MonthlyRecognition] = Field(default_factory=list)
    issues: list[RevenueContractIssue] = Field(default_factory=list)
    summary: str = Field(
        ...,
        description=(
            "3-5 sentence analytical narrative. Lead with the largest-value "
            "contract issue. Mention the day-based month revenue if notable."
        ),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _period_end(period: str) -> date:
    y, m = (int(x) for x in period.split("-"))
    if m == 12:
        return date(y, 12, 31)
    return date(y, m + 1, 1) - timedelta(days=1)


def _period_start(period: str) -> date:
    y, m = (int(x) for x in period.split("-"))
    return date(y, m, 1)


def _month_recognition_for_contract(
    contract: RevenueContract, period: str
) -> dict[str, Any]:
    """Day-based ASC 606 proration for one contract in one period."""
    y, m = (int(x) for x in period.split("-"))
    period_start = date(y, m, 1)
    period_end = date(y, m, monthrange(y, m)[1])
    days_in_month = monthrange(y, m)[1]

    obligations = contract.performance_obligations or []
    recognized = Decimal("0")

    for po in obligations:
        po_value = Decimal(str(po.get("value", 0)))
        method = po.get("revenue_recognition")

        if method == "ratable":
            total_days = (contract.end_date - contract.start_date).days + 1
            if total_days <= 0:
                continue
            overlap_start = max(contract.start_date, period_start)
            overlap_end = min(contract.end_date, period_end)
            overlap_days = max((overlap_end - overlap_start).days + 1, 0)
            if overlap_days:
                recognized += po_value * Decimal(overlap_days) / Decimal(total_days)

        elif method == "milestone":
            pct = Decimal(str(po.get("completion_percentage", 0))) / Decimal(100)
            if contract.end_date and period_start <= contract.end_date <= period_end:
                recognized += po_value * pct

        elif method == "point_in_time":
            if contract.start_date and period_start <= contract.start_date <= period_end:
                recognized += po_value

    return {
        "contract_id": contract.contract_id,
        "customer": contract.customer,
        "period": period,
        "recognized_revenue": float(recognized),
        "days_in_month": days_in_month,
    }


# =============================================================================
# TOOL 1 — verify_revenue_recognition  (baseline + month proration)
# =============================================================================

def verify_revenue_recognition(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic ASC-606-flavored audit + day-based monthly revenue recognition.

    RULES:
      R1. sum(performance_obligation values) must equal TCV within $100
      R2. Ended contracts must have 100% milestone completion
      R3. revenue_recognition must be in the whitelist
    PLUS: day-based proration for each active contract in the period.
    """
    db = SessionLocal()
    try:
        contracts = db.scalars(
            select(RevenueContract).where(RevenueContract.company_id == company_id)
        ).all()

        if not contracts:
            return {
                "company_id": company_id, "period": period, "found": False,
                "error": "No revenue contracts found.",
                "contracts_examined": 0, "issues": [],
                "monthly_recognitions": [], "total_month_revenue_recognized": 0.0,
            }

        as_of = _period_end(period)
        p_start = _period_start(period)
        issues: list[dict[str, Any]] = []
        active_count = 0
        flagged_value = Decimal("0")

        for c in contracts:
            tcv = c.total_contract_value or Decimal("0")
            pos = c.performance_obligations or []
            start = c.start_date
            end = c.end_date

            if start and end and start <= as_of and end >= p_start:
                active_count += 1

            if pos:
                po_sum = sum(Decimal(str(p.get("value", 0))) for p in pos)
                if abs(po_sum - tcv) > ALLOCATION_TOLERANCE_USD:
                    flagged_value += tcv
                    issues.append({
                        "contract_id": c.contract_id, "customer": c.customer,
                        "issue_type": "ALLOCATION_ERROR",
                        "total_contract_value": float(tcv),
                        "detail": (
                            f"Obligations sum to {po_sum} but TCV is {tcv} "
                            f"(unallocated {tcv - po_sum})."
                        ),
                    })

            if end and end < as_of:
                for p in pos:
                    if p.get("revenue_recognition") == "milestone":
                        pct = int(p.get("completion_percentage", 100))
                        if pct < 100:
                            flagged_value += tcv
                            issues.append({
                                "contract_id": c.contract_id, "customer": c.customer,
                                "issue_type": "STALE_MILESTONE",
                                "total_contract_value": float(tcv),
                                "detail": (
                                    f"Contract ended {end} but milestone "
                                    f"'{p.get('description')}' is {pct}% complete."
                                ),
                            })

            for p in pos:
                method = p.get("revenue_recognition")
                if method not in VALID_METHODS:
                    issues.append({
                        "contract_id": c.contract_id, "customer": c.customer,
                        "issue_type": "UNKNOWN_METHOD",
                        "total_contract_value": float(tcv),
                        "detail": f"revenue_recognition='{method}' invalid.",
                    })

        issues.sort(key=lambda x: abs(x["total_contract_value"]), reverse=True)

        monthly_recognitions: list[dict[str, Any]] = []
        for c in contracts:
            if c.start_date and c.end_date and c.start_date <= as_of and c.end_date >= p_start:
                monthly_recognitions.append(_month_recognition_for_contract(c, period))

        total_month_revenue = sum(
            (Decimal(str(r["recognized_revenue"])) for r in monthly_recognitions),
            Decimal("0"),
        )

        return {
            "company_id": company_id, "period": period, "found": True,
            "contracts_examined": len(contracts),
            "contracts_active_in_period": active_count,
            "flagged_count": len(issues),
            "flagged_value": float(flagged_value),
            "issues": issues,
            "monthly_recognitions": monthly_recognitions,
            "total_month_revenue_recognized": float(total_month_revenue),
            "note": (
                f"{len(issues)} contract issues detected."
                if issues else "All contracts compliant."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("verify_revenue_recognition failed for %s@%s", company_id, period)
        return {
            "company_id": company_id, "period": period, "found": False,
            "error": str(exc), "contracts_examined": 0, "issues": [],
            "monthly_recognitions": [], "total_month_revenue_recognized": 0.0,
        }
    finally:
        db.close()


# =============================================================================
# TOOL 2 — get_contract_detail  (single-contract drill-down)
# =============================================================================

def get_contract_detail(contract_id: str) -> dict[str, Any]:
    """
    Return the full detail for one contract.

    Use this after `verify_revenue_recognition` flags a specific contract
    to see its obligations, billing schedule, and dates in full.

    Returns:
        Full contract dict, or {found: False, error: ...}.
    """
    db = SessionLocal()
    try:
        c = db.scalar(
            select(RevenueContract).where(RevenueContract.contract_id == contract_id)
        )
        if c is None:
            return {"found": False, "contract_id": contract_id, "error": "Not found."}
        return {
            "found": True,
            "contract_id": c.contract_id,
            "company_id": c.company_id,
            "customer": c.customer,
            "start_date": c.start_date.isoformat(),
            "end_date": c.end_date.isoformat(),
            "total_contract_value": float(c.total_contract_value or 0),
            "billing_schedule": c.billing_schedule,
            "performance_obligations": c.performance_obligations or [],
        }
    finally:
        db.close()


# =============================================================================
# TOOL 3 — compute_contract_month_revenue  (single-contract proration)
# =============================================================================

def compute_contract_month_revenue(
    contract_id: str, period: str
) -> dict[str, Any]:
    """
    Day-based proration for ONE contract for ONE period.

    Use this to verify that a specific contract's month revenue is computed
    correctly (not flat monthly). Handy when the CFO asks "how much did we
    recognise from contract X this month?".

    Returns:
        Same shape as `_month_recognition_for_contract`, or {found: False}.
    """
    db = SessionLocal()
    try:
        c = db.scalar(
            select(RevenueContract).where(RevenueContract.contract_id == contract_id)
        )
        if c is None:
            return {"found": False, "contract_id": contract_id, "error": "Not found."}
        result = _month_recognition_for_contract(c, period)
        result["found"] = True
        # Add per-day detail so the LLM can explain the math
        total_days = (c.end_date - c.start_date).days + 1
        result["contract_total_days"] = total_days
        result["daily_rate"] = (
            float(c.total_contract_value or 0) / total_days if total_days > 0 else 0
        )
        return result
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    db = SessionLocal()
    try:
        return db.scalar(
            select(TrialBalance.period)
            .where(TrialBalance.company_id == company_id)
            .order_by(TrialBalance.period.desc())
            .limit(1)
        )
    finally:
        db.close()


# =============================================================================
# AGENT DEFINITION — ReAct
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are a Revenue Recognition (ASC 606) Agent for a Private Equity month-end close system.",
    "You have THREE tools. Reason about which to call and in what order.",
    "",
    "TOOLS:",
    "  1. verify_revenue_recognition(company_id, period)",
    "       — Full audit + day-based monthly proration for the period.",
    "       ALWAYS call this first.",
    "",
    "  2. get_contract_detail(contract_id)",
    "       — Full detail for one contract (obligations, dates, TCV).",
    "       Use this on the LARGEST flagged contract to understand its",
    "       structure (billing schedule, obligation mix).",
    "",
    "  3. compute_contract_month_revenue(contract_id, period)",
    "       — Day-based proration for one contract, with daily_rate and",
    "         contract_total_days so you can explain the math.",
    "       Use this to spot-check that proration is correct.",
    "",
    "WORKFLOW (guideline):",
    "  - Step 1: Call verify_revenue_recognition.",
    "  - Step 2: If a large contract is flagged, call get_contract_detail",
    "            on it to see its obligation structure.",
    "  - Step 3: Optionally call compute_contract_month_revenue on 1-2",
    "            interesting contracts to confirm day-based proration.",
    "  - Step 4: STOP and synthesize.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. NEVER invent numbers.",
    "2. NEVER recalculate — cite tool values verbatim.",
    "3. Include EVERY issue from verify_revenue_recognition.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` is 3-5 sentences AND analytical:",
    "     - Lead with the largest-value contract issue.",
    "     - Cite total_month_revenue_recognized and note it's day-based",
    "       (not flat monthly).",
    "     - If you called get_contract_detail or compute_contract_month_revenue,",
    "       mention what you found.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    """Build the Revenue Recognition Agent."""
    return Agent(
        name="Revenue Recognition Agent",
        model=get_model(),
        tools=[
            verify_revenue_recognition,
            get_contract_detail,
            compute_contract_month_revenue,
        ],
        description="Audits revenue contracts for ASC 606 compliance with day-based proration.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=RevenueRecognitionResult,
        markdown=False,
        use_json_mode=True,
        
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
    )


def _safe_parse(content: Any) -> RevenueRecognitionResult:
    if isinstance(content, RevenueRecognitionResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return RevenueRecognitionResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return RevenueRecognitionResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_revenue_recognition(
    company_id: str, period: str | None = None
) -> RevenueRecognitionResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return RevenueRecognitionResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                contracts_examined=0, contracts_active_in_period=0,
                flagged_count=0, flagged_value=0.0, issues=[],
                total_month_revenue_recognized=0.0, monthly_recognitions=[],
                summary=f"No trial balance data for '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = verify_revenue_recognition(company_id, period)
    if precheck.get("found") and precheck.get("flagged_count", 0) == 0:
        logger.info("[RevRec] %s@%s clean — skipping LLM.", company_id, period)
        monthly = [
            MonthlyRecognition(**r) for r in precheck.get("monthly_recognitions", [])
        ]
        return RevenueRecognitionResult(
            company_id=company_id, period=period, status="PASSED",
            contracts_examined=precheck.get("contracts_examined", 0),
            contracts_active_in_period=precheck.get("contracts_active_in_period", 0),
            flagged_count=0, flagged_value=0.0,
            total_month_revenue_recognized=precheck.get("total_month_revenue_recognized", 0.0),
            monthly_recognitions=monthly, issues=[],
            summary=(
                f"All {precheck.get('contracts_examined', 0)} contracts compliant "
                f"with ASC 606. Day-based month revenue recognised = "
                f"${precheck.get('total_month_revenue_recognized', 0.0):,.2f}. "
                f"LLM reasoning skipped."
            ),
        )

    prompt = (
        f"Audit revenue recognition for company_id={company_id!r} and "
        f"period={period!r}. Use the tools to understand the structure of "
        f"any flagged contracts, then produce the structured result."
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
                continue
            break

    logger.warning(
        "LLM path failed for revrec %s@%s (%s) — falling back.",
        company_id, period, type(last_exc).__name__,
    )
    facts = verify_revenue_recognition(company_id, period)
    issues = [
        RevenueContractIssue(
            contract_id=i["contract_id"], customer=i["customer"],
            issue_type=i["issue_type"],
            total_contract_value=i["total_contract_value"], detail=i["detail"],
        )
        for i in facts.get("issues", [])
    ]
    monthly = [
        MonthlyRecognition(**r) for r in facts.get("monthly_recognitions", [])
    ]
    return RevenueRecognitionResult(
        company_id=company_id, period=period,
        status="PASSED" if not issues else "FAILED",
        contracts_examined=facts.get("contracts_examined", 0),
        contracts_active_in_period=facts.get("contracts_active_in_period", 0),
        flagged_count=len(issues),
        flagged_value=facts.get("flagged_value", 0.0),
        total_month_revenue_recognized=facts.get("total_month_revenue_recognized", 0.0),
        monthly_recognitions=monthly,
        issues=issues,
        summary=(
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: {len(issues)} contract issues flagged; "
            f"day-based month revenue recognised = "
            f"${facts.get('total_month_revenue_recognized', 0.0):,.2f}."
        ),
    )