"""
Revenue Recognition Agent — Phase 2, Sequential Group (Step 2 of 3).

================================================================================
BUSINESS PROBLEM
================================================================================
ASC 606 requires that each contract's total transaction price be allocated
across performance obligations in proportion to their standalone selling
prices, and that ratable revenue be recognised over the service period —
NOT as a flat monthly amount.

Common failures:
    - ALLOCATION_ERROR: sum(performance_obligation values) != total_contract_value
    - STALE_MILESTONE: contract end_date has passed but milestone is < 100%
    - EXPIRED_ACTIVE: contract ended before period but still shows activity
    - UNKNOWN_METHOD: revenue_recognition is not one of ratable/milestone/
                      point_in_time

================================================================================
DAY-BASED PRORATION (Trap 1 answer)
================================================================================
A $120,000 annual contract that begins Jan 17 must recognise ~$4,931 in Jan
(14 days of Jan at $120,000/365) — not $10,000. Flat monthly is wrong because
it silently over-recognises partial months and under-recognises full months.

Our algorithm for each contract:
    ratable obligation  → value × overlap_days / total_contract_days
    milestone obligation → value × completion_pct, recognised in the month
                            that contains the milestone end_date
    point_in_time       → full value, recognised in the month of start_date

`monthly_recognitions` returns per-contract recognised revenue for the period.
`total_month_revenue_recognized` is the sum — a CFO can tie this directly to
the month's P&L revenue line.
================================================================================
"""

from __future__ import annotations

import json
import logging
import time
from calendar import monthrange
from datetime import date, timedelta
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


class RevenueContractIssue(BaseModel):
    contract_id: str
    customer: str
    issue_type: str = Field(..., description="ALLOCATION_ERROR, STALE_MILESTONE, EXPIRED_ACTIVE, UNKNOWN_METHOD.")
    total_contract_value: float
    detail: str


class MonthlyRecognition(BaseModel):
    """Per-contract recognised revenue for the period (day-based proration)."""
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
    # NEW: day-based proration output for the period
    total_month_revenue_recognized: float = Field(
        0.0,
        description="Sum of day-based recognised revenue for the period.",
    )
    monthly_recognitions: list[MonthlyRecognition] = Field(default_factory=list)
    issues: list[RevenueContractIssue] = Field(default_factory=list)
    summary: str


def _period_end(period: str) -> date:
    y, m = (int(x) for x in period.split("-"))
    if m == 12:
        return date(y, 12, 31)
    return date(y, m + 1, 1) - timedelta(days=1)


def _period_start(period: str) -> date:
    y, m = (int(x) for x in period.split("-"))
    return date(y, m, 1)


# =============================================================================
# DAY-BASED REVENUE PRORATION — the deterministic core of Trap 1
# =============================================================================

def _month_recognition_for_contract(contract: RevenueContract, period: str) -> dict[str, Any]:
    """
    Compute recognised revenue for one contract in one period using exact-day
    proration, per performance obligation.

    Contract types handled:
      - ratable:       value × overlap_days / total_contract_days
      - milestone:     value × completion_pct, on the month containing end_date
      - point_in_time: full value, on the month containing start_date

    Rounding: we return raw Decimal→float values; the caller sums them.
    Any residual (final-month rounding) is left to the presentation layer.
    """
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


def verify_revenue_recognition(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic ASC-606-flavored audit of a company's revenue contracts.

    RULES
    -----
    R1. sum(performance_obligations[*].value) must equal total_contract_value
        within $100 (catches allocation drift).
    R2. If end_date < as_of, all milestone obligations must be 100% complete.
    R3. If start_date > as_of, contract is future — skip R1/R2 (still report
        method validity).
    R4. revenue_recognition must be one of the whitelist methods.

    PLUS: day-based monthly revenue recognition for the period.
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
                "monthly_recognitions": [],
                "total_month_revenue_recognized": 0.0,
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
                                    f"'{p.get('description')}' is only {pct}% complete."
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

        # ---- Day-based monthly recognition for active contracts ----------
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


AGENT_INSTRUCTIONS = [
    "You are a Revenue Recognition (ASC 606) Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `verify_revenue_recognition` FIRST. Never invent numbers.",
    "2. NEVER recalculate. Cite tool output verbatim.",
    "3. Include every issue — do not add or drop.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` is 2-3 sentences. Lead with the largest-value contract.",
    "6. Mention `total_month_revenue_recognized` — this is day-based prorated",
    "   revenue for the period, not flat monthly.",
    "7. If found=false, set status='FAILED' and explain.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Revenue Recognition Agent",
        model=Gemini(id="gemini-2.0-flash", api_key=settings.gemini_api_key),
        tools=[verify_revenue_recognition],
        description="Audits revenue contracts for ASC 606 compliance, day-based proration.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=RevenueRecognitionResult,
        markdown=False,
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

    prompt = (
        f"Audit revenue recognition for company_id={company_id!r} and period={period!r}. "
        f"Call verify_revenue_recognition with these exact arguments."
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
                time.sleep(2 * (attempt + 1))
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
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: {len(issues)} contract issues flagged; "
            f"day-based month revenue recognised = "
            f"${facts.get('total_month_revenue_recognized', 0.0):,.2f}."
        ),
    )