"""
Accrual Verification Agent — Phase 2, Sequential Group (Step 1 of 3).

================================================================================
BUSINESS PROBLEM
================================================================================
Accruals are the #1 source of month-end surprises. A quarterly audit fee that
wasn't booked, a monthly rent accrual that's stale by 60 days, a bonus accrual
sitting there since December — these are silent until they hit cash. This agent
audits the `accrual_schedules` table and flags:

    - STALE: last_booked_date is older than the expected cadence implies.
             e.g. monthly accrual whose last booking was 45+ days ago.
    - OVERDUE: frequency=monthly but last_booked_date > 40 days ago.
    - ORPHAN: gl_account doesn't exist in the company's trial balance.
    - ZERO_AMOUNT: amount is 0 or negative for a normal accrual.
    - DUPLICATE: same (accrual_type, gl_account) appears more than once.

"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import AccrualSchedule, TrialBalance

logger = logging.getLogger(__name__)

FREQUENCY_MAX_DAYS = {"monthly": 40, "quarterly": 100, "annual": 380, "weekly": 12}


class AccrualIssue(BaseModel):
    accrual_type: str
    gl_account: str
    issue_type: str = Field(..., description="STALE, OVERDUE, ORPHAN, ZERO_AMOUNT, DUPLICATE.")
    amount: float
    days_since_booked: int
    description: str


class AccrualVerificationResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if no issues, else 'FAILED'.")
    total_accruals: int
    flagged_count: int
    total_flagged_amount: float
    issues: list[AccrualIssue] = Field(default_factory=list)
    summary: str


def _as_of_date(period: str) -> date:
    year, month = (int(x) for x in period.split("-"))
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def verify_accruals(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic audit of the accrual schedule for one company.

    ALGORITHM
    ---------
    1. Load all AccrualSchedule rows for the company.
    2. Compute as_of = last day of `period`.
    3. For each row:
         days_since = (as_of - last_booked_date).days
         max_days   = FREQUENCY_MAX_DAYS.get(frequency, 40)
         - if days_since > max_days → OVERDUE
         - if days_since > max_days * 1.5 → STALE (escalated severity)
         - if amount <= 0 → ZERO_AMOUNT
         - if gl_account missing from company's trial balance → ORPHAN
    4. Detect duplicates by (accrual_type, gl_account).
    5. Return issues sorted by absolute amount.

    Never raises — returns found=False if no data.
    """
    db = SessionLocal()
    try:
        accruals = db.scalars(
            select(AccrualSchedule).where(AccrualSchedule.company_id == company_id)
        ).all()

        if not accruals:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No accrual schedules found.", "total_accruals": 0, "issues": []}

        tb_accounts = {
            code for (code,) in db.execute(
                select(TrialBalance.account_code).where(TrialBalance.company_id == company_id)
            ).all()
        }

        as_of = _as_of_date(period)
        issues: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        total_flagged = Decimal("0")

        for a in accruals:
            amount = a.amount or Decimal("0")
            freq = (a.frequency or "").strip().lower()
            max_days = FREQUENCY_MAX_DAYS.get(freq, 40)
            days_since = (as_of - a.last_booked_date).days if a.last_booked_date else 9999

            if days_since > max_days:
                issue_type = "STALE" if days_since > max_days * 1.5 else "OVERDUE"
                issues.append({
                    "accrual_type": a.accrual_type, "gl_account": a.gl_account,
                    "issue_type": issue_type, "amount": float(amount),
                    "days_since_booked": days_since,
                    "description": f"{freq} accrual last booked {days_since}d ago (expected {max_days}d).",
                })
                total_flagged += amount

            if amount <= 0:
                issues.append({
                    "accrual_type": a.accrual_type, "gl_account": a.gl_account,
                    "issue_type": "ZERO_AMOUNT", "amount": float(amount),
                    "days_since_booked": days_since,
                    "description": f"Accrual amount is non-positive ({amount}).",
                })

            if a.gl_account and a.gl_account not in tb_accounts:
                issues.append({
                    "accrual_type": a.accrual_type, "gl_account": a.gl_account,
                    "issue_type": "ORPHAN", "amount": float(amount),
                    "days_since_booked": days_since,
                    "description": f"GL account '{a.gl_account}' has no matching trial-balance line.",
                })

            key = (a.accrual_type or "", a.gl_account or "")
            if key in seen_keys:
                issues.append({
                    "accrual_type": a.accrual_type, "gl_account": a.gl_account,
                    "issue_type": "DUPLICATE", "amount": float(amount),
                    "days_since_booked": days_since,
                    "description": "Same (accrual_type, gl_account) pair appears more than once.",
                })
            seen_keys.add(key)

        issues.sort(key=lambda x: abs(x["amount"]), reverse=True)

        return {
            "company_id": company_id, "period": period, "found": True,
            "total_accruals": len(accruals), "flagged_count": len(issues),
            "total_flagged_amount": float(total_flagged), "issues": issues,
            "note": f"{len(issues)} accrual issues detected." if issues else "All accruals current.",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("verify_accruals failed for %s@%s", company_id, period)
        return {"company_id": company_id, "period": period, "found": False,
                "error": str(exc), "total_accruals": 0, "issues": []}
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
    "You are an Accrual Verification Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `verify_accruals` FIRST. Never invent numbers.",
    "2. NEVER recalculate. Cite tool output verbatim.",
    "3. Include every issue — do not add or drop.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` is 2-3 sentences. Lead with the largest-amount issue.",
    "6. If found=false, set status='FAILED' and explain.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Accrual Verification Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[verify_accruals],
        description="Audits accrual schedules for staleness, orphans, duplicates.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=AccrualVerificationResult,
        markdown=False,
    )


def _safe_parse(content: Any) -> AccrualVerificationResult:
    if isinstance(content, AccrualVerificationResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return AccrualVerificationResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return AccrualVerificationResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_accrual_verification(company_id: str, period: str | None = None) -> AccrualVerificationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return AccrualVerificationResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                total_accruals=0, flagged_count=0, total_flagged_amount=0.0,
                issues=[], summary=f"No trial balance data for '{company_id}'.",
            )

    prompt = (
        f"Verify accruals for company_id={company_id!r} and period={period!r}. "
        f"Call verify_accruals with these exact arguments."
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

    logger.warning("LLM path failed for accruals %s@%s (%s) — falling back.", company_id, period, type(last_exc).__name__)
    facts = verify_accruals(company_id, period)
    issues = [
        AccrualIssue(
            accrual_type=i["accrual_type"], gl_account=i["gl_account"],
            issue_type=i["issue_type"], amount=i["amount"],
            days_since_booked=i["days_since_booked"], description=i["description"],
        )
        for i in facts.get("issues", [])
    ]
    return AccrualVerificationResult(
        company_id=company_id, period=period,
        status="PASSED" if not issues else "FAILED",
        total_accruals=facts.get("total_accruals", 0),
        flagged_count=len(issues),
        total_flagged_amount=facts.get("total_flagged_amount", 0.0),
        issues=issues,
        summary=(
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: {len(issues)} accrual issues flagged."
        ),
    )