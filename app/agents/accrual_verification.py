"""
Accrual Verification Agent — Phase 2, Sequential Group (Step 1 of 3).

================================================================================
REACT PATTERN
================================================================================
Three tools, LLM decides which to call based on what it sees:

    1. verify_accruals             — baseline (stale, zero, orphan, duplicate)
    2. list_accruals_by_cadence    — drill into monthly/quarterly/annual bucket
    3. check_gl_account_in_tb      — confirm whether a GL account exists

The LLM's job is to explain WHY the accruals are problematic — not just
list them. E.g., "3 of 4 monthly accruals are stale by 45+ days" is far
more actionable than "4 stale accruals found."

================================================================================
BUSINESS PROBLEM
================================================================================
Accruals are the #1 source of month-end surprises. Silent failure modes:
    - STALE:    last_booked_date older than the cadence implies
    - OVERDUE:  monthly accrual not booked in 40+ days
    - ORPHAN:   gl_account doesn't exist in the company's TB
    - ZERO:     amount is 0 or negative for a normal accrual
    - DUPLICATE: same (accrual_type, gl_account) appears more than once
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import AccrualSchedule, TrialBalance

logger = logging.getLogger(__name__)

FREQUENCY_MAX_DAYS = {"monthly": 40, "quarterly": 100, "annual": 380, "weekly": 12}
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class AccrualIssue(BaseModel):
    accrual_type: str
    gl_account: str
    issue_type: str = Field(
        ..., description="STALE, OVERDUE, ORPHAN, ZERO_AMOUNT, DUPLICATE."
    )
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
    summary: str = Field(
        ...,
        description=(
            "3-5 sentence analytical narrative. Lead with the largest-amount "
            "issue. Note any cadence pattern you noticed."
        ),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _as_of_date(period: str) -> date:
    """Return the last day of the given YYYY-MM period."""
    year, month = (int(x) for x in period.split("-"))
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


# =============================================================================
# TOOL 1 — verify_accruals  (primary analysis, unchanged)
# =============================================================================

def verify_accruals(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic audit of the accrual schedule for one company.

    Loads all AccrualSchedule rows, computes days_since_booked against
    the last day of `period`, and flags:
        - OVERDUE / STALE   (based on frequency cadence)
        - ZERO_AMOUNT
        - ORPHAN            (gl_account not in company's TB)
        - DUPLICATE         (same accrual_type + gl_account twice)
    """
    db = SessionLocal()
    try:
        accruals = db.scalars(
            select(AccrualSchedule).where(AccrualSchedule.company_id == company_id)
        ).all()

        if not accruals:
            return {
                "company_id": company_id, "period": period, "found": False,
                "error": "No accrual schedules found.",
                "total_accruals": 0, "issues": [],
            }

        tb_accounts = {
            code for (code,) in db.execute(
                select(TrialBalance.account_code)
                .where(TrialBalance.company_id == company_id)
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
                    "description": (
                        f"{freq} accrual last booked {days_since}d ago "
                        f"(expected ≤ {max_days}d)."
                    ),
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
                    "description": (
                        f"GL account '{a.gl_account}' has no matching "
                        f"trial-balance line."
                    ),
                })

            key = (a.accrual_type or "", a.gl_account or "")
            if key in seen_keys:
                issues.append({
                    "accrual_type": a.accrual_type, "gl_account": a.gl_account,
                    "issue_type": "DUPLICATE", "amount": float(amount),
                    "days_since_booked": days_since,
                    "description": (
                        "Same (accrual_type, gl_account) pair appears "
                        "more than once."
                    ),
                })
            seen_keys.add(key)

        issues.sort(key=lambda x: abs(x["amount"]), reverse=True)

        return {
            "company_id": company_id, "period": period, "found": True,
            "total_accruals": len(accruals), "flagged_count": len(issues),
            "total_flagged_amount": float(total_flagged), "issues": issues,
            "note": (
                f"{len(issues)} accrual issues detected."
                if issues else "All accruals current."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("verify_accruals failed for %s@%s", company_id, period)
        return {
            "company_id": company_id, "period": period, "found": False,
            "error": str(exc), "total_accruals": 0, "issues": [],
        }
    finally:
        db.close()


# =============================================================================
# TOOL 2 — list_accruals_by_cadence  (drill-down)
# =============================================================================

def list_accruals_by_cadence(
    company_id: str, cadence: str
) -> list[dict[str, Any]]:
    """
    List all accruals matching a given frequency ('monthly', 'quarterly',
    'annual', 'weekly').

    Use this AFTER `verify_accruals` to drill into a specific cadence and
    see whether the problem is concentrated (e.g., all monthly accruals
    stale) or diffuse.

    Returns:
        List of {accrual_type, gl_account, frequency, amount, last_booked_date}.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(AccrualSchedule).where(
                AccrualSchedule.company_id == company_id,
                AccrualSchedule.frequency.ilike(cadence),
            )
        ).all()
        return [
            {
                "accrual_type": r.accrual_type,
                "gl_account": r.gl_account,
                "frequency": r.frequency,
                "amount": float(r.amount or 0),
                "last_booked_date": (
                    r.last_booked_date.isoformat() if r.last_booked_date else None
                ),
            }
            for r in rows
        ]
    finally:
        db.close()


# =============================================================================
# TOOL 3 — check_gl_account_in_tb  (single lookup)
# =============================================================================

def check_gl_account_in_tb(company_id: str, gl_account: str) -> dict[str, Any]:
    """
    Check whether a GL account exists in this company's trial balance.

    Use this to confirm ORPHAN findings from `verify_accruals` — a single
    focused lookup so the LLM can state with certainty whether the account
    is missing or merely spelled differently.

    Returns:
        {exists: bool, gl_account: str, account_name: str | None}
    """
    db = SessionLocal()
    try:
        row = db.scalar(
            select(TrialBalance.account_name).where(
                TrialBalance.company_id == company_id,
                TrialBalance.account_code == gl_account,
            ).limit(1)
        )
        return {
            "exists": row is not None,
            "gl_account": gl_account,
            "account_name": row,
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


# =============================================================================
# AGENT DEFINITION — ReAct
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are an Accrual Verification Agent for a Private Equity month-end close system.",
    "You have THREE tools. Reason about which to call and in what order.",
    "",
    "TOOLS:",
    "  1. verify_accruals(company_id, period)",
    "       — Baseline: stale, zero, orphan, duplicate accruals.",
    "       ALWAYS call this first.",
    "",
    "  2. list_accruals_by_cadence(company_id, cadence)",
    "       — List accruals with a specific frequency",
    "         ('monthly', 'quarterly', 'annual', 'weekly').",
    "       Use this to see whether a problem is concentrated in one",
    "       cadence bucket (e.g., all monthly accruals stale).",
    "",
    "  3. check_gl_account_in_tb(company_id, gl_account)",
    "       — Confirm whether a single GL account exists in the TB.",
    "       Use this to double-check ORPHAN findings.",
    "",
    "WORKFLOW (guideline, not a script):",
    "  - Step 1: Call verify_accruals.",
    "  - Step 2: If stale/overdue accruals exist, drill into their cadence",
    "            with list_accruals_by_cadence to see if it's a pattern.",
    "  - Step 3: If ORPHAN issues exist, use check_gl_account_in_tb to",
    "            confirm (one or two accounts, not all).",
    "  - Step 4: After 2-4 tool calls, STOP and synthesize.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. NEVER invent numbers. Every figure MUST appear in a tool return value.",
    "2. NEVER recalculate.",
    "3. Include EVERY issue from verify_accruals in the `issues` array.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` is 3-5 sentences AND analytical:",
    "     - Lead with the largest-amount issue.",
    "     - Mention any cadence pattern (e.g., 'all 5 stale accruals are",
    "       monthly, suggesting a recurring month-end miss').",
    "     - Recommend a next action.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    return Agent(
        name="Accrual Verification Agent",
        model=get_model(),
        tools=[
            verify_accruals,
            list_accruals_by_cadence,
            check_gl_account_in_tb,
        ],
        description="Audits accrual schedules and reasons about cadence patterns.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=AccrualVerificationResult,
        markdown=False,
        use_json_mode=True,
        
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
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


def run_accrual_verification(
    company_id: str, period: str | None = None
) -> AccrualVerificationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return AccrualVerificationResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                total_accruals=0, flagged_count=0, total_flagged_amount=0.0,
                issues=[], summary=f"No trial balance data for '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = verify_accruals(company_id, period)
    if precheck.get("found") and precheck.get("flagged_count", 0) == 0:
        logger.info("[Accrual] %s@%s clean — skipping LLM.", company_id, period)
        return AccrualVerificationResult(
            company_id=company_id, period=period, status="PASSED",
            total_accruals=precheck.get("total_accruals", 0),
            flagged_count=0, total_flagged_amount=0.0, issues=[],
            summary=(
                f"All {precheck.get('total_accruals', 0)} accruals are current. "
                f"No stale, zero, orphan, or duplicate entries. "
                f"LLM reasoning skipped."
            ),
        )

    prompt = (
        f"Verify accruals for company_id={company_id!r} and period={period!r}. "
        f"Use the tools to understand the cadence pattern of any issues. "
        f"Then produce the structured result."
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
        "LLM path failed for accruals %s@%s (%s) — falling back.",
        company_id, period, type(last_exc).__name__,
    )
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
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: {len(issues)} accrual issues flagged."
        ),
    )