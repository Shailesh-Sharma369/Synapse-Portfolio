"""
Variance Analysis Agent — Phase 1, Parallel Group (ReAct).

================================================================================
REACT PATTERN
================================================================================
Three tools, and the LLM decides which to call based on what it finds:

    1. analyze_variances        — full list of flagged variances
    2. get_account_history      — trailing N periods for one account
    3. get_budget_for_account   — budget for one account, one month

The LLM's job is to:
    - Find the largest variance (from tool 1)
    - Check whether it's a trend or a spike (tool 2)
    - Confirm the budget was flat or shifted (tool 3)
    - Write a narrative explaining the pattern

================================================================================
TWO-THRESHOLD RULE (unchanged)
================================================================================
An item is flagged if EITHER:
    |variance|   >  $50,000       (materiality in dollars)
    |variance%|  >  10%           (materiality in relative terms)
OR-ed because both real-dollar misses and plan-shape misses matter.
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Budget, TrialBalance

logger = logging.getLogger(__name__)

VARIANCE_ABS_THRESHOLD_USD = Decimal("50000")
VARIANCE_PCT_THRESHOLD = Decimal("10")

_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class VarianceItem(BaseModel):
    account_code: str
    account_name: str
    account_type: str
    actual: float
    budget: float
    variance: float
    variance_pct: float
    breach_reason: str = Field(..., description="'DOLLAR', 'PERCENT', or 'BOTH'.")
    severity: str = Field(..., description="'HIGH' if both breached, else 'MEDIUM'.")


class VarianceAnalysisResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if no flagged variances, else 'FAILED'.")
    accounts_compared: int
    flagged_count: int
    total_favorable_variance: float
    total_unfavorable_variance: float
    variances: list[VarianceItem] = Field(default_factory=list)
    summary: str = Field(
        ...,
        description=(
            "3-5 sentence analytical narrative. Lead with the largest variance, "
            "explain whether it's a trend or a spike, mention budget context."
        ),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _parse_period(period: str) -> tuple[int, int]:
    y, m = period.split("-")
    return int(y), int(m)


def _normalize_actual(balance: Decimal, account_type: str) -> Decimal:
    """Credit-normal accounts are stored negative; flip to positive for comparison."""
    if account_type in ("Revenue", "Liability", "Equity"):
        return -balance
    return balance


def _previous_periods(period: str, n: int) -> list[str]:
    """Return the last N periods ending just before `period` (oldest first)."""
    y, m = _parse_period(period)
    out: list[str] = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        out.append(f"{y}-{m:02d}")
    return list(reversed(out))


# =============================================================================
# TOOL 1 — analyze_variances  (primary analysis, unchanged)
# =============================================================================

def analyze_variances(company_id: str, period: str) -> dict[str, Any]:
    """
    Compare actuals to budget for one company+period, flag material variances.

    Flags an account if EITHER:
        |variance| > $50,000   OR   |variance%| > 10%

    Returns the flagged list sorted by absolute variance, descending.
    """
    db = SessionLocal()
    try:
        try:
            year, month = _parse_period(period)
        except Exception:
            return {
                "company_id": company_id, "period": period, "found": False,
                "error": f"Invalid period: {period!r}.",
                "accounts_compared": 0, "flagged": [],
                "total_favorable": 0.0, "total_unfavorable": 0.0,
            }

        budgets = {
            b.account_code: b
            for b in db.scalars(
                select(Budget).where(
                    Budget.company_id == company_id,
                    Budget.year == year,
                    Budget.month == month,
                )
            ).all()
        }
        actuals = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
            )
        ).all()

        if not budgets and not actuals:
            return {
                "company_id": company_id, "period": period, "found": False,
                "error": "No budget or actuals found.",
                "accounts_compared": 0, "flagged": [],
                "total_favorable": 0.0, "total_unfavorable": 0.0,
            }

        flagged: list[dict[str, Any]] = []
        compared = 0
        total_fav = Decimal("0")
        total_unfav = Decimal("0")

        for a in actuals:
            compared += 1
            actual_norm = _normalize_actual(a.balance or Decimal("0"), a.account_type or "")

            budget_row = budgets.get(a.account_code)
            if budget_row is None:
                continue

            budget_amt = budget_row.budget_amount or Decimal("0")
            variance = actual_norm - budget_amt
            variance_pct = (variance / budget_amt) * Decimal("100") if budget_amt != 0 else Decimal("0")

            if variance < 0:
                total_fav += variance
            else:
                total_unfav += variance

            dollar_breach = abs(variance) > VARIANCE_ABS_THRESHOLD_USD
            pct_breach = abs(variance_pct) > VARIANCE_PCT_THRESHOLD
            if not (dollar_breach or pct_breach):
                continue

            if dollar_breach and pct_breach:
                reason, severity = "BOTH", "HIGH"
            elif dollar_breach:
                reason, severity = "DOLLAR", "MEDIUM"
            else:
                reason, severity = "PERCENT", "MEDIUM"

            flagged.append({
                "account_code": a.account_code, "account_name": a.account_name,
                "account_type": a.account_type,
                "actual": float(actual_norm), "budget": float(budget_amt),
                "variance": float(variance), "variance_pct": float(variance_pct),
                "breach_reason": reason, "severity": severity,
            })

        flagged.sort(key=lambda x: abs(x["variance"]), reverse=True)

        return {
            "company_id": company_id, "period": period, "found": True,
            "accounts_compared": compared, "flagged": flagged,
            "flagged_count": len(flagged),
            "total_favorable": float(total_fav), "total_unfavorable": float(total_unfav),
            "thresholds": {
                "abs_usd": float(VARIANCE_ABS_THRESHOLD_USD),
                "pct": float(VARIANCE_PCT_THRESHOLD),
            },
            "note": f"{len(flagged)} variances breached thresholds.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_variances failed for %s@%s", company_id, period)
        return {
            "company_id": company_id, "period": period, "found": False,
            "error": str(exc), "accounts_compared": 0, "flagged": [],
            "total_favorable": 0.0, "total_unfavorable": 0.0,
        }
    finally:
        db.close()


# =============================================================================
# TOOL 2 — get_account_history  (trend context for one account)
# =============================================================================

def get_account_history(
    company_id: str, account_code: str, periods_back: int = 6
) -> list[dict[str, Any]]:
    """
    Return the last N periods of ACTUAL balances for a single account.

    Use this to distinguish a trend from a spike:
        - Gradual climb = headcount ramp, pricing drift, growth
        - Sudden jump   = one-time charge, misposting, data issue
        - Erratic       = data quality problem

    Args:
        company_id:   Company slug.
        account_code: 4-digit GL account code, e.g. "8000".
        periods_back: How many historical periods to return. Default 6.

    Returns:
        List of {period, balance, account_name} sorted oldest → newest.
        Returns [] if the account has no history.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance)
            .where(
                TrialBalance.company_id == company_id,
                TrialBalance.account_code == account_code,
            )
            .order_by(TrialBalance.period.desc())
            .limit(periods_back)
        ).all()

        # We fetched newest-first; reverse to oldest-first for trend reading
        out = [
            {
                "period": r.period,
                "account_name": r.account_name,
                "account_type": r.account_type,
                "balance": float(r.balance or 0),
            }
            for r in rows
        ]
        out.reverse()
        return out
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_budget_for_account  (single account budget lookup)
# =============================================================================

def get_budget_for_account(
    company_id: str, account_code: str, year: int, month: int
) -> dict[str, Any]:
    """
    Return the budgeted amount for one account in one month.

    Use this after identifying a large variance to CONFIRM whether the
    budget was stable (making the variance real) or itself shifted
    (making the variance a plan artifact).

    Returns:
        {found, account_code, account_name, budget_amount, year, month}
        or {found: False, error: ...} if no budget line exists.
    """
    db = SessionLocal()
    try:
        row = db.scalar(
            select(Budget).where(
                Budget.company_id == company_id,
                Budget.account_code == account_code,
                Budget.year == year,
                Budget.month == month,
            )
        )
        if row is None:
            return {
                "found": False,
                "account_code": account_code,
                "year": year, "month": month,
                "error": "No budget line for this account+period.",
            }
        return {
            "found": True,
            "account_code": row.account_code,
            "account_name": row.account_name,
            "budget_amount": float(row.budget_amount or 0),
            "year": year, "month": month,
        }
    finally:
        db.close()


# =============================================================================
# AGENT DEFINITION — ReAct-style instructions
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are a Variance Analysis Agent for a Private Equity month-end close system.",
    "You have THREE tools. Reason about which to call and in what order.",
    "",
    "TOOLS:",
    "  1. analyze_variances(company_id, period)",
    "       — Full list of flagged variances (breached $50K OR 10%).",
    "       ALWAYS call this first.",
    "",
    "  2. get_account_history(company_id, account_code, periods_back=6)",
    "       — Trailing actuals for ONE account. Use this on the LARGEST",
    "         variance to see whether it's a trend or a one-month spike.",
    "",
    "  3. get_budget_for_account(company_id, account_code, year, month)",
    "       — Budget for ONE account. Use this to confirm whether the",
    "         budget was stable (making the variance real) or shifted.",
    "",
    "WORKFLOW (guideline, not a script):",
    "  - Step 1: Call analyze_variances.",
    "  - Step 2: Identify the TOP 1-2 largest variances.",
    "  - Step 3: For the largest one, call get_account_history to see",
    "            whether the spike is a trend or a one-off.",
    "  - Step 4: If helpful, call get_budget_for_account to confirm the",
    "            budget baseline.",
    "  - Step 5: After 3-5 tool calls, STOP and synthesize your answer.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. NEVER invent numbers. Every figure in your output MUST appear in a",
    "   tool return value.",
    "2. NEVER recalculate. Cite tool values verbatim.",
    "3. Include EVERY flagged variance from analyze_variances in the",
    "   `variances` array. Do not add, drop, or merge entries.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` must be 3-5 sentences AND analytical:",
    "     - Lead with the LARGEST absolute variance (account + $ amount).",
    "     - Say whether it's a trend or a spike (from get_account_history).",
    "     - Mention the budget context (from get_budget_for_account) if you",
    "       called it.",
    "     - Suggest a next action (e.g., 'investigate headcount ramp').",
    "",
    "Output ONLY the structured JSON schema. No prose outside the schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Variance Analysis Agent",
        model=Gemini(id="gemini-3.1-flash-lite", api_key=settings.gemini_api_key),
        tools=[
            analyze_variances,
            get_account_history,
            get_budget_for_account,
        ],
        description=(
            "Compares actuals vs budget, flags material variances, and drills "
            "into the largest ones to distinguish trends from spikes."
        ),
        instructions=AGENT_INSTRUCTIONS,
        output_schema=VarianceAnalysisResult,
        markdown=False,
        use_json_mode=True,
        
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
    )


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================

def _safe_parse(content: Any) -> VarianceAnalysisResult:
    if isinstance(content, VarianceAnalysisResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return VarianceAnalysisResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return VarianceAnalysisResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_variance_analysis(
    company_id: str, period: str | None = None
) -> VarianceAnalysisResult:
    if period is None:
        db = SessionLocal()
        try:
            period = db.scalar(
                select(TrialBalance.period)
                .where(TrialBalance.company_id == company_id)
                .order_by(TrialBalance.period.desc())
                .limit(1)
            )
        finally:
            db.close()
        if period is None:
            return VarianceAnalysisResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                accounts_compared=0, flagged_count=0,
                total_favorable_variance=0.0, total_unfavorable_variance=0.0,
                variances=[], summary=f"No trial balance data for '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = analyze_variances(company_id, period)
    if precheck.get("found") and precheck.get("flagged_count", 0) == 0:
        logger.info("[Variance] %s@%s clean — skipping LLM.", company_id, period)
        return VarianceAnalysisResult(
            company_id=company_id, period=period, status="PASSED",
            accounts_compared=precheck.get("accounts_compared", 0),
            flagged_count=0,
            total_favorable_variance=precheck.get("total_favorable", 0.0),
            total_unfavorable_variance=precheck.get("total_unfavorable", 0.0),
            variances=[],
            summary=(
                f"No material variances detected across "
                f"{precheck.get('accounts_compared', 0)} accounts. "
                f"All actuals are within $50K or 10% of budget. "
                f"LLM reasoning skipped."
            ),
        )

    prompt = (
        f"Analyze variances for company_id={company_id!r} and period={period!r}. "
        f"Use the tools to find the largest variance and understand whether it's "
        f"a trend or a one-time event. Then produce the structured result."
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

    # ---- Deterministic fallback ------------------------------------------
    logger.warning(
        "LLM path failed for variance %s@%s (%s) — falling back.",
        company_id, period, type(last_exc).__name__,
    )
    facts = analyze_variances(company_id, period)
    items = [
        VarianceItem(
            account_code=v["account_code"], account_name=v["account_name"],
            account_type=v["account_type"], actual=v["actual"], budget=v["budget"],
            variance=v["variance"], variance_pct=v["variance_pct"],
            breach_reason=v["breach_reason"], severity=v["severity"],
        )
        for v in facts.get("flagged", [])
    ]
    return VarianceAnalysisResult(
        company_id=company_id, period=period,
        status="PASSED" if not items else "FAILED",
        accounts_compared=facts.get("accounts_compared", 0),
        flagged_count=len(items),
        total_favorable_variance=facts.get("total_favorable", 0.0),
        total_unfavorable_variance=facts.get("total_unfavorable", 0.0),
        variances=items,
        summary=(
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: {len(items)} variances breached thresholds "
            f"across {facts.get('accounts_compared', 0)} accounts."
        ),
    )