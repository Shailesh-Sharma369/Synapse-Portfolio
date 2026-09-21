# Variance Analysis Agent — Phase 1, Parallel Group.
from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Budget, TrialBalance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Materiality thresholds — defined as constants so they're easy to audit and
# easy to override per-company later (some portfolio companies may want $10K,
# others $500K, depending on size).
# ---------------------------------------------------------------------------
VARIANCE_ABS_THRESHOLD_USD = Decimal("50000")
VARIANCE_PCT_THRESHOLD = Decimal("10")  # percent

# =============================================================================
# 1. STRUCTURED OUTPUT SCHEMA
# =============================================================================

class VarianceItem(BaseModel):
    """A single account-level variance that breached at least one threshold."""

    account_code: str
    account_name: str
    account_type: str
    actual: float = Field(..., description="Normalized actual amount (positive = same sign as budget).")
    budget: float = Field(..., description="Budgeted amount for this account and period.")
    variance: float = Field(..., description="actual - budget.")
    variance_pct: float = Field(..., description="(variance / budget) * 100. Zero when budget is 0.")
    breach_reason: str = Field(
        ..., description="'DOLLAR', 'PERCENT', or 'BOTH' — which threshold was breached."
    )
    severity: str = Field(..., description="'HIGH' if both breached, else 'MEDIUM'.")


class VarianceAnalysisResult(BaseModel):
    """Final structured verdict returned by the agent for one company+period."""

    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if no flagged variances, else 'FAILED'.")
    accounts_compared: int
    flagged_count: int
    total_favorable_variance: float = Field(
        ..., description="Sum of variances where actual < budget (for expense/revenue logic)."
    )
    total_unfavorable_variance: float = Field(
        ..., description="Sum of variances where actual > budget."
    )
    variances: list[VarianceItem] = Field(default_factory=list)
    summary: str = Field(..., description="Controller-friendly narrative summarising the top movements.")

# =============================================================================
# 2. DETERMINISTIC ANALYSIS TOOL
# =============================================================================

def _parse_period(period: str) -> tuple[int, int]:
    """
    Split 'YYYY-MM' into (year, month) as ints.

    Raises ValueError on malformed input so the tool returns a clean error
    payload rather than silently comparing against the wrong period.
    """
    year_str, month_str = period.split("-")
    return int(year_str), int(month_str)

def _normalize_actual(balance: Decimal, account_type: str) -> Decimal:
    """
    Re-sign the actual balance so it matches the budget's sign convention.

    WHY THIS EXISTS
    ---------------
    Our TrialBalance schema stores balances in signed form:
        - Assets, Expenses, COGS : positive balance (debit-normal)
        - Revenue, Liab, Equity  : negative balance (credit-normal)

    Our Budget schema stores everything as positive amounts (the "expected"
    level of activity).
    """
    if account_type in ("Revenue", "Liability", "Equity"):
        return -balance
    return balance

def analyze_variances(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic actual-vs-budget comparison for one company and period.

    This is exposed to the Agno agent as a tool. It returns raw facts only;
    all judgement (which variances matter, how to describe them) is left to
    the LLM layer downstream.

    ALGORITHM
    ---------
    1. Parse `period` into (year, month).
    2. Load all Budget rows for (company_id, year, month).
    3. Load all TrialBalance rows for (company_id, period).
    4. For every account present in EITHER source:
         actual_norm = _normalize_actual(actual.balance, account_type)
         variance    = actual_norm - budget_amount
         variance_%  = variance / budget_amount * 100   (0 if budget == 0)
       A flag is raised when |variance| > $50K OR |variance_%| > 10%.
    5. Return the full comparison plus a sorted-by-absolute-impact list of
       flagged items.
    """
    db = SessionLocal()
    try:
        try:
            year, month = _parse_period(period)
        except Exception:
            return {
                "company_id": company_id, "period": period, "found": False,
                "error": f"Invalid period format: {period!r}. Expected 'YYYY-MM'.",
                "accounts_compared": 0, "flagged": [],
                "total_favorable": 0.0, "total_unfavorable": 0.0,
            }

        # ---- Load budget and actuals in two bulk queries (no N+1) ---------
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
                "error": "No budget or actuals found for this company and period.",
                "accounts_compared": 0, "flagged": [],
                "total_favorable": 0.0, "total_unfavorable": 0.0,
            }

        flagged: list[dict[str, Any]] = []
        compared = 0
        total_fav = Decimal("0")
        total_unfav = Decimal("0")
        missing_budget_accounts: list[str] = []

        for a in actuals:
            compared += 1
            actual_norm = _normalize_actual(a.balance or Decimal("0"), a.account_type or "")

            budget_row = budgets.get(a.account_code)
            if budget_row is None:
                # Finding, not error: the account has activity but no plan.
                missing_budget_accounts.append(a.account_code)
                continue

            budget_amt = budget_row.budget_amount or Decimal("0")
            variance = actual_norm - budget_amt

            # Compute % safely — division by zero is a real case here.
            if budget_amt != 0:
                variance_pct = (variance / budget_amt) * Decimal("100")
            else:
                variance_pct = Decimal("0")

            # Track running favorable/unfavorable totals for the summary.
            # Convention: actual < budget is "favorable" for a cost line and
            # "unfavorable" for a revenue line. We keep it sign-agnostic here
            # and let the LLM apply business meaning.
            if variance < 0:
                total_fav += variance
            else:
                total_unfav += variance

            # ---- Threshold check (OR, not AND) ---------------------------
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
                "account_code": a.account_code,
                "account_name": a.account_name,
                "account_type": a.account_type,
                "actual": float(actual_norm),
                "budget": float(budget_amt),
                "variance": float(variance),
                "variance_pct": float(variance_pct),
                "breach_reason": reason,
                "severity": severity,
            })

        # Sort by absolute impact so the biggest movers are first — the LLM
        # then naturally leads its narrative with the most material items.
        flagged.sort(key=lambda x: abs(x["variance"]), reverse=True)

        return {
            "company_id": company_id,
            "period": period,
            "found": True,
            "accounts_compared": compared,
            "flagged": flagged,
            "flagged_count": len(flagged),
            "total_favorable": float(total_fav),
            "total_unfavorable": float(total_unfav),
            "missing_budget_accounts": missing_budget_accounts,
            "thresholds": {
                "abs_usd": float(VARIANCE_ABS_THRESHOLD_USD),
                "pct": float(VARIANCE_PCT_THRESHOLD),
            },
            "note": (
                f"{len(flagged)} variances breached thresholds."
                if flagged else "No variances breached thresholds."
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_variances failed for %s@%s", company_id, period)
        return {
            "company_id": company_id, "period": period, "found": False,
            "error": str(exc),
            "accounts_compared": 0, "flagged": [],
            "total_favorable": 0.0, "total_unfavorable": 0.0,
        }
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    """Return most recent period present for the company's trial balance."""
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
# 3. AGNO AGENT DEFINITION
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are a Variance Analysis Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call the `analyze_variances` tool first. Never invent numbers.",
    "2. NEVER recalculate variances, percentages, or totals yourself. Cite the tool output verbatim.",
    "3. For every item in the tool's `flagged` array, include it in your `variances` output array.",
    "   Do not add, remove, or reorder items.",
    "4. Set status = 'PASSED' if and only if flagged_count == 0. Otherwise 'FAILED'.",
    "5. `summary` must be 2–3 sentences for a controller. Lead with the LARGEST absolute variance.",
    "   Mention the account name, direction (over/under plan), and dollar amount.",
    "   If `missing_budget_accounts` is non-empty, note that planning coverage is incomplete.",
    "6. If the tool returned found=false, set status='FAILED' and explain the missing-data reason.",
    "",
    "You output ONLY the structured JSON schema. No prose outside the schema.",
]

def _build_agent() -> Agent:
    """Build a fresh Agent per call — no shared state across Celery workers."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot initialise LLM.")

    return Agent(
        name="Variance Analysis Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[analyze_variances],
        description=(
            "Compares actuals against budget for a company+period, flags "
            "material variances, and produces a controller-ready summary."
        ),
        instructions=AGENT_INSTRUCTIONS,
        output_schema=VarianceAnalysisResult,
        markdown=False,
    )


# =============================================================================
# 4. PUBLIC ENTRYPOINT
# =============================================================================

def run_variance_analysis(
    company_id: str,
    period: str | None = None,
) -> VarianceAnalysisResult:
    """
    Run the Variance Analysis agent for one company.

    Args:
        company_id: Portfolio company slug.
        period:     'YYYY-MM'. If None, uses the latest period for this company.

    Returns:
        VarianceAnalysisResult. Falls back to a Python-only verdict if the LLM
        is unreachable, so the orchestrator never blocks on a rate-limit.
    """
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return VarianceAnalysisResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                accounts_compared=0, flagged_count=0,
                total_favorable_variance=0.0, total_unfavorable_variance=0.0,
                variances=[],
                summary=f"No trial balance data for company '{company_id}'.",
            )

    prompt = (
        f"Run variance analysis for company_id={company_id!r} and period={period!r}. "
        f"Call analyze_variances with these exact arguments and summarise the results."
    )

    try:
        agent = _build_agent()
        response = agent.run(prompt)
        content = response.content

        if isinstance(content, VarianceAnalysisResult):
            return content
        if isinstance(content, dict):
            return VarianceAnalysisResult(**content)
        if isinstance(content, str):
            return VarianceAnalysisResult(**json.loads(content))
        raise ValueError(f"Unexpected agent response type: {type(content)}")

    except Exception as exc:  # noqa: BLE001
        # Fallback: return Python-only verdict if Gemini is unavailable.
        logger.exception("LLM path failed for variance %s@%s — falling back.", company_id, period)
        facts = analyze_variances(company_id, period)

        items = [
            VarianceItem(
                account_code=v["account_code"],
                account_name=v["account_name"],
                account_type=v["account_type"],
                actual=v["actual"],
                budget=v["budget"],
                variance=v["variance"],
                variance_pct=v["variance_pct"],
                breach_reason=v["breach_reason"],
                severity=v["severity"],
            )
            for v in facts.get("flagged", [])
        ]

        status = "PASSED" if not items else "FAILED"
        summary = (
            f"LLM unavailable ({type(exc).__name__}). "
            f"Deterministic result: {len(items)} variances breached thresholds "
            f"across {facts.get('accounts_compared', 0)} accounts."
        )

        return VarianceAnalysisResult(
            company_id=company_id, period=period, status=status,
            accounts_compared=facts.get("accounts_compared", 0),
            flagged_count=len(items),
            total_favorable_variance=facts.get("total_favorable", 0.0),
            total_unfavorable_variance=facts.get("total_unfavorable", 0.0),
            variances=items,
            summary=summary,
        )
