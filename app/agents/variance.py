"""
Variance Analysis Agent — Phase 1, Parallel Group.
Comparing raw signed balances to positive budgets would generate 100% false positives on every revenue line."
Two thresholds must be exceeded OR-ed (not AND-ed) to flag an item:
      |variance|  >  $50,000           (materiality in dollars)
      |variance%| >  10%               (materiality in relative terms)
  Why OR? A $200K miss on a $50M line is a rounding error (0.4%) but it's still
  $200K of real cash the PE fund should know about. A 15% miss on a $200K line
  is only $30K but it means the plan was wrong by a wide margin. Both matter.
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
from app.db.models import Budget, TrialBalance

logger = logging.getLogger(__name__)

VARIANCE_ABS_THRESHOLD_USD = Decimal("50000")
VARIANCE_PCT_THRESHOLD = Decimal("10")


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
    summary: str


def _parse_period(period: str) -> tuple[int, int]:
    y, m = period.split("-")
    return int(y), int(m)


def _normalize_actual(balance: Decimal, account_type: str) -> Decimal:
    if account_type in ("Revenue", "Liability", "Equity"):
        return -balance
    return balance


def analyze_variances(company_id: str, period: str) -> dict[str, Any]:
    """Deterministic actual-vs-budget comparison. Never raises."""
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
        missing_budget_accounts: list[str] = []

        for a in actuals:
            compared += 1
            actual_norm = _normalize_actual(a.balance or Decimal("0"), a.account_type or "")

            budget_row = budgets.get(a.account_code)
            if budget_row is None:
                missing_budget_accounts.append(a.account_code)
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
            "missing_budget_accounts": missing_budget_accounts,
            "thresholds": {"abs_usd": float(VARIANCE_ABS_THRESHOLD_USD), "pct": float(VARIANCE_PCT_THRESHOLD)},
            "note": f"{len(flagged)} variances breached thresholds." if flagged else "No variances breached thresholds.",
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
    "You are a Variance Analysis Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `analyze_variances` first. Never invent numbers.",
    "2. NEVER recalculate. Cite tool output verbatim.",
    "3. Include every item from `flagged` — do not add or drop.",
    "4. status='PASSED' iff flagged_count == 0, else 'FAILED'.",
    "5. `summary` is 2-3 sentences. Lead with the LARGEST absolute variance.",
    "6. If found=false, set status='FAILED' and explain.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Variance Analysis Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[analyze_variances],
        description="Compares actuals vs budget, flags material variances.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=VarianceAnalysisResult,
        markdown=False,
    )


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


def run_variance_analysis(company_id: str, period: str | None = None) -> VarianceAnalysisResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return VarianceAnalysisResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                accounts_compared=0, flagged_count=0,
                total_favorable_variance=0.0, total_unfavorable_variance=0.0,
                variances=[], summary=f"No trial balance data for '{company_id}'.",
            )

    prompt = (
        f"Run variance analysis for company_id={company_id!r} and period={period!r}. "
        f"Call analyze_variances with these exact arguments."
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

    logger.warning("LLM path failed for variance %s@%s (%s) — falling back.", company_id, period, type(last_exc).__name__)
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
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: {len(items)} variances breached thresholds "
            f"across {facts.get('accounts_compared', 0)} accounts."
        ),
    )