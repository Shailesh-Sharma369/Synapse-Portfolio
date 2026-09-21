"""
Expense Categorization Agent — Phase 2, Sequential Group (Step 3 of 3).

================================================================================
BUSINESS PROBLEM
================================================================================
Expense mis-categorization silently distorts EBITDA, departmental P&Ls, and
cost-center reporting. Common patterns:

    - MISCLASSIFIED: expense-like account code booked in a COGS bucket (or
                     vice-versa) based on the account NAME.
    - UNUSUAL_RATIO: an expense account is >2x its own mean over the periods
                     we have data for. Catches transposition errors.
    - ORPHAN: expense account with no budget line.
    - NEGATIVE_EXPENSE: expense account with negative balance (should be ≥ 0).
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
from sqlalchemy import func, select

from app.db.database import SessionLocal, settings
from app.db.models import Budget, TrialBalance

logger = logging.getLogger(__name__)

RATIO_THRESHOLD = Decimal("2.0")
EXPENSE_NAME_HINTS = ("expense", "cost of", "cogs", "salaries", "wages", "rent",
                      "utilities", "marketing", "advertising", "travel")


class ExpenseIssue(BaseModel):
    account_code: str
    account_name: str
    account_type: str
    amount: float
    issue_type: str = Field(..., description="UNUSUAL_RATIO, NEGATIVE_EXPENSE, ORPHAN, MISCLASSIFIED.")
    description: str


class ExpenseCategorizationResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if no issues, else 'FAILED'.")
    expense_accounts_examined: int
    flagged_count: int
    total_expenses: float
    issues: list[ExpenseIssue] = Field(default_factory=list)
    summary: str


def categorize_expenses(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic audit of expense accounts for one company+period.

    ALGORITHM
    ---------
    1. Load all expense-type TrialBalance rows for (company, period).
       ("Expense", "COGS", "Operating Expense").
    2. Compute each account's trailing mean across all other periods.
    3. Flag:
         - NEGATIVE_EXPENSE: balance < 0
         - UNUSUAL_RATIO:   |balance| > RATIO_THRESHOLD * |trailing_mean|
         - ORPHAN: no matching Budget row for this account+year+month
         - MISCLASSIFIED: account name contains expense keywords but
                          account_type is Asset/Revenue (or vice-versa)
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
            )
        ).all()

        expense_rows = [
            r for r in rows
            if (r.account_type or "").lower() in ("expense", "cogs", "operating expense")
        ]

        if not expense_rows:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No expense accounts found.", "expense_accounts_examined": 0, "issues": []}

        mean_stmt = (
            select(TrialBalance.account_code, func.avg(TrialBalance.balance))
            .where(TrialBalance.company_id == company_id, TrialBalance.period != period)
            .group_by(TrialBalance.account_code)
        )
        trailing_mean = {code: Decimal(str(avg or 0)) for code, avg in db.execute(mean_stmt).all()}

        try:
            year, month = (int(x) for x in period.split("-"))
        except Exception:
            year, month = 0, 0
        budget_codes = {
            code for (code,) in db.execute(
                select(Budget.account_code).where(
                    Budget.company_id == company_id,
                    Budget.year == year, Budget.month == month,
                )
            ).all()
        }

        issues: list[dict[str, Any]] = []
        total_expenses = Decimal("0")

        for r in expense_rows:
            balance = r.balance or Decimal("0")
            total_expenses += balance
            name_lower = (r.account_name or "").lower()

            if balance < 0:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "NEGATIVE_EXPENSE",
                    "description": f"Expense account has negative balance {balance}.",
                })

            tm = trailing_mean.get(r.account_code)
            if tm and abs(tm) > Decimal("1") and abs(balance) > RATIO_THRESHOLD * abs(tm):
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "UNUSUAL_RATIO",
                    "description": f"Balance {balance} is {abs(balance / tm):.1f}x trailing mean {tm:.2f}.",
                })

            if budget_codes and r.account_code not in budget_codes:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "ORPHAN",
                    "description": "No budget line for this account in this period.",
                })

            if any(h in name_lower for h in EXPENSE_NAME_HINTS) and (r.account_type or "").lower() not in (
                "expense", "cogs", "operating expense"
            ):
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "MISCLASSIFIED",
                    "description": f"Name suggests expense but account_type is '{r.account_type}'.",
                })

        issues.sort(key=lambda x: abs(x["amount"]), reverse=True)

        return {
            "company_id": company_id, "period": period, "found": True,
            "expense_accounts_examined": len(expense_rows),
            "flagged_count": len(issues),
            "total_expenses": float(total_expenses), "issues": issues,
            "note": f"{len(issues)} expense issues detected." if issues else "All expenses clean.",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("categorize_expenses failed for %s@%s", company_id, period)
        return {"company_id": company_id, "period": period, "found": False,
                "error": str(exc), "expense_accounts_examined": 0, "issues": []}
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
    "You are an Expense Categorization Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `categorize_expenses` FIRST. Never invent numbers.",
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
        name="Expense Categorization Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[categorize_expenses],
        description="Audits expense accounts for miscategorization, ratios, orphans.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=ExpenseCategorizationResult,
        markdown=False,
    )


def _safe_parse(content: Any) -> ExpenseCategorizationResult:
    if isinstance(content, ExpenseCategorizationResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return ExpenseCategorizationResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return ExpenseCategorizationResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_expense_categorization(company_id: str, period: str | None = None) -> ExpenseCategorizationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return ExpenseCategorizationResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                expense_accounts_examined=0, flagged_count=0,
                total_expenses=0.0, issues=[],
                summary=f"No trial balance data for '{company_id}'.",
            )

    prompt = (
        f"Audit expense categorization for company_id={company_id!r} and period={period!r}. "
        f"Call categorize_expenses with these exact arguments."
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

    logger.warning("LLM path failed for expenses %s@%s (%s) — falling back.", company_id, period, type(last_exc).__name__)
    facts = categorize_expenses(company_id, period)
    issues = [
        ExpenseIssue(
            account_code=i["account_code"], account_name=i["account_name"],
            account_type=i["account_type"], amount=i["amount"],
            issue_type=i["issue_type"], description=i["description"],
        )
        for i in facts.get("issues", [])
    ]
    return ExpenseCategorizationResult(
        company_id=company_id, period=period,
        status="PASSED" if not issues else "FAILED",
        expense_accounts_examined=facts.get("expense_accounts_examined", 0),
        flagged_count=len(issues),
        total_expenses=facts.get("total_expenses", 0.0),
        issues=issues,
        summary=(
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: {len(issues)} expense issues flagged."
        ),
    )