"""
Expense Categorization Agent — Phase 2, ReAct.

ReAct tools:
    1. categorize_expenses      — full audit
    2. get_master_coa           — CoA list for a company
    3. get_expense_trend        — trailing balances for one account

LLM reasons about misclassification patterns and CoA fits.
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
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.db.database import SessionLocal, settings
from app.db.models import Budget, TrialBalance

logger = logging.getLogger(__name__)

RATIO_THRESHOLD = Decimal("2.0")
EXPENSE_NAME_HINTS = (
    "expense", "cost of", "cogs", "salaries", "wages", "rent",
    "utilities", "marketing", "advertising", "travel",
)
EXPENSE_TYPES = ("expense", "cogs", "operating expense")
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


class ExpenseIssue(BaseModel):
    account_code: str
    account_name: str
    account_type: str
    amount: float
    issue_type: str = Field(..., description="UNUSUAL_RATIO, NEGATIVE_EXPENSE, ORPHAN, MISCLASSIFIED.")
    description: str
    suggested_reclass: dict[str, Any] | None = None


class ExpenseCategorizationResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' or 'FAILED'.")
    expense_accounts_examined: int
    flagged_count: int
    total_expenses: float
    issues: list[ExpenseIssue] = Field(default_factory=list)
    summary: str = Field(..., description="3-5 sentence analytical narrative.")


# =============================================================================
# TOOL 1 — categorize_expenses
# =============================================================================

def categorize_expenses(company_id: str, period: str) -> dict[str, Any]:
    """Deterministic audit of expense accounts for one company+period."""
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
            )
        ).all()

        if not rows:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No TB rows.", "expense_accounts_examined": 0, "issues": []}

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
        expense_rows_examined = 0

        for r in rows:
            balance = r.balance or Decimal("0")
            acct_type = (r.account_type or "").strip().lower()
            name_lower = (r.account_name or "").lower()
            is_expense_type = acct_type in EXPENSE_TYPES
            has_expense_name = any(h in name_lower for h in EXPENSE_NAME_HINTS)

            if has_expense_name and not is_expense_type:
                rec = suggest_reclassification(company_id, r.account_code, "Operating Expense")
                suggested = rec.get("selected") if rec.get("valid") else None
                desc = (
                    f"Name '{r.account_name}' suggests expense but account_type is "
                    f"'{r.account_type}'. "
                    + (f"Suggested reclass → {suggested['account_code']} ({suggested['account_name']})."
                       if suggested else rec["reason"])
                )
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "MISCLASSIFIED", "description": desc,
                    "suggested_reclass": suggested,
                })

            if not is_expense_type:
                continue

            expense_rows_examined += 1
            total_expenses += balance

            if balance < 0:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "NEGATIVE_EXPENSE",
                    "description": f"Expense account has negative balance {balance}.",
                    "suggested_reclass": None,
                })

            tm = trailing_mean.get(r.account_code)
            if tm and abs(tm) > Decimal("1") and abs(balance) > RATIO_THRESHOLD * abs(tm):
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "UNUSUAL_RATIO",
                    "description": f"Balance {balance} is {abs(balance / tm):.1f}x trailing mean {tm:.2f}.",
                    "suggested_reclass": None,
                })

            if budget_codes and r.account_code not in budget_codes:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "account_type": r.account_type, "amount": float(balance),
                    "issue_type": "ORPHAN",
                    "description": "No budget line for this account in this period.",
                    "suggested_reclass": None,
                })

        issues.sort(key=lambda x: abs(x["amount"]), reverse=True)

        return {
            "company_id": company_id, "period": period, "found": True,
            "expense_accounts_examined": expense_rows_examined,
            "flagged_count": len(issues),
            "total_expenses": float(total_expenses),
            "issues": issues,
            "note": f"{len(issues)} expense issues detected." if issues else "All expenses clean.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("categorize_expenses failed for %s@%s", company_id, period)
        return {"company_id": company_id, "period": period, "found": False,
                "error": str(exc), "expense_accounts_examined": 0, "issues": []}
    finally:
        db.close()


def suggest_reclassification(company_id: str, account_code: str, target_type: str) -> dict[str, Any]:
    """Return valid account codes ONLY from this company's CoA. Never invents."""
    db = SessionLocal()
    try:
        rows = db.execute(
            select(TrialBalance.account_code, TrialBalance.account_name, TrialBalance.account_type)
            .where(TrialBalance.company_id == company_id, TrialBalance.account_type == target_type)
            .distinct()
        ).all()
        candidates = [{"account_code": c, "account_name": n, "account_type": t} for c, n, t in rows]
        candidates.sort(key=lambda x: x["account_code"])
        if not candidates:
            return {"valid": False,
                    "reason": f"No '{target_type}' account exists in Master CoA for '{company_id}'. Human review required.",
                    "candidates": []}
        return {"valid": True, "selected": candidates[0], "candidates": candidates}
    finally:
        db.close()


# =============================================================================
# TOOL 2 — get_master_coa
# =============================================================================

def get_master_coa(company_id: str) -> list[dict[str, Any]]:
    """
    List every account in this company's Master CoA (across all periods).

    Use this to see what valid categories exist for reclassification — the
    LLM must pick from here, never invent.
    """
    db = SessionLocal()
    try:
        rows = db.execute(
            select(TrialBalance.account_code, TrialBalance.account_name, TrialBalance.account_type)
            .where(TrialBalance.company_id == company_id)
            .distinct()
            .order_by(TrialBalance.account_code)
        ).all()
        return [{"account_code": c, "account_name": n, "account_type": t} for c, n, t in rows]
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_expense_trend
# =============================================================================

def get_expense_trend(company_id: str, account_code: str, periods_back: int = 6) -> list[dict[str, Any]]:
    """Trailing balances for one expense account (oldest → newest)."""
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance)
            .where(TrialBalance.company_id == company_id, TrialBalance.account_code == account_code)
            .order_by(TrialBalance.period.desc())
            .limit(periods_back)
        ).all()
        out = [{"period": r.period, "balance": float(r.balance or 0), "account_name": r.account_name} for r in rows]
        out.reverse()
        return out
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    db = SessionLocal()
    try:
        return db.scalar(
            select(TrialBalance.period).where(TrialBalance.company_id == company_id)
            .order_by(TrialBalance.period.desc()).limit(1)
        )
    finally:
        db.close()


AGENT_INSTRUCTIONS = [
    "You are an Expense Categorization Agent for a PE month-end close system.",
    "You have THREE tools. Reason about which to call.",
    "",
    "TOOLS:",
    "  1. categorize_expenses(company_id, period) — audit. ALWAYS first.",
    "  2. get_master_coa(company_id) — list valid accounts. Use to verify",
    "     that any suggested reclass code actually exists.",
    "  3. get_expense_trend(company_id, account_code, periods_back) — trailing",
    "     history for one account. Use on UNUSUAL_RATIO to see if it's a",
    "     trend or one-off spike.",
    "",
    "WORKFLOW:",
    "  - Step 1: categorize_expenses.",
    "  - Step 2: If MISCLASSIFIED found, verify suggested codes via get_master_coa.",
    "  - Step 3: If UNUSUAL_RATIO on biggest account, drill with get_expense_trend.",
    "  - Step 4: STOP after 3-5 tool calls and synthesize.",
    "",
    "STRICT RULES:",
    "1. NEVER invent account codes — only use codes returned by tools.",
    "2. Include EVERY issue from categorize_expenses.",
    "3. status='PASSED' iff flagged_count == 0.",
    "4. `summary` = 3-5 analytical sentences leading with largest issue.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    """Build the Expense Categorization Agent."""
    return Agent(
        name="Expense Categorization Agent",
        model=get_model(),
        tools=[categorize_expenses, get_master_coa, get_expense_trend],
        description="Audits expense accounts and reasons about misclassification patterns.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=ExpenseCategorizationResult,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
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
        raise ValueError(f"Non-JSON: {stripped[:200]}")
    raise ValueError(f"Unexpected type: {type(content)}")


def run_expense_categorization(company_id: str, period: str | None = None) -> ExpenseCategorizationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return ExpenseCategorizationResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                expense_accounts_examined=0, flagged_count=0, total_expenses=0.0, issues=[],
                summary=f"No trial balance data for '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = categorize_expenses(company_id, period)
    if precheck.get("found") and precheck.get("flagged_count", 0) == 0:
        logger.info("[Expense] %s@%s clean — skipping LLM.", company_id, period)
        return ExpenseCategorizationResult(
            company_id=company_id, period=period, status="PASSED",
            expense_accounts_examined=precheck.get("expense_accounts_examined", 0),
            flagged_count=0,
            total_expenses=precheck.get("total_expenses", 0.0),
            issues=[],
            summary=(
                f"All {precheck.get('expense_accounts_examined', 0)} expense "
                f"accounts pass categorization checks. No misclassified, "
                f"unusual-ratio, negative, or orphan entries. LLM reasoning skipped."
            ),
        )

    prompt = (f"Audit expense categorization for company_id={company_id!r} and period={period!r}. "
              f"Use tools to reason about patterns, then produce the structured result.")

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

    logger.warning("LLM path failed for expenses %s@%s (%s) — falling back.",
                   company_id, period, type(last_exc).__name__)
    facts = categorize_expenses(company_id, period)
    issues = [ExpenseIssue(
        account_code=i["account_code"], account_name=i["account_name"],
        account_type=i["account_type"], amount=i["amount"],
        issue_type=i["issue_type"], description=i["description"],
        suggested_reclass=i.get("suggested_reclass"),
    ) for i in facts.get("issues", [])]
    return ExpenseCategorizationResult(
        company_id=company_id, period=period,
        status="PASSED" if not issues else "FAILED",
        expense_accounts_examined=facts.get("expense_accounts_examined", 0),
        flagged_count=len(issues),
        total_expenses=facts.get("total_expenses", 0.0),
        issues=issues,
        summary=f"LLM unavailable ({type(last_exc).__name__}). Deterministic: {len(issues)} issues.",
    )