"""
Trial Balance Validator Agent — Phase 1, Parallel Group (ReAct).

================================================================================
REACT PATTERN
================================================================================
This agent has THREE tools and uses the ReAct loop:

    1. analyze_trial_balance         — baseline (debits=credits, sign sanity)
    2. list_account_balances_by_type — drill into a category (Asset/Liability...)
    3. get_prior_period_summary      — compare current vs prior for trend

The LLM reasons about which tool to call, in what order, based on what it
sees. It then synthesises a 3-5 sentence narrative explaining the pattern —
not just "PASSED" or "FAILED".

================================================================================
DETERMINISTIC FALLBACK
================================================================================
If the LLM path fails (Gemini down, rate limit, malformed response), the
agent falls back to the deterministic tool output with a fixed summary.
Numbers are ALWAYS exact because Python owns all arithmetic.
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
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import TrialBalance

logger = logging.getLogger(__name__)

LEDGER_TOLERANCE = Decimal("1.00")
ROW_TOLERANCE = Decimal("0.01")
CONTRA_MARKERS = ("allowance", "accumulated", "reserve", "discount on", "provision for")

# Toggle ReAct trace logging via env var (AGENT_DEBUG=1). Off by default so
# production logs stay clean; on during development/testing.
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


def _is_contra_normal(account_name: str) -> bool:
    n = (account_name or "").lower()
    return any(m in n for m in CONTRA_MARKERS)


def _previous_period(period: str) -> str:
    """'2026-01' → '2025-12'. Handles year rollover."""
    y, m = (int(x) for x in period.split("-"))
    if m == 1:
        return f"{y - 1}-12"
    return f"{y}-{m - 1:02d}"


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class TrialBalanceIssue(BaseModel):
    account_code: str
    account_name: str
    issue_type: str = Field(
        ..., description="BALANCE_MISMATCH, UNUSUAL_SIGN, BOTH_SIDES_NONZERO, DUPLICATE_ACCOUNT."
    )
    description: str


class TrialBalanceValidatorResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if clean, else 'FAILED'.")
    total_debits: float
    total_credits: float
    difference: float
    account_count: int
    issues: list[TrialBalanceIssue] = Field(default_factory=list)
    summary: str = Field(
        ...,
        description=(
            "3-5 sentence analytical narrative. Not just a status. "
            "Cite tool values. Explain patterns you noticed across tools."
        ),
    )


# =============================================================================
# TOOL 1 — analyze_trial_balance  (primary analysis, unchanged)
# =============================================================================

def analyze_trial_balance(company_id: str, period: str) -> dict[str, Any]:
    """
    Primary trial balance analysis for one company+period.

    Validates:
        R1. SUM(debit) == SUM(credit) within $1
        R2. A single row must not carry both debit AND credit
        R3. row.balance == row.debit - row.credit
        R4. Sign sanity per account_type (contra-normal accounts exempt)

    Returns:
        total_debits, total_credits, difference, is_balanced,
        account_count, issues[]
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
            )
        ).all()

        if not rows:
            return {
                "company_id": company_id, "period": period, "found": False,
                "total_debits": 0.0, "total_credits": 0.0, "difference": 0.0,
                "account_count": 0, "is_balanced": False, "issues": [],
                "note": "No trial balance rows found.",
            }

        total_debits = sum((r.debit or Decimal("0")) for r in rows)
        total_credits = sum((r.credit or Decimal("0")) for r in rows)
        difference = total_debits - total_credits
        is_balanced = abs(difference) < LEDGER_TOLERANCE

        issues: list[dict[str, Any]] = []
        seen_accounts: set[str] = set()

        for r in rows:
            debit = r.debit or Decimal("0")
            credit = r.credit or Decimal("0")
            balance = r.balance or Decimal("0")
            acct_type = (r.account_type or "").strip().lower()
            is_contra = _is_contra_normal(r.account_name or "")

            if debit > 0 and credit > 0:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "issue_type": "BOTH_SIDES_NONZERO",
                    "description": f"Row has both debit ({debit}) and credit ({credit}).",
                })

            expected_balance = debit - credit
            if abs(expected_balance - balance) > ROW_TOLERANCE:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "issue_type": "BALANCE_MISMATCH",
                    "description": (
                        f"balance={balance} but debit-credit={expected_balance}; "
                        f"off by {balance - expected_balance}."
                    ),
                })

            if not is_contra:
                if acct_type in ("asset", "expense") and balance < 0:
                    issues.append({
                        "account_code": r.account_code, "account_name": r.account_name,
                        "issue_type": "UNUSUAL_SIGN",
                        "description": f"Debit-normal '{acct_type}' has credit balance {balance}.",
                    })
                if acct_type in ("liability", "equity", "revenue") and balance > 0:
                    issues.append({
                        "account_code": r.account_code, "account_name": r.account_name,
                        "issue_type": "UNUSUAL_SIGN",
                        "description": f"Credit-normal '{acct_type}' has debit balance {balance}.",
                    })

            if r.account_code in seen_accounts:
                issues.append({
                    "account_code": r.account_code, "account_name": r.account_name,
                    "issue_type": "DUPLICATE_ACCOUNT",
                    "description": "Account code appears more than once in this period.",
                })
            seen_accounts.add(r.account_code)

        return {
            "company_id": company_id, "period": period, "found": True,
            "total_debits": float(total_debits), "total_credits": float(total_credits),
            "difference": float(difference), "account_count": len(rows),
            "is_balanced": is_balanced, "issues": issues,
            "note": "Ledger is balanced." if is_balanced else f"Out of balance by {difference}.",
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_trial_balance failed for %s@%s", company_id, period)
        return {
            "company_id": company_id, "period": period, "found": False,
            "error": str(exc), "total_debits": 0.0, "total_credits": 0.0,
            "difference": 0.0, "account_count": 0, "is_balanced": False, "issues": [],
        }
    finally:
        db.close()


# =============================================================================
# TOOL 2 — list_account_balances_by_type  (drill-down)
# =============================================================================

def list_account_balances_by_type(
    company_id: str, period: str, account_type: str
) -> list[dict[str, Any]]:
    """
    List all accounts of a given type for one company+period with their balances.

    Use this to DRILL INTO a specific category after seeing issues in the
    baseline analysis. For example, if `analyze_trial_balance` flagged
    UNUSUAL_SIGN issues, call this with account_type='Asset' to see every
    asset account's balance at once.

    Args:
        account_type: One of 'Asset', 'Liability', 'Equity', 'Revenue',
                      'COGS', 'Expense', 'Operating Expense'. Case-insensitive.

    Returns:
        List of {account_code, account_name, balance, debit, credit} sorted
        by account_code.
    """
    db = SessionLocal()
    try:
        rows = db.execute(
            select(
                TrialBalance.account_code,
                TrialBalance.account_name,
                TrialBalance.balance,
                TrialBalance.debit,
                TrialBalance.credit,
            )
            .where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
                TrialBalance.account_type.ilike(account_type),
            )
            .order_by(TrialBalance.account_code)
        ).all()

        return [
            {
                "account_code": code,
                "account_name": name,
                "balance": float(balance or 0),
                "debit": float(debit or 0),
                "credit": float(credit or 0),
            }
            for code, name, balance, debit, credit in rows
        ]
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_prior_period_summary  (trend context)
# =============================================================================

def get_prior_period_summary(company_id: str, period: str) -> dict[str, Any]:
    """
    Return a compact summary of the trial balance for the period BEFORE the
    given period (i.e., `period` minus one month).

    Use this to distinguish "new issue this month" from "long-standing pattern."
    For example, if current debits are $5M, was prior period $4.9M (normal
    growth) or $2M (sudden double)? Both are balanced but tell very different
    stories.

    Returns:
        {found, prior_period, total_debits, total_credits, difference,
         account_count, delta_debits_vs_current}
        or {found: False, error: ...} if no prior data exists.
    """
    prior = _previous_period(period)
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == prior,
            )
        ).all()

        if not rows:
            return {
                "found": False,
                "prior_period": prior,
                "note": f"No trial balance rows for prior period {prior}.",
            }

        total_debits = sum((r.debit or Decimal("0")) for r in rows)
        total_credits = sum((r.credit or Decimal("0")) for r in rows)

        return {
            "found": True,
            "prior_period": prior,
            "total_debits": float(total_debits),
            "total_credits": float(total_credits),
            "difference": float(total_debits - total_credits),
            "account_count": len(rows),
        }
    finally:
        db.close()


# =============================================================================
# AGENT DEFINITION — ReAct-style instructions
# =============================================================================

AGENT_INSTRUCTIONS = [
    "You are a Trial Balance Validator for a Private Equity month-end close system.",
    "You have THREE tools. Reason about which to call and in what order.",
    "",
    "TOOLS:",
    "  1. analyze_trial_balance(company_id, period)",
    "       — Baseline analysis: debits=credits, sign sanity, duplicates.",
    "       ALWAYS call this first.",
    "",
    "  2. list_account_balances_by_type(company_id, period, account_type)",
    "       — List every account in a category (Asset/Liability/Revenue/etc.).",
    "       Call this if the baseline flagged issues in a specific category,",
    "       to see which accounts are involved and how they cluster.",
    "",
    "  3. get_prior_period_summary(company_id, period)",
    "       — Summary of the period BEFORE this one.",
    "       Call this when the current numbers look unusual — to distinguish",
    "       a new problem from a long-standing pattern.",
    "",
    "WORKFLOW (use your judgment, this is a guideline not a script):",
    "  - Step 1: Always call analyze_trial_balance first.",
    "  - Step 2: If issues exist, use list_account_balances_by_type to drill",
    "            into the affected categories.",
    "  - Step 3: If numbers look surprising, use get_prior_period_summary to",
    "            see whether this is new.",
    "  - Step 4: After 2-4 tool calls, STOP and synthesize your answer.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. NEVER invent numbers. Every figure in your output MUST appear in a",
    "   tool return value. If a tool didn't return it, don't say it.",
    "2. NEVER recalculate or re-sum. Cite tool values verbatim.",
    "3. status='PASSED' iff is_balanced=true AND issues is empty. Else 'FAILED'.",
    "4. Include EVERY issue from the tool output. Do not add, drop, or merge.",
    "5. `summary` must be 3-5 sentences AND analytical:",
    "     - Lead with the status (PASSED/FAILED) and the difference amount.",
    "     - Cite the biggest issue (by account) if any.",
    "     - Mention any pattern you noticed across the tools (e.g., 'all sign",
    "       issues are on Asset accounts, and the prior period was clean').",
    "     - Do NOT list every issue — that's what the `issues` field is for.",
    "",
    "Output ONLY the structured JSON schema. No prose outside the schema.",
]


def _build_agent() -> Agent:
    """Build a fresh Agent per call — no shared state across Celery workers."""
    return Agent(
        name="Trial Balance Validator",
        model=get_model(),
        tools=[
            analyze_trial_balance,
            list_account_balances_by_type,
            get_prior_period_summary,
        ],
        description=(
            "Validates a trial balance via multiple deterministic tools, "
            "reasons about which to call, and synthesises an analytical narrative."
        ),
        instructions=AGENT_INSTRUCTIONS,
        output_schema=TrialBalanceValidatorResult,
        markdown=False,
        use_json_mode=True,
           # ← ReAct trace in logs when AGENT_DEBUG=1
        debug_mode=_AGENT_DEBUG,        # ← extra Agno internal logging
        tool_call_limit=4,
    )


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================

def _safe_parse(content: Any) -> TrialBalanceValidatorResult:
    if isinstance(content, TrialBalanceValidatorResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return TrialBalanceValidatorResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return TrialBalanceValidatorResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def validate_trial_balance(
    company_id: str, period: str | None = None
) -> TrialBalanceValidatorResult:
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
            return TrialBalanceValidatorResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                total_debits=0.0, total_credits=0.0, difference=0.0,
                account_count=0, issues=[],
                summary=f"No trial balance data exists for company '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = analyze_trial_balance(company_id, period)
    if precheck.get("is_balanced") and not precheck.get("issues"):
        logger.info("[TB] %s@%s clean — skipping LLM (0 requests).", company_id, period)
        return TrialBalanceValidatorResult(
            company_id=company_id, period=period, status="PASSED",
            total_debits=precheck["total_debits"],
            total_credits=precheck["total_credits"],
            difference=precheck["difference"],
            account_count=precheck["account_count"],
            issues=[],
            summary=(
                f"Trial balance is balanced at ${precheck['total_debits']:,.2f} "
                f"across {precheck['account_count']} accounts "
                f"(difference ${precheck['difference']:.2f}, within tolerance). "
                f"No sign, duplicate, or mismatch issues detected. "
                f"LLM reasoning skipped — nothing to investigate."
            ),
        )

    prompt = (
        f"Validate the trial balance for company_id={company_id!r} and "
        f"period={period!r}. Use the tools available to build a complete "
        f"picture, then produce the structured result."
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
        "LLM path failed for %s@%s (%s) — falling back.",
        company_id, period, type(last_exc).__name__,
    )
    facts = analyze_trial_balance(company_id, period)
    issues = [
        TrialBalanceIssue(
            account_code=i["account_code"], account_name=i["account_name"],
            issue_type=i["issue_type"], description=i["description"],
        )
        for i in facts.get("issues", [])
    ]
    status = "PASSED" if facts.get("is_balanced") and not issues else "FAILED"
    return TrialBalanceValidatorResult(
        company_id=company_id, period=period, status=status,
        total_debits=facts.get("total_debits", 0.0),
        total_credits=facts.get("total_credits", 0.0),
        difference=facts.get("difference", 0.0),
        account_count=facts.get("account_count", 0),
        issues=issues,
        summary=(
            f"Deterministic result (LLM narrative skipped: {type(last_exc).__name__}). "
            f"Deterministic result: TB is "
            f"{'balanced' if facts.get('is_balanced') else 'out of balance'} "
            f"by {facts.get('difference', 0.0)} across "
            f"{facts.get('account_count', 0)} accounts. "
            f"{len(issues)} issues flagged."
        ),
    )