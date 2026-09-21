"""
Trial Balance Validator Agent — Phase 1, Parallel Group.

================================================================================
HYBRID DESIGN (Python Math + LLM Reasoning)
================================================================================
- Python (SQLAlchemy + Decimal) owns ALL arithmetic and rule enforcement.
- Gemini owns ONLY reasoning, prioritization, and natural-language summary.
- The LLM NEVER invents or recalculates numbers.

Rules enforced in Python:
    R1. SUM(debit) == SUM(credit) at the ledger level (within $1 tolerance).
    R2. A single row must not carry both debit AND credit.
    R3. row.balance == row.debit - row.credit.
    R4. Sign sanity per account type — skipped for contra-normal accounts
        (Allowance, Accumulated, Reserve, Discount, Provision) which are
        Asset-type but credit-normal by accounting convention.
================================================================================
"""

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
from app.db.models import TrialBalance

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS — accounting tolerances and conventions
# =============================================================================

# $1.00 tolerance for ledger-level balance. Catches real imbalances ($10K+,
# $40K+) without false-positive on float rounding artifacts (e.g. $0.04).
LEDGER_TOLERANCE = Decimal("1.00")

# Row-level tolerance for balance == debit - credit. Individual rows are
# small, so $0.01 is a fine floor.
ROW_TOLERANCE = Decimal("0.01")

# Contra-normal account name prefixes. Accounts with these substrings in
# their name are contra-assets (Asset type, credit-normal) or contra-
# liabilities (Liability type, debit-normal). Sign rules do not apply.
CONTRA_MARKERS = (
    "allowance",
    "accumulated",
    "reserve",
    "discount on",
    "provision for",
)


def _is_contra_normal(account_name: str) -> bool:
    """True if the account name indicates a contra-normal account."""
    n = (account_name or "").lower()
    return any(marker in n for marker in CONTRA_MARKERS)


# =============================================================================
# 1. STRUCTURED OUTPUT SCHEMA
# =============================================================================

class TrialBalanceIssue(BaseModel):
    """A single rule violation detected by the deterministic Python tool."""

    account_code: str = Field(..., description="Chart-of-accounts code, e.g. '1100'.")
    account_name: str = Field(..., description="Human-readable account name.")
    issue_type: str = Field(
        ...,
        description="One of: BALANCE_MISMATCH, UNUSUAL_SIGN, BOTH_SIDES_NONZERO, DUPLICATE_ACCOUNT.",
    )
    description: str = Field(..., description="One-sentence explanation of the issue.")


class TrialBalanceValidatorResult(BaseModel):
    """Final structured verdict returned by the agent for one company+period."""

    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if clean, else 'FAILED'.")
    total_debits: float
    total_credits: float
    difference: float = Field(..., description="total_debits - total_credits. Zero means balanced.")
    account_count: int
    issues: list[TrialBalanceIssue] = Field(default_factory=list)
    summary: str = Field(..., description="Concise natural-language summary for the audit trail.")


# =============================================================================
# 2. DETERMINISTIC ANALYSIS TOOL (Python owns the math)
# =============================================================================

def analyze_trial_balance(company_id: str, period: str) -> dict[str, Any]:
    """
    Deterministic, side-effect-free analysis of one company's trial balance.

    Exposed to the Agno agent as a *tool*. Agno reads the docstring + type
    hints and auto-generates the Gemini function-calling schema.
    """
    db = SessionLocal()
    try:
        stmt = select(TrialBalance).where(
            TrialBalance.company_id == company_id,
            TrialBalance.period == period,
        )
        rows = db.scalars(stmt).all()

        # ---- Guard: no data -----------------------------------------------
        if not rows:
            logger.warning("No trial balance rows for %s @ %s", company_id, period)
            return {
                "company_id": company_id,
                "period": period,
                "found": False,
                "total_debits": 0.0,
                "total_credits": 0.0,
                "difference": 0.0,
                "account_count": 0,
                "is_balanced": False,
                "issues": [],
                "note": "No trial balance rows found for this company and period.",
            }

        # ---- R1: Ledger-level equality ------------------------------------
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

            # R2: both sides non-zero on same row
            if debit > 0 and credit > 0:
                issues.append({
                    "account_code": r.account_code,
                    "account_name": r.account_name,
                    "issue_type": "BOTH_SIDES_NONZERO",
                    "description": f"Row has both debit ({debit}) and credit ({credit}) populated.",
                })

            # R3: balance != debit - credit
            expected_balance = debit - credit
            if abs(expected_balance - balance) > ROW_TOLERANCE:
                issues.append({
                    "account_code": r.account_code,
                    "account_name": r.account_name,
                    "issue_type": "BALANCE_MISMATCH",
                    "description": (
                        f"balance={balance} but debit-credit={expected_balance}; "
                        f"off by {balance - expected_balance}."
                    ),
                })

            # R4: sign anomaly per account type — skip contra-normal accounts.
            # Contra-assets (Allowance, Accumulated Depreciation/Amortization)
            # legitimately carry credit balances despite being Asset type.
            if not is_contra:
                if acct_type in ("asset", "expense") and balance < 0:
                    issues.append({
                        "account_code": r.account_code,
                        "account_name": r.account_name,
                        "issue_type": "UNUSUAL_SIGN",
                        "description": f"Debit-normal account '{acct_type}' has credit balance {balance}.",
                    })
                if acct_type in ("liability", "equity", "revenue") and balance > 0:
                    issues.append({
                        "account_code": r.account_code,
                        "account_name": r.account_name,
                        "issue_type": "UNUSUAL_SIGN",
                        "description": f"Credit-normal account '{acct_type}' has debit balance {balance}.",
                    })

            # Duplicate account detection
            if r.account_code in seen_accounts:
                issues.append({
                    "account_code": r.account_code,
                    "account_name": r.account_name,
                    "issue_type": "DUPLICATE_ACCOUNT",
                    "description": "Account code appears more than once in this period.",
                })
            seen_accounts.add(r.account_code)

        return {
            "company_id": company_id,
            "period": period,
            "found": True,
            "total_debits": float(total_debits),
            "total_credits": float(total_credits),
            "difference": float(difference),
            "account_count": len(rows),
            "is_balanced": is_balanced,
            "issues": issues,
            "note": (
                "Ledger is balanced." if is_balanced
                else f"Ledger is OUT OF BALANCE by {difference}."
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_trial_balance failed for %s@%s", company_id, period)
        return {
            "company_id": company_id,
            "period": period,
            "found": False,
            "error": str(exc),
            "total_debits": 0.0,
            "total_credits": 0.0,
            "difference": 0.0,
            "account_count": 0,
            "is_balanced": False,
            "issues": [],
        }
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    """Return the most recent period present for this company, or None."""
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
    "You are a Trial Balance Validator for a Private Equity month-end close system.",
    "",
    "STRICT RULES — violating these is a critical failure:",
    "1. ALWAYS call the `analyze_trial_balance` tool FIRST. Never guess numbers.",
    "2. NEVER compute, re-sum, or modify any numeric value yourself.",
    "   The tool is the single source of truth for every number you output.",
    "3. Set status = 'PASSED' if and only if the tool returned is_balanced=true "
    "AND issues is empty. Otherwise set status = 'FAILED'.",
    "4. For each issue returned by the tool, include it in the `issues` array "
    "with a one-sentence description. Do not invent issues that were not returned.",
    "5. Write `summary` as 1-2 short sentences in plain English suitable for a "
    "controller. Mention the difference amount when the TB is out of balance.",
    "6. If the tool returns found=false, set status='FAILED' and explain in the "
    "summary that no trial balance was found for the requested period.",
    "",
    "You output ONLY the structured JSON schema. No prose, no markdown.",
]


def _build_agent() -> Agent:
    """Build a fresh Agent per call — no shared state across Celery workers."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot initialise LLM.")

    return Agent(
        name="Trial Balance Validator",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[analyze_trial_balance],
        description=(
            "Validates a company's trial balance for a given period by calling "
            "a deterministic analysis tool and summarising the results."
        ),
        instructions=AGENT_INSTRUCTIONS,
        output_schema=TrialBalanceValidatorResult,   # Agno 3.x: renamed from response_model
        markdown=False,
        # show_tool_calls removed (not in Agno 3.x signature)
    )


# =============================================================================
# 4. PUBLIC ENTRYPOINT
# =============================================================================

def validate_trial_balance(
    company_id: str,
    period: str | None = None,
) -> TrialBalanceValidatorResult:
    """Run the Trial Balance Validator for a single company."""
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return TrialBalanceValidatorResult(
                company_id=company_id,
                period="UNKNOWN",
                status="FAILED",
                total_debits=0.0,
                total_credits=0.0,
                difference=0.0,
                account_count=0,
                issues=[],
                summary=f"No trial balance data exists for company '{company_id}'.",
            )

    prompt = (
        f"Validate the trial balance for company_id={company_id!r} and period={period!r}. "
        f"Call the analyze_trial_balance tool with exactly these arguments."
    )

    try:
        agent = _build_agent()
        response = agent.run(prompt)
        content = response.content

        if isinstance(content, TrialBalanceValidatorResult):
            return content
        if isinstance(content, dict):
            return TrialBalanceValidatorResult(**content)
        if isinstance(content, str):
            return TrialBalanceValidatorResult(**json.loads(content))

        raise ValueError(f"Unexpected agent response type: {type(content)}")

    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM path failed for %s@%s — falling back.", company_id, period)
        facts = analyze_trial_balance(company_id, period)

        issues = [
            TrialBalanceIssue(
                account_code=i["account_code"],
                account_name=i["account_name"],
                issue_type=i["issue_type"],
                description=i["description"],
            )
            for i in facts.get("issues", [])
        ]

        status = "PASSED" if facts.get("is_balanced") and not issues else "FAILED"
        summary = (
            f"LLM unavailable ({type(exc).__name__}). "
            f"Deterministic result: TB is {'balanced' if facts.get('is_balanced') else 'out of balance'} "
            f"by {facts.get('difference', 0.0)} across {facts.get('account_count', 0)} accounts."
        )

        return TrialBalanceValidatorResult(
            company_id=company_id,
            period=period,
            status=status,
            total_debits=facts.get("total_debits", 0.0),
            total_credits=facts.get("total_credits", 0.0),
            difference=facts.get("difference", 0.0),
            account_count=facts.get("account_count", 0),
            issues=issues,
            summary=summary,
        )