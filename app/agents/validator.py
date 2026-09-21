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
import time
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import TrialBalance

logger = logging.getLogger(__name__)

LEDGER_TOLERANCE = Decimal("1.00")
ROW_TOLERANCE = Decimal("0.01")
CONTRA_MARKERS = ("allowance", "accumulated", "reserve", "discount on", "provision for")


def _is_contra_normal(account_name: str) -> bool:
    n = (account_name or "").lower()
    return any(m in n for m in CONTRA_MARKERS)


class TrialBalanceIssue(BaseModel):
    account_code: str = Field(..., description="Chart-of-accounts code, e.g. '1100'.")
    account_name: str = Field(..., description="Human-readable account name.")
    issue_type: str = Field(..., description="BALANCE_MISMATCH, UNUSUAL_SIGN, BOTH_SIDES_NONZERO, DUPLICATE_ACCOUNT.")
    description: str = Field(..., description="One-sentence explanation of the issue.")


class TrialBalanceValidatorResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'PASSED' if clean, else 'FAILED'.")
    total_debits: float
    total_credits: float
    difference: float
    account_count: int
    issues: list[TrialBalanceIssue] = Field(default_factory=list)
    summary: str


def analyze_trial_balance(company_id: str, period: str) -> dict[str, Any]:
    """Deterministic analysis of one company's trial balance. Never raises."""
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
                "note": "No trial balance rows found for this company and period.",
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
                    "description": f"Row has both debit ({debit}) and credit ({credit}) populated.",
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
                        "description": f"Debit-normal account '{acct_type}' has credit balance {balance}.",
                    })
                if acct_type in ("liability", "equity", "revenue") and balance > 0:
                    issues.append({
                        "account_code": r.account_code, "account_name": r.account_name,
                        "issue_type": "UNUSUAL_SIGN",
                        "description": f"Credit-normal account '{acct_type}' has debit balance {balance}.",
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
            "note": "Ledger is balanced." if is_balanced else f"Ledger is OUT OF BALANCE by {difference}.",
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
    "You are a Trial Balance Validator for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `analyze_trial_balance` FIRST. Never guess numbers.",
    "2. NEVER compute or re-sum. The tool is the source of truth.",
    "3. status='PASSED' iff is_balanced=true AND issues is empty. Else 'FAILED'.",
    "4. Include every issue returned by the tool. Do not invent.",
    "5. `summary` is 1-2 sentences for a controller. Mention the difference amount.",
    "6. If found=false, set status='FAILED' and explain the missing-data reason.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Trial Balance Validator",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[analyze_trial_balance],
        description="Validates a trial balance via deterministic tool, summarises results.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=TrialBalanceValidatorResult,
        markdown=False,
    )


def _safe_parse(content: Any) -> TrialBalanceValidatorResult:
    """Parse agent response; raise RuntimeError on Gemini error payloads."""
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


def validate_trial_balance(company_id: str, period: str | None = None) -> TrialBalanceValidatorResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return TrialBalanceValidatorResult(
                company_id=company_id, period="UNKNOWN", status="FAILED",
                total_debits=0.0, total_credits=0.0, difference=0.0,
                account_count=0, issues=[],
                summary=f"No trial balance data exists for company '{company_id}'.",
            )

    prompt = (
        f"Validate the trial balance for company_id={company_id!r} and period={period!r}. "
        f"Call analyze_trial_balance with these exact arguments."
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

    logger.warning("LLM path failed for %s@%s (%s) — falling back.", company_id, period, type(last_exc).__name__)
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
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: TB is {'balanced' if facts.get('is_balanced') else 'out of balance'} "
            f"by {facts.get('difference', 0.0)} across {facts.get('account_count', 0)} accounts."
        ),
    )