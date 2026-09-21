"""
Cash Flow Reconciliation Agent — Phase 1, Parallel Group.
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
from app.db.models import BankStatement, TrialBalance

logger = logging.getLogger(__name__)

CASH_RECON_TOLERANCE_USD = Decimal("1000")
TOP_N_TRANSACTIONS = 5
CASH_ACCOUNT_PREFIXES = ("1000", "1010", "1020")


class BankTransactionSummary(BaseModel):
    date: str
    description: str
    amount: float


class CashFlowReconciliationResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'RECONCILED' if |gap| <= tolerance, else 'UNRECONCILED'.")
    gl_opening_balance: float
    gl_closing_balance: float
    gl_movement: float
    bank_opening_balance: float
    bank_closing_balance: float
    bank_movement: float
    gap: float
    tolerance_usd: float
    top_transactions: list[BankTransactionSummary] = Field(default_factory=list)
    summary: str


def _parse_period(period: str) -> tuple[int, int]:
    y, m = period.split("-")
    return int(y), int(m)


def _previous_period(period: str) -> str:
    year, month = _parse_period(period)
    if month == 1:
        return f"{year - 1}-12"
    return f"{year}-{month - 1:02d}"


def reconcile_cash_flow(company_id: str, period: str) -> dict[str, Any]:
    """Deterministic GL-vs-bank cash reconciliation. Never raises."""
    db = SessionLocal()
    try:
        prior_period = _previous_period(period)

        def gl_cash_balance(p: str) -> Decimal | None:
            rows = db.scalars(
                select(TrialBalance).where(
                    TrialBalance.company_id == company_id,
                    TrialBalance.period == p,
                )
            ).all()
            if not rows:
                return None
            cash_rows = [
                r for r in rows
                if any((r.account_code or "").startswith(pfx) for pfx in CASH_ACCOUNT_PREFIXES)
            ]
            if not cash_rows:
                return Decimal("0")
            return sum((r.balance or Decimal("0")) for r in cash_rows)

        gl_closing = gl_cash_balance(period)
        gl_opening = gl_cash_balance(prior_period)
        gl_opening_available = gl_opening is not None

        if gl_closing is None:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No trial balance rows for this period.", "status": "UNRECONCILED"}

        if gl_opening is None:
            gl_opening = Decimal("0")

        bank_rows = db.scalars(
            select(BankStatement)
            .where(BankStatement.company_id == company_id, BankStatement.period == period)
            .order_by(BankStatement.date.asc())
        ).all()

        if not bank_rows:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No bank statement rows for this period.", "status": "UNRECONCILED"}

        bank_opening = bank_rows[0].balance or Decimal("0")
        bank_closing = bank_rows[-1].balance or Decimal("0")
        bank_movement = bank_closing - bank_opening

        gl_movement = gl_closing - gl_opening
        gap = gl_movement - bank_movement
        within_tolerance = abs(gap) <= CASH_RECON_TOLERANCE_USD

        def signed_amount(row: BankStatement) -> Decimal:
            return (row.credit or Decimal("0")) - (row.debit or Decimal("0"))

        sorted_txns = sorted(bank_rows, key=lambda r: abs(signed_amount(r)), reverse=True)
        top_txns = [
            {
                "date": r.date.isoformat() if isinstance(r.date, date) else str(r.date),
                "description": r.description or "",
                "amount": float(signed_amount(r)),
            }
            for r in sorted_txns[:TOP_N_TRANSACTIONS]
        ]

        return {
            "company_id": company_id, "period": period, "found": True,
            "gl_opening_balance": float(gl_opening), "gl_closing_balance": float(gl_closing),
            "gl_movement": float(gl_movement), "gl_opening_available": gl_opening_available,
            "bank_opening_balance": float(bank_opening), "bank_closing_balance": float(bank_closing),
            "bank_movement": float(bank_movement), "gap": float(gap),
            "within_tolerance": within_tolerance,
            "tolerance_usd": float(CASH_RECON_TOLERANCE_USD),
            "bank_row_count": len(bank_rows), "top_transactions": top_txns,
            "note": (
                "GL and bank movements agree within tolerance."
                if within_tolerance
                else f"Unexplained gap of {gap} between GL and bank movements."
            ),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("reconcile_cash_flow failed for %s@%s", company_id, period)
        return {"company_id": company_id, "period": period, "found": False,
                "error": str(exc), "status": "UNRECONCILED"}
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    db = SessionLocal()
    try:
        return db.scalar(
            select(BankStatement.period)
            .where(BankStatement.company_id == company_id)
            .order_by(BankStatement.period.desc())
            .limit(1)
        )
    finally:
        db.close()


AGENT_INSTRUCTIONS = [
    "You are a Cash Flow Reconciliation Agent for a Private Equity month-end close system.",
    "",
    "STRICT RULES:",
    "1. ALWAYS call `reconcile_cash_flow` first. Never invent numbers.",
    "2. NEVER recalculate. Cite tool output verbatim.",
    "3. status='RECONCILED' iff within_tolerance=true. Else 'UNRECONCILED'.",
    "4. Include every item from `top_transactions`.",
    "5. `summary` is 2-3 sentences. State gap amount and direction.",
    "6. If found=false, set status='UNRECONCILED' and explain.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Agent(
        name="Cash Flow Reconciliation Agent",
        model=Gemini(id="gemini-3.5-flash", api_key=settings.gemini_api_key),
        tools=[reconcile_cash_flow],
        description="Reconciles GL cash vs bank statement movements.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=CashFlowReconciliationResult,
        markdown=False,
    )


def _safe_parse(content: Any) -> CashFlowReconciliationResult:
    if isinstance(content, CashFlowReconciliationResult):
        return content
    if isinstance(content, dict):
        if "error" in content:
            raise RuntimeError(f"Gemini API error: {content['error']}")
        return CashFlowReconciliationResult(**content)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "error" in parsed:
                raise RuntimeError(f"Gemini API error: {parsed['error']}")
            return CashFlowReconciliationResult(**parsed)
        raise ValueError(f"Non-JSON agent response: {stripped[:200]}")
    raise ValueError(f"Unexpected agent response type: {type(content)}")


def run_cash_flow_reconciliation(company_id: str, period: str | None = None) -> CashFlowReconciliationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return CashFlowReconciliationResult(
                company_id=company_id, period="UNKNOWN", status="UNRECONCILED",
                gl_opening_balance=0.0, gl_closing_balance=0.0, gl_movement=0.0,
                bank_opening_balance=0.0, bank_closing_balance=0.0, bank_movement=0.0,
                gap=0.0, tolerance_usd=float(CASH_RECON_TOLERANCE_USD),
                top_transactions=[], summary=f"No bank data found for '{company_id}'.",
            )

    prompt = (
        f"Reconcile cash for company_id={company_id!r} and period={period!r}. "
        f"Call reconcile_cash_flow with these exact arguments."
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

    logger.warning("LLM path failed for cash flow %s@%s (%s) — falling back.", company_id, period, type(last_exc).__name__)
    facts = reconcile_cash_flow(company_id, period)

    if not facts.get("found"):
        return CashFlowReconciliationResult(
            company_id=company_id, period=period, status="UNRECONCILED",
            gl_opening_balance=0.0, gl_closing_balance=0.0, gl_movement=0.0,
            bank_opening_balance=0.0, bank_closing_balance=0.0, bank_movement=0.0,
            gap=0.0, tolerance_usd=float(CASH_RECON_TOLERANCE_USD),
            top_transactions=[],
            summary=f"LLM unavailable ({type(last_exc).__name__}). {facts.get('error', 'No data.')}",
        )

    txns = [BankTransactionSummary(**t) for t in facts.get("top_transactions", [])]
    return CashFlowReconciliationResult(
        company_id=company_id, period=period,
        status="RECONCILED" if facts.get("within_tolerance") else "UNRECONCILED",
        gl_opening_balance=facts["gl_opening_balance"], gl_closing_balance=facts["gl_closing_balance"],
        gl_movement=facts["gl_movement"], bank_opening_balance=facts["bank_opening_balance"],
        bank_closing_balance=facts["bank_closing_balance"], bank_movement=facts["bank_movement"],
        gap=facts["gap"], tolerance_usd=facts["tolerance_usd"], top_transactions=txns,
        summary=(
            f"LLM unavailable ({type(last_exc).__name__}). "
            f"Deterministic result: gap = {facts['gap']:.2f} USD "
            f"({'within' if facts.get('within_tolerance') else 'outside'} tolerance)."
        ),
    )