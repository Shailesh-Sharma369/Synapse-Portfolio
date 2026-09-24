"""
Cash Flow Reconciliation Agent — Phase 1, ReAct.

ReAct tools:
    1. reconcile_cash_flow   — GL vs bank movement gap
    2. list_uncleared_items  — top N bank transactions by size
    3. get_cash_accounts     — all cash-equivalent GL accounts
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date
from decimal import Decimal
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import BankStatement, TrialBalance

logger = logging.getLogger(__name__)

CASH_RECON_TOLERANCE_USD = Decimal("1000")
TOP_N_TRANSACTIONS = 5
CASH_ACCOUNT_PREFIXES = ("1000", "1010", "1020")
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


class BankTransactionSummary(BaseModel):
    date: str
    description: str
    amount: float


class CashFlowReconciliationResult(BaseModel):
    company_id: str
    period: str
    status: str = Field(..., description="'RECONCILED' or 'UNRECONCILED'.")
    gl_opening_balance: float
    gl_closing_balance: float
    gl_movement: float
    bank_opening_balance: float
    bank_closing_balance: float
    bank_movement: float
    gap: float
    tolerance_usd: float
    top_transactions: list[BankTransactionSummary] = Field(default_factory=list)
    summary: str = Field(..., description="3-5 sentence analysis.")


def _parse_period(period: str) -> tuple[int, int]:
    y, m = period.split("-")
    return int(y), int(m)


def _previous_period(period: str) -> str:
    year, month = _parse_period(period)
    if month == 1:
        return f"{year - 1}-12"
    return f"{year}-{month - 1:02d}"


# =============================================================================
# TOOL 1 — reconcile_cash_flow
# =============================================================================

def reconcile_cash_flow(company_id: str, period: str) -> dict[str, Any]:
    """Deterministic GL-vs-bank cash reconciliation."""
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
            cash_rows = [r for r in rows
                         if any((r.account_code or "").startswith(pfx) for pfx in CASH_ACCOUNT_PREFIXES)]
            if not cash_rows:
                return Decimal("0")
            return sum((r.balance or Decimal("0")) for r in cash_rows)

        gl_closing = gl_cash_balance(period)
        gl_opening = gl_cash_balance(prior_period)
        gl_opening_available = gl_opening is not None

        if gl_closing is None:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No TB rows.", "status": "UNRECONCILED"}
        if gl_opening is None:
            gl_opening = Decimal("0")

        bank_rows = db.scalars(
            select(BankStatement)
            .where(BankStatement.company_id == company_id, BankStatement.period == period)
            .order_by(BankStatement.date.asc())
        ).all()

        if not bank_rows:
            return {"company_id": company_id, "period": period, "found": False,
                    "error": "No bank rows.", "status": "UNRECONCILED"}

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
            {"date": r.date.isoformat() if isinstance(r.date, date) else str(r.date),
             "description": r.description or "",
             "amount": float(signed_amount(r))}
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
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("reconcile_cash_flow failed for %s@%s", company_id, period)
        return {"company_id": company_id, "period": period, "found": False,
                "error": str(exc), "status": "UNRECONCILED"}
    finally:
        db.close()


# =============================================================================
# TOOL 2 — list_uncleared_items
# =============================================================================

def list_uncleared_items(company_id: str, period: str, top_n: int = 10) -> list[dict[str, Any]]:
    """
    List top N bank transactions by absolute size for the period.

    Use this to investigate a cash gap — the largest movements usually
    explain most of the discrepancy.
    """
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(BankStatement)
            .where(BankStatement.company_id == company_id, BankStatement.period == period)
        ).all()
        def signed(r: BankStatement) -> Decimal:
            return (r.credit or Decimal("0")) - (r.debit or Decimal("0"))
        sorted_rows = sorted(rows, key=lambda r: abs(signed(r)), reverse=True)[:top_n]
        return [
            {"date": r.date.isoformat() if isinstance(r.date, date) else str(r.date),
             "description": r.description or "",
             "signed_amount": float(signed(r)),
             "balance_after": float(r.balance or 0)}
            for r in sorted_rows
        ]
    finally:
        db.close()


# =============================================================================
# TOOL 3 — get_cash_accounts
# =============================================================================

def get_cash_accounts(company_id: str, period: str) -> list[dict[str, Any]]:
    """List all GL accounts treated as cash (prefix 1000/1010/1020)."""
    db = SessionLocal()
    try:
        rows = db.scalars(
            select(TrialBalance).where(
                TrialBalance.company_id == company_id,
                TrialBalance.period == period,
            )
        ).all()
        return [
            {"account_code": r.account_code, "account_name": r.account_name,
             "balance": float(r.balance or 0)}
            for r in rows
            if any((r.account_code or "").startswith(pfx) for pfx in CASH_ACCOUNT_PREFIXES)
        ]
    finally:
        db.close()


def _latest_period_for(company_id: str) -> str | None:
    db = SessionLocal()
    try:
        return db.scalar(
            select(BankStatement.period).where(BankStatement.company_id == company_id)
            .order_by(BankStatement.period.desc()).limit(1)
        )
    finally:
        db.close()


AGENT_INSTRUCTIONS = [
    "You are a Cash Flow Reconciliation Agent for a PE month-end close system.",
    "You have THREE tools.",
    "",
    "TOOLS:",
    "  1. reconcile_cash_flow(company_id, period) — baseline. ALWAYS first.",
    "  2. get_cash_accounts(company_id, period) — list cash GL accounts.",
    "  3. list_uncleared_items(company_id, period, top_n) — top bank txns by size.",
    "",
    "WORKFLOW:",
    "  - Step 1: reconcile_cash_flow.",
    "  - Step 2: If gap exists, call list_uncleared_items to find biggest movements.",
    "  - Step 3: Optionally call get_cash_accounts to confirm GL composition.",
    "  - Step 4: STOP after 2-4 tool calls.",
    "",
    "STRICT RULES:",
    "1. NEVER invent numbers.",
    "2. status='RECONCILED' iff within_tolerance=true.",
    "3. Include top_transactions verbatim.",
    "4. `summary` = 3-5 sentences, state gap amount and direction.",
    "",
    "Output ONLY the structured JSON schema.",
]


def _build_agent() -> Agent:
    """Build the Cash Flow Reconciliation Agent."""
    return Agent(
        name="Cash Flow Reconciliation Agent",
        model=get_model(),
        tools=[reconcile_cash_flow, list_uncleared_items, get_cash_accounts],
        description="Reconciles GL cash vs bank movements and investigates gaps.",
        instructions=AGENT_INSTRUCTIONS,
        output_schema=CashFlowReconciliationResult,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
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
        raise ValueError(f"Non-JSON: {stripped[:200]}")
    raise ValueError(f"Unexpected type: {type(content)}")


def run_cash_flow_reconciliation(company_id: str, period: str | None = None) -> CashFlowReconciliationResult:
    if period is None:
        period = _latest_period_for(company_id)
        if period is None:
            return CashFlowReconciliationResult(
                company_id=company_id, period="UNKNOWN", status="UNRECONCILED",
                gl_opening_balance=0.0, gl_closing_balance=0.0, gl_movement=0.0,
                bank_opening_balance=0.0, bank_closing_balance=0.0, bank_movement=0.0,
                gap=0.0, tolerance_usd=float(CASH_RECON_TOLERANCE_USD),
                top_transactions=[], summary=f"No bank data for '{company_id}'.",
            )

    # ---- TIER 1: deterministic pre-check --------------------------------
    precheck = reconcile_cash_flow(company_id, period)
    if precheck.get("found") and precheck.get("within_tolerance"):
        logger.info("[CashFlow] %s@%s clean — skipping LLM.", company_id, period)
        txns = [BankTransactionSummary(**t) for t in precheck.get("top_transactions", [])]
        return CashFlowReconciliationResult(
            company_id=company_id, period=period, status="RECONCILED",
            gl_opening_balance=precheck["gl_opening_balance"],
            gl_closing_balance=precheck["gl_closing_balance"],
            gl_movement=precheck["gl_movement"],
            bank_opening_balance=precheck["bank_opening_balance"],
            bank_closing_balance=precheck["bank_closing_balance"],
            bank_movement=precheck["bank_movement"],
            gap=precheck["gap"],
            tolerance_usd=precheck["tolerance_usd"],
            top_transactions=txns,
            summary=(
                f"GL and bank movements reconcile within tolerance "
                f"(gap ${precheck['gap']:,.2f}). No investigation needed. "
                f"LLM reasoning skipped."
            ),
        )

    prompt = (f"Reconcile cash for company_id={company_id!r} and period={period!r}. "
              f"Use tools to investigate any gap, then produce the structured result.")

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

    logger.warning("LLM path failed for cash flow %s@%s (%s) — falling back.",
                   company_id, period, type(last_exc).__name__)
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
        summary=(f"LLM unavailable ({type(last_exc).__name__}). Deterministic: "
                 f"gap ${facts['gap']:,.2f} "
                 f"({'within' if facts.get('within_tolerance') else 'outside'} tolerance)."),
    )