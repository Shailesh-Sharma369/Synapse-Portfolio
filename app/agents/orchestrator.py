from __future__ import annotations

import time
import uuid
from typing import Any
import json

import redis
from celery import chain, chord, group

from app.core.celery_app import celery_app
from app.db.database import SessionLocal, settings
from app.db.models import Company
from app.agents.validator import validate_trial_balance
from app.agents.variance import run_variance_analysis
from app.agents.cash_flow import run_cash_flow_reconciliation
from app.agents.accrual_verification import run_accrual_verification
from app.agents.revenue_recognition import run_revenue_recognition
from app.agents.expense_categorization import run_expense_categorization
from app.agents.elimination import run_elimination
from app.agents.consolidation import run_consolidation
from app.agents.reporting import send_executive_summary_email

import logging

logger = logging.getLogger(__name__)

redis_client: redis.Redis = redis.from_url(settings.redis_url, decode_responses=True)

PHASE_TIMEOUT = 60 * 30  # 30 min safety expiry on state keys


# ---------------------------------------------------------------------------
# Dummy agent tasks (Phase 1 - run in parallel per company)
# ---------------------------------------------------------------------------
@celery_app.task(name="agents.trial_balance")
def run_trial_balance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 — Trial Balance Validator.

    Runs the Agno-powered validator for one company. Returns a JSON-safe dict
    so Celery's JSON serializer can carry it into the chord callback.
    """
    result=validate_trial_balance(company_id,period=None)# None → latest period
    payload=result.model_dump()
    payload["run_id"]=run_id
    payload["agent"]="trial_balance"
    time.sleep(2)
    return payload


@celery_app.task(name="agents.variance")
def run_variance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 — Variance Analysis. Runs in parallel with TB Validator and Cash Flow."""
    result = run_variance_analysis(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "variance"
    return payload

@celery_app.task(name="agents.cash_flow")
def run_cash_flow_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 — Cash Flow Reconciliation. Runs in parallel with TB and Variance."""
    result = run_cash_flow_reconciliation(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "cash_flow"
    return payload

# ---------------------------------------------------------------------------
# Dummy agent tasks (Phase 2 - run sequentially per company)
# ---------------------------------------------------------------------------
@celery_app.task(name="agents.accruals")
def run_accruals_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 1 — Accrual Verification (sequential)."""
    result = run_accrual_verification(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "accruals"
    return payload


@celery_app.task(name="agents.revenue")
def run_revenue_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 2 — Revenue Recognition (runs after accruals)."""
    result = run_revenue_recognition(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "revenue"
    return payload


@celery_app.task(name="agents.expenses")
def run_expenses_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 3 — Expense Categorization (runs after revenue)."""
    result = run_expense_categorization(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "expenses"
    return payload

# ---------------------------------------------------------------------------
# Master orchestration
# ---------------------------------------------------------------------------
@celery_app.task(name="orchestrator.run_month_end_close")
def run_month_end_close(run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())

    redis_client.set(f"close:{run_id}:status", "running", ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:phase2_count", 0, ex=PHASE_TIMEOUT)

    db = SessionLocal()
    try:
        company_ids = [str(cid) for (cid,) in db.query(Company.id).all()]
    finally:
        db.close()

    if not company_ids:
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "status": "failed", "reason": "no companies found"}

    redis_client.set(f"close:{run_id}:total_companies", len(company_ids), ex=PHASE_TIMEOUT)
    redis_client.lpush("close:runs:recent", run_id)
    redis_client.ltrim("close:runs:recent", 0, 49)

    # Phase 1: parallel chord per company → on completion, kick Phase 2 chain
    for company_id in company_ids:
        chord(
            group(
                run_trial_balance_agent.s(run_id, company_id),
                run_variance_agent.s(run_id, company_id),
                run_cash_flow_agent.s(run_id, company_id),
            )
        )(phase1_complete.s(run_id, company_id))

    return {"run_id": run_id, "status": "initiated", "companies": len(company_ids)}


@celery_app.task(name="orchestrator.phase1_complete")
def phase1_complete(results: list[Any], run_id: str, company_id: str) -> dict[str, Any]:
    """
    Fires when all 3 Phase-1 agents for one company finish.

    Transition guard: if any Phase-1 agent returned a non-PASSED status, we
    record the failure in Redis but STILL proceed to Phase 2. Rationale: the
    controller needs the full picture across all phases, not a partial run.
    Phase 3 consolidation reads these failure keys to annotate the final report.
    """
    redis_client.set(f"close:{run_id}:phase1:{company_id}", "done", ex=PHASE_TIMEOUT)

    # Audit: which Phase-1 agents failed for this company?
    phase1_failures: list[str] = []
    for r in results or []:
        if isinstance(r, dict):
            agent = r.get("agent", "unknown")
            status = str(r.get("status", "")).upper()
            if status in ("FAILED", "UNRECONCILED"):
                phase1_failures.append(agent)

    if phase1_failures:
        redis_client.set(
            f"close:{run_id}:phase1_failures:{company_id}",
            ",".join(phase1_failures),
            ex=PHASE_TIMEOUT,
        )
        logger.warning(
            "Phase 1 for %s had non-PASSED status in: %s — proceeding to Phase 2.",
            company_id, phase1_failures,
        )

    # Phase 2: sequential chain per company → on completion, bump counter
    chain(
        run_accruals_agent.si(run_id, company_id),
        run_revenue_agent.si(run_id, company_id),
        run_expenses_agent.si(run_id, company_id),
        phase2_complete.si(run_id, company_id),
    ).apply_async()

    return {
        "run_id": run_id, "company_id": company_id,
        "phase": 1, "status": "done",
        "phase1_failures": phase1_failures,
    }

@celery_app.task(name="orchestrator.phase2_complete")
def phase2_complete(run_id: str, company_id: str) -> dict[str, Any]:
    redis_client.set(f"close:{run_id}:phase2:{company_id}", "done", ex=PHASE_TIMEOUT)

    count = int(redis_client.incr(f"close:{run_id}:phase2_count"))
    total = int(redis_client.get(f"close:{run_id}:total_companies") or 0)

    if count >= total:
        # Atomic lock so only ONE worker triggers Phase 3
        locked = redis_client.set(f"close:{run_id}:phase3_lock", "1", nx=True, ex=PHASE_TIMEOUT)
        if locked:
            run_cross_company_elimination.delay(run_id)

    return {"run_id": run_id, "company_id": company_id, "phase2_count": count, "total": total}


@celery_app.task(name="orchestrator.run_cross_company_elimination")
def run_cross_company_elimination(run_id: str) -> dict[str, Any]:
    """
    Phase 3 — Cross-Company Intercompany Elimination.

    Fires exactly ONCE per month-end close run, after ALL 8 companies have
    completed their Phase 2 chains. Does the elimination reconciliation and
    then HANDS OFF to Phase 4 (consolidation) instead of marking the run
    complete itself — Phase 4 owns the final status stamp.
    """
    logger.info("[Phase 3] Intercompany elimination started for run %s", run_id)

    try:
        result = run_elimination(period=None)
        payload = result.model_dump()
        payload["run_id"] = run_id
        payload["phase"] = 3

        # Persist full structured result for Phase 4 and the UI
        redis_client.set(
            f"close:{run_id}:phase3:result",
            json.dumps(payload, default=str),
            ex=PHASE_TIMEOUT,
        )
        redis_client.set(f"close:{run_id}:phase3", "done", ex=PHASE_TIMEOUT)
        redis_client.set(
            f"close:{run_id}:phase3_status", payload["status"], ex=PHASE_TIMEOUT
        )
        # NOTE: we do NOT set `status=completed` here — Phase 4 owns that.

        logger.info(
            "[Phase 3] Elimination complete — status=%s, mismatches=%d, asymmetry=$%.2f",
            payload["status"], payload.get("mismatch_count", 0),
            payload.get("total_asymmetry_usd", 0.0),
        )

        # ---- Hand off to Phase 4 ----------------------------------------
        run_consolidation_task.delay(run_id)
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 3] Elimination failed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:phase3", "failed", ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "phase": 3, "status": "failed", "error": str(exc)}


@celery_app.task(name="orchestrator.run_consolidation_task")
def run_consolidation_task(run_id: str) -> dict[str, Any]:
    """Phase 4 — Final Consolidation."""
    logger.info("[Phase 4] Consolidation started for run %s", run_id)

    try:
        result = run_consolidation(period=None, run_id=run_id)
        payload = result.model_dump()
        payload["run_id"] = run_id
        payload["phase"] = 4

        redis_client.set(
            f"close:{run_id}:final_result",
            json.dumps(payload, default=str),
            ex=PHASE_TIMEOUT,
        )
        redis_client.set(f"close:{run_id}:phase4", "done", ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:phase4_status", payload["status"], ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:status", "completed", ex=PHASE_TIMEOUT)

        # ---- Hand off to Phase 5 (Reporting) -----------------------------
        run_reporting_task.delay(run_id)

        logger.info(
            "[Phase 4] Consolidation complete — adjusted EBITDA $%.2f across %d entities",
            payload.get("adjusted_group_ebitda", 0.0), payload.get("entity_count", 0),
        )
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 4] Consolidation failed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:phase4", "failed", ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "phase": 4, "status": "failed", "error": str(exc)}

@celery_app.task(name="orchestrator.run_reporting_task")
def run_reporting_task(run_id: str) -> dict[str, Any]:
    """
    Phase 5 — Reporting & Communication.

    Sends the executive summary email to stakeholders. Runs AFTER the close
    is marked 'completed' — a failure here does NOT roll back the pipeline.

    State keys:
        close:{run_id}:reporting       → 'done' | 'failed'
        close:{run_id}:reporting_email → recipient (for audit)
    """
    logger.info("[Phase 5] Reporting started for run %s", run_id)

    try:
        ok = send_executive_summary_email(run_id)
        status = "done" if ok else "failed"

        redis_client.set(f"close:{run_id}:reporting", status, ex=PHASE_TIMEOUT)
        redis_client.set(
            f"close:{run_id}:reporting_email",
            settings.to_email if hasattr(settings, "to_email") else "",
            ex=PHASE_TIMEOUT,
        )

        logger.info("[Phase 5] Reporting %s for run %s", status, run_id)
        return {"run_id": run_id, "phase": 5, "status": status}

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 5] Reporting crashed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:reporting", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "phase": 5, "status": "failed", "error": str(exc)}