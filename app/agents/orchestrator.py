from __future__ import annotations

import time
import uuid
from typing import Any

import redis
from celery import chain, chord, group

from app.core.celery_app import celery_app
from app.db.database import SessionLocal, settings
from app.db.models import Company
from app.agents.validator import validate_trial_balance
from app.agents.variance import run_variance_analysis
from app.agents.cash_flow import run_cash_flow_reconciliation

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
    time.sleep(2)
    return {"run_id": run_id, "company_id": company_id, "agent": "accruals", "status": "success"}


@celery_app.task(name="agents.revenue")
def run_revenue_agent(run_id: str, company_id: str) -> dict[str, Any]:
    time.sleep(2)
    return {"run_id": run_id, "company_id": company_id, "agent": "revenue", "status": "success"}


@celery_app.task(name="agents.expenses")
def run_expenses_agent(run_id: str, company_id: str) -> dict[str, Any]:
    time.sleep(2)
    return {"run_id": run_id, "company_id": company_id, "agent": "expenses", "status": "success"}


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
def phase1_complete(_results: list[Any], run_id: str, company_id: str) -> dict[str, Any]:
    redis_client.set(f"close:{run_id}:phase1:{company_id}", "done", ex=PHASE_TIMEOUT)

    # Phase 2: sequential chain per company → on completion, bump counter
    chain(
        run_accruals_agent.si(run_id, company_id),
        run_revenue_agent.si(run_id, company_id),
        run_expenses_agent.si(run_id, company_id),
        phase2_complete.si(run_id, company_id),
    ).apply_async()

    return {"run_id": run_id, "company_id": company_id, "phase": 1, "status": "done"}


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
    time.sleep(2)  # Replace with real elimination logic in Phase 3
    redis_client.set(f"close:{run_id}:phase3", "done", ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:status", "completed", ex=PHASE_TIMEOUT)
    return {"run_id": run_id, "phase": 3, "status": "success"}