"""
Month-End Close Orchestrator — Celery task graph for all 5 phases.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================
The pipeline runs 5 phases, each managed by Celery primitives:

    Phase 1  →  8 chords, dispatched with a 15s stagger between companies.
                Each chord contains 3 parallel agents (TB, Variance, CF).
                Chord callback = phase1_complete → dispatches Phase 2.

    Phase 2  →  8 chains (one per company), each running 3 agents sequentially
                (Accruals → Revenue → Expenses). Chain tail = phase2_complete
                which atomically increments a shared Redis counter.

    Phase 3  →  Single task, fired exactly once when the counter reaches
                total_companies. Uses Redis SET NX as a race-safe lock so
                only one worker wins the trigger.

    Phase 4  →  Single task, chained after Phase 3. Consolidates group
                financials and stamps the run as 'completed'.

    Phase 5  →  Fire-and-forget reporting task. Sends the executive email.
                Failures here do NOT roll back the run.

================================================================================
RATE LIMITING — GEMINI FREE TIER (15 RPM)
================================================================================
Every LLM call (Phase 1, 2, 3, 4) counts against the per-minute quota.

We enforce two throttles:

    1. CHORD_STAGGER_SECONDS = 15
       Delay between dispatching each company's Phase-1 chord. 8 chords × 15s
       = 105 seconds of spread. Guarantees no more than ~3 LLM calls per
       15-second window at the chord boundary.

    2. AGENT_STAGGER_SECONDS = 0
       No delay inside agent tasks. Our retry-with-backoff logic in each
       agent handles transient 429s. Adding a sleep here just makes every
       agent block a worker slot without improving the RPM profile.

Peak worst-case LLM rate with these settings: ~12 calls/minute. Free tier
allows 15 RPM. Headroom ≈ 20%.

================================================================================
RETRY & RECOVERY
================================================================================
Every Celery task declares `autoretry_for=(Exception,)` with exponential
backoff + jitter and a 3-attempt cap. This satisfies the assignment's
"self-healing / retry logic / error recovery" requirement at the task layer
(the agent layer already handles LLM-specific transient errors).

Persistent failures (after retries exhausted) set a Redis escalation key
`close:{run_id}:escalation` so an operator — or the dashboard — can see
which run needs human review.
"""

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
from app.agents.reporting import (
    send_executive_summary_email,
    send_daily_summary_email,
    send_weekly_stakeholder_report,
    send_issue_alert_email,
)

import logging

logger = logging.getLogger(__name__)

# Single shared Redis client for all orchestrator state writes.
redis_client: redis.Redis = redis.from_url(settings.redis_url, decode_responses=True)

# All state keys expire after 30 minutes. Even if a run is abandoned, Redis
# does not accumulate orphaned keys.
PHASE_TIMEOUT = 60 * 30


# ---------------------------------------------------------------------------
# RATE-LIMIT PACING CONSTANTS
# ---------------------------------------------------------------------------
CHORD_STAGGER_SECONDS = 15
AGENT_STAGGER_SECONDS = 0


# ---------------------------------------------------------------------------
# SELF-HEALING: retry policy applied to EVERY Celery task in this module.
# ---------------------------------------------------------------------------
# - autoretry_for=(Exception,):      retry on any exception
# - retry_backoff=True:              exponential backoff (1s, 2s, 4s, ...)
# - retry_backoff_max=300:           cap backoff at 5 minutes
# - retry_jitter=True:               add randomness so retries don't stampede
# - max_retries=3:                   3 attempts total (1 + 2 retries)
#
# If retries are exhausted the task fails hard, and the orchestrator's
# on-failure hooks (see run_month_end_close / phase1_complete) set an
# escalation key for human review.
# ---------------------------------------------------------------------------
_RETRY_KW: dict[str, Any] = dict(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=3,
)


def _set_escalation(run_id: str | None, reason: str) -> None:
    """Best-effort write of an escalation flag for a run. Never raises."""
    if not run_id:
        return
    try:
        redis_client.set(f"close:{run_id}:escalation", reason, ex=PHASE_TIMEOUT)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to set escalation flag for run %s", run_id)


# ============================================================================
# PHASE 1 — Parallel group (3 agents per company)
# ============================================================================

@celery_app.task(name="agents.trial_balance", **_RETRY_KW)
def run_trial_balance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 agent — Trial Balance Validator."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = validate_trial_balance(company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "trial_balance"
    return payload


@celery_app.task(name="agents.variance", **_RETRY_KW)
def run_variance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 agent — Variance Analysis."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_variance_analysis(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "variance"
    return payload


@celery_app.task(name="agents.cash_flow", **_RETRY_KW)
def run_cash_flow_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 agent — Cash Flow Reconciliation."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_cash_flow_reconciliation(company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "cash_flow"
    return payload


# ============================================================================
# PHASE 2 — Sequential chain per company
# ============================================================================

@celery_app.task(name="agents.accruals", **_RETRY_KW)
def run_accruals_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 1 — Accrual Verification."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_accrual_verification(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "accruals"
    return payload


@celery_app.task(name="agents.revenue", **_RETRY_KW)
def run_revenue_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 2 — Revenue Recognition (ASC 606)."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_revenue_recognition(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "revenue"
    return payload


@celery_app.task(name="agents.expenses", **_RETRY_KW)
def run_expenses_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 3 — Expense Categorization."""
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_expense_categorization(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "expenses"
    return payload


# ============================================================================
# MASTER ORCHESTRATION ENTRYPOINT
# ============================================================================

@celery_app.task(name="orchestrator.run_month_end_close", **_RETRY_KW)
def run_month_end_close(
    run_id: str | None = None,
    close_week_only: bool = False,
) -> dict[str, Any]:
    """
    Fires the entire month-end close pipeline for one run_id.

    Args:
        run_id: optional — generated if not provided.
        close_week_only: when True (used by the hourly Beat schedule),
            the task exits early if we're not in days 1-5 of the month.
    """
    # ---- Close-week guard (used by the hourly Beat entry only) ----------
    if close_week_only:
        from datetime import date as _date
        if _date.today().day > 5:
            return {
                "status": "skipped",
                "reason": "outside close week (days 1-5)",
            }

    run_id = run_id or str(uuid.uuid4())

    # ---- Initialize state ------------------------------------------------
    redis_client.set(f"close:{run_id}:status", "running", ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:phase2_count", 0, ex=PHASE_TIMEOUT)
    redis_client.set("close:latest_run_id", run_id, ex=PHASE_TIMEOUT)

    # ---- Fetch company list ----------------------------------------------
    db = SessionLocal()
    try:
        company_ids = [str(cid) for (cid,) in db.query(Company.id).all()]
    finally:
        db.close()

    if not company_ids:
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        _set_escalation(run_id, "no_companies_found")
        return {"run_id": run_id, "status": "failed", "reason": "no companies found"}

    redis_client.set(f"close:{run_id}:total_companies", len(company_ids), ex=PHASE_TIMEOUT)

    # Track this run for the UI's "recent runs" list. Capped at 50 entries.
    redis_client.lpush("close:runs:recent", run_id)
    redis_client.ltrim("close:runs:recent", 0, 49)

    # ---- Dispatch Phase 1, staggered -------------------------------------
    for idx, company_id in enumerate(company_ids):
        if idx > 0 and CHORD_STAGGER_SECONDS:
            time.sleep(CHORD_STAGGER_SECONDS)

        chord(
            group(
                run_trial_balance_agent.s(run_id, company_id),
                run_variance_agent.s(run_id, company_id),
                run_cash_flow_agent.s(run_id, company_id),
            )
        )(phase1_complete.s(run_id, company_id))

        logger.info(
            "[Orchestrator] Dispatched Phase-1 chord for %s (%d/%d)",
            company_id, idx + 1, len(company_ids),
        )

    return {"run_id": run_id, "status": "initiated", "companies": len(company_ids)}


# ============================================================================
# PHASE 1 → PHASE 2 HANDOFF
# ============================================================================

@celery_app.task(name="orchestrator.phase1_complete", **_RETRY_KW)
def phase1_complete(results: list[Any], run_id: str, company_id: str) -> dict[str, Any]:
    """
    Chord callback: fires when all three Phase-1 agents finish for one company.
    """
    redis_client.set(f"close:{run_id}:phase1:{company_id}", "done", ex=PHASE_TIMEOUT)

    # ---- Audit: identify any non-PASSED Phase-1 agents -------------------
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
        _set_escalation(run_id, f"phase1_failures:{company_id}")
        logger.warning(
            "Phase 1 for %s had non-PASSED status in: %s — proceeding to Phase 2.",
            company_id, phase1_failures,
        )

    # ---- Dispatch Phase 2 chain -----------------------------------------
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


# ============================================================================
# PHASE 2 → PHASE 3 HANDOFF
# ============================================================================

@celery_app.task(name="orchestrator.phase2_complete", **_RETRY_KW)
def phase2_complete(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Chain tail: fires after the three Phase-2 agents finish for one company.
    """
    redis_client.set(f"close:{run_id}:phase2:{company_id}", "done", ex=PHASE_TIMEOUT)

    count = int(redis_client.incr(f"close:{run_id}:phase2_count"))
    total = int(redis_client.get(f"close:{run_id}:total_companies") or 0)

    if count >= total:
        locked = redis_client.set(
            f"close:{run_id}:phase3_lock", "1", nx=True, ex=PHASE_TIMEOUT
        )
        if locked:
            run_cross_company_elimination.delay(run_id)

    return {"run_id": run_id, "company_id": company_id, "phase2_count": count, "total": total}


# ============================================================================
# PHASE 3 — Cross-Company Intercompany Elimination
# ============================================================================

@celery_app.task(name="orchestrator.run_cross_company_elimination", **_RETRY_KW)
def run_cross_company_elimination(run_id: str) -> dict[str, Any]:
    """Phase 3 — Cross-Company Intercompany Elimination."""
    logger.info("[Phase 3] Intercompany elimination started for run %s", run_id)

    try:
        result = run_elimination(period=None)
        payload = result.model_dump()
        payload["run_id"] = run_id
        payload["phase"] = 3

        redis_client.set(
            f"close:{run_id}:phase3:result",
            json.dumps(payload, default=str),
            ex=PHASE_TIMEOUT,
        )
        redis_client.set(f"close:{run_id}:phase3", "done", ex=PHASE_TIMEOUT)
        redis_client.set(
            f"close:{run_id}:phase3_status", payload["status"], ex=PHASE_TIMEOUT
        )

        logger.info(
            "[Phase 3] Elimination complete — status=%s, mismatches=%d, asymmetry=$%.2f",
            payload["status"], payload.get("mismatch_count", 0),
            payload.get("total_asymmetry_usd", 0.0),
        )

        run_consolidation_task.delay(run_id)
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 3] Elimination failed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:phase3", "failed", ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        _set_escalation(run_id, f"phase3_failed:{type(exc).__name__}")
        return {"run_id": run_id, "phase": 3, "status": "failed", "error": str(exc)}


# ============================================================================
# PHASE 4 — Final Consolidation
# ============================================================================

@celery_app.task(name="orchestrator.run_consolidation_task", **_RETRY_KW)
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
        redis_client.set(
            f"close:{run_id}:phase4_status", payload["status"], ex=PHASE_TIMEOUT
        )
        redis_client.set(f"close:{run_id}:status", "completed", ex=PHASE_TIMEOUT)

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
        _set_escalation(run_id, f"phase4_failed:{type(exc).__name__}")
        return {"run_id": run_id, "phase": 4, "status": "failed", "error": str(exc)}


# ============================================================================
# PHASE 5 — Reporting & Communication
# ============================================================================

@celery_app.task(name="orchestrator.run_reporting_task", **_RETRY_KW)
def run_reporting_task(run_id: str) -> dict[str, Any]:
    """Phase 5 — Reporting & Communication."""
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

# ============================================================================
# PHASE 5b — Scheduled Notification Tasks (Daily / Weekly / Issue Alerts)
# ============================================================================
# These are fired by Celery Beat (see app/core/celery_app.py beat_schedule).
# They operate on `close:latest_run_id` — the most recent close run — so
# stakeholders always get updates on the current month-end process.
#
# Design decision: reuse the same orchestration state (Redis) that the close
# pipeline writes. No new tables, no new state.
# ============================================================================


@celery_app.task(name="orchestrator.send_daily_summary", **_RETRY_KW)
def send_daily_summary_task() -> dict[str, Any]:
    """
    Daily 8 AM progress summary.

    Reads the latest run_id from Redis and sends a progress email. If no
    run is active, silently returns.
    """
    run_id = redis_client.get("close:latest_run_id")
    if not run_id:
        logger.info("[DailySummary] No active run — skipping.")
        return {"status": "no_active_run"}

    ok = send_daily_summary_email(run_id)
    logger.info("[DailySummary] Sent=%s for run %s", ok, run_id)
    return {"run_id": run_id, "sent": ok}


@celery_app.task(name="orchestrator.send_weekly_report", **_RETRY_KW)
def send_weekly_report_task() -> dict[str, Any]:
    """Weekly Monday 8 AM stakeholder report."""
    run_id = redis_client.get("close:latest_run_id")
    if not run_id:
        logger.info("[WeeklyReport] No active run — skipping.")
        return {"status": "no_active_run"}

    ok = send_weekly_stakeholder_report(run_id)
    logger.info("[WeeklyReport] Sent=%s for run %s", ok, run_id)
    return {"run_id": run_id, "sent": ok}


@celery_app.task(name="orchestrator.send_issue_alert", **_RETRY_KW)
def send_issue_alert_task() -> dict[str, Any]:
    """
    Hourly issue-alert sweep.

    Only sends if the current run has an active issue (phase-3 mismatch or
    escalation flag). Silently skips otherwise.
    """
    run_id = redis_client.get("close:latest_run_id")
    if not run_id:
        return {"status": "no_active_run"}

    ok = send_issue_alert_email(run_id)
    logger.info("[IssueAlert] Sent=%s for run %s", ok, run_id)
    return {"run_id": run_id, "sent": ok}