"""
Month-End Close Orchestrator — Celery task graph for all 5 phases.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================
The pipeline runs 5 phases, each managed by Celery primitives:

    Phase 0  →  Pre-Flight (Orchestrator Agent decides proceed/skip)
    Phase 1  →  8 chords, staggered 15s apart. Each chord = 3 parallel agents.
    Phase 2  →  8 chains, sequential per company (Accruals → RevRec → Expenses).
    Phase 3  →  Single elimination task (Redis SET NX lock).
    Phase 4  →  Single consolidation task.
    Phase 5  →  Reporting Agent decides which emails to send, then executes.
    Phase 6  →  Post-Flight (Orchestrator Agent decides escalation).

================================================================================
AGENTIC LAYERS
================================================================================
Two Agno agents wrap the deterministic Celery pipeline:

    - orchestrator_agent.run_preflight()   → proceed / skip decision
    - orchestrator_agent.run_postflight()  → escalation decision
    - reporting_agent.plan_emails()        → which emails to send

All three degrade gracefully: if the LLM is unavailable, we fall back to
safe defaults (proceed=True, escalate=False, send completion email).
================================================================================
"""

from __future__ import annotations

import time
import uuid
import os    
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

redis_client: redis.Redis = redis.from_url(settings.redis_url, decode_responses=True)
PHASE_TIMEOUT = 60 * 60


# ---------------------------------------------------------------------------
# SELF-HEALING RETRY POLICY
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
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
    result = validate_trial_balance(company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "trial_balance"
    return payload


@celery_app.task(name="agents.variance", **_RETRY_KW)
def run_variance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 agent — Variance Analysis."""
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
    result = run_variance_analysis(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "variance"
    return payload


@celery_app.task(name="agents.cash_flow", **_RETRY_KW)
def run_cash_flow_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 1 agent — Cash Flow Reconciliation."""
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
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
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
    result = run_accrual_verification(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "accruals"
    return payload


@celery_app.task(name="agents.revenue", **_RETRY_KW)
def run_revenue_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 2 — Revenue Recognition (ASC 606)."""
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
    result = run_revenue_recognition(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "revenue"
    return payload


@celery_app.task(name="agents.expenses", **_RETRY_KW)
def run_expenses_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """Phase 2 Step 3 — Expense Categorization."""
    if settings.agent_stagger_seconds > 0:
        time.sleep(settings.agent_stagger_seconds)
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

    Order of operations (critical for UI responsiveness):
      1. Set status=running + latest_run_id          (UI shows "running" instantly)
      2. Fetch companies + apply CLOSE_COMPANIES     (fast DB call)
      3. Write total_companies + companies to Redis  (UI knows the target)
      4. Pre-Flight (agentic — LLM, ~30-60s)         (UI already showing progress)
      5. Dispatch Phase-1 chords (staggered)
    """
    # ---- Close-week guard ------------------------------------------------
    if close_week_only:
        from datetime import date as _date
        if _date.today().day > 5:
            return {"status": "skipped", "reason": "outside close week (days 1-5)"}

    run_id = run_id or str(uuid.uuid4())

    # ---- 1. Initialize state FIRST — UI shows "running" immediately ------
    redis_client.set(f"close:{run_id}:status", "running", ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:phase2_count", 0, ex=PHASE_TIMEOUT)
    redis_client.set("close:latest_run_id", run_id, ex=PHASE_TIMEOUT)

    # ---- 2. Fetch company list -------------------------------------------
    db = SessionLocal()
    try:
        company_ids = [str(cid) for (cid,) in db.query(Company.id).all()]
    finally:
        db.close()

    # ---- DEMO FILTER (env-controlled) -----------------------------------
    _demo_filter = os.getenv("CLOSE_COMPANIES", "").strip()
    if _demo_filter:
        allowed = {c.strip() for c in _demo_filter.split(",") if c.strip()}
        before = len(company_ids)
        company_ids = [cid for cid in company_ids if cid in allowed]
        logger.info(
            "[Orchestrator] DEMO MODE — filtered %d → %d companies (CLOSE_COMPANIES=%s)",
            before, len(company_ids), _demo_filter,
        )

    if not company_ids:
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        _set_escalation(run_id, "no_companies_found")
        return {"run_id": run_id, "status": "failed", "reason": "no companies found"}

    # ---- 3. Persist company list + total BEFORE preflight ---------------
    redis_client.set(f"close:{run_id}:total_companies", len(company_ids), ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:companies", json.dumps(company_ids), ex=PHASE_TIMEOUT)
    redis_client.lpush("close:runs:recent", run_id)
    redis_client.ltrim("close:runs:recent", 0, 49)

    # ---- 4. PHASE 0: Pre-Flight (agentic) -------------------------------
    try:
        from app.agents.orchestrator_agent import run_preflight
        pre = run_preflight(run_id)
        if not pre.proceed:
            logger.warning("[Orchestrator] Pre-Flight REJECTED: %s", pre.reason)
            redis_client.set(f"close:{run_id}:status", "skipped", ex=PHASE_TIMEOUT)
            return {"run_id": run_id, "status": "skipped", "reason": pre.reason}
        logger.info("[Orchestrator] Pre-Flight OK: %s", pre.reason)
        redis_client.set(
            f"close:{run_id}:preflight",
            json.dumps(pre.model_dump(), default=str),
            ex=PHASE_TIMEOUT,
        )
        redis_client.set(f"close:{run_id}:phase0_status", "done", ex=PHASE_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Orchestrator] Pre-Flight error (%s) — proceeding.",
                       type(exc).__name__)

    # ---- 5. Dispatch Phase 1, staggered ---------------------------------
    stagger = settings.chord_stagger_seconds
    for idx, company_id in enumerate(company_ids):
        if idx > 0 and stagger > 0:
            time.sleep(stagger)

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
    """Chord callback: fires when all three Phase-1 agents finish for one company."""
    redis_client.set(f"close:{run_id}:phase1:{company_id}", "done", ex=PHASE_TIMEOUT)

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
    """Chain tail: fires after the three Phase-2 agents finish for one company."""
    redis_client.set(f"close:{run_id}:phase2:{company_id}", "done", ex=PHASE_TIMEOUT)

    count = int(redis_client.incr(f"close:{run_id}:phase2_count"))

    # Resolve total from the most reliable source available, with fallbacks.
    total = 0
    raw = redis_client.get(f"close:{run_id}:total_companies")
    if raw:
        try:
            total = int(raw)
        except (TypeError, ValueError):
            total = 0
    if total == 0:
        companies_raw = redis_client.get(f"close:{run_id}:companies")
        if companies_raw:
            try:
                total = len(json.loads(companies_raw))
            except Exception:
                total = 0

    # CRITICAL GUARD: only fire Phase 3 when we have a non-zero target AND
    # every company has actually completed Phase 2. Without the `total > 0`
    # check, `count >= 0` is always True and Phase 3 fires prematurely the
    # moment any single company finishes — which is what caused the UI to
    # show "Phase 1/2 pending" while "Phase 3/4 done".
    if total > 0 and count >= total:
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
    """
    Phase 4 — Final Consolidation.

    On success:
      1. Persist final_result to Redis.
      2. Mark run status = 'completed'.
      3. Call Post-Flight (agentic) to decide escalation.
      4. Hand off to Phase 5 (reporting).
    """
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

        # ---- PHASE 6: Post-Flight (agentic) -----------------------------
        try:
            from app.agents.orchestrator_agent import run_postflight
            post = run_postflight(run_id)
            logger.info(
                "[Orchestrator] Post-Flight: %s (escalate=%s)",
                post.status_summary, post.escalate_to_human,
            )
            redis_client.set(
                f"close:{run_id}:postflight",
                json.dumps(post.model_dump(), default=str),
                ex=PHASE_TIMEOUT,
            )
            redis_client.set(f"close:{run_id}:postflight_status", "done", ex=PHASE_TIMEOUT)
            if post.escalate_to_human:
                _set_escalation(
                    run_id, post.escalation_reason or "postflight_escalation"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Orchestrator] Post-Flight error (%s).", type(exc).__name__)

        # ---- Hand off to Phase 5 ----------------------------------------
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
# PHASE 5 — Reporting & Communication (agentic)
# ============================================================================

@celery_app.task(name="orchestrator.run_reporting_task", **_RETRY_KW)
def run_reporting_task(run_id: str) -> dict[str, Any]:
    """
    Phase 5 — Reporting & Communication.

    The Reporting Agent (reporting_agent.plan_emails) DECIDES which email
    types to send based on run state. The executor functions in
    app.agents.reporting then send them.

    If the planner LLM fails, we default to sending only the completion email.
    """
    logger.info("[Phase 5] Reporting started for run %s", run_id)

    try:
        # ---- Decide plan (agentic) --------------------------------------
        try:
            from app.agents.reporting_agent import plan_emails
            plan = plan_emails(run_id)
            emails_to_send = plan.emails_to_send
            logger.info(
                "[Phase 5] Reporting plan: %s (priority=%s) — %s",
                emails_to_send, plan.priority, plan.reasoning,
            )
            redis_client.set(
                f"close:{run_id}:email_plan",
                json.dumps(plan.model_dump(), default=str),
                ex=PHASE_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001  # noqa: BLE001
            logger.warning(
                "[Phase 5] Planner failed (%s) — defaulting to completion email.",
                type(exc).__name__,
            )
            emails_to_send = ["completion"]

        # ---- Execute plan -----------------------------------------------
        sent: list[str] = []
        for email_type in emails_to_send:
            try:
                if email_type == "completion":
                    if send_executive_summary_email(run_id):
                        sent.append("completion")
                elif email_type == "issue_alert":
                    if send_issue_alert_email(run_id):
                        sent.append("issue_alert")
                elif email_type == "daily_summary":
                    if send_daily_summary_email(run_id):
                        sent.append("daily_summary")
                elif email_type == "weekly_report":
                    if send_weekly_stakeholder_report(run_id):
                        sent.append("weekly_report")
                else:
                    logger.warning("[Phase 5] Unknown email type: %s", email_type)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Phase 5] Email %s failed: %s", email_type, exc)

        status = "done" if sent else "failed"
        redis_client.set(f"close:{run_id}:reporting", status, ex=PHASE_TIMEOUT)
        redis_client.set(
            f"close:{run_id}:reporting_emails_sent",
            ",".join(sent), ex=PHASE_TIMEOUT,
        )
        redis_client.set(
            f"close:{run_id}:reporting_email",
            settings.to_email if hasattr(settings, "to_email") else "",
            ex=PHASE_TIMEOUT,
        )

        logger.info("[Phase 5] Reporting %s — emails sent: %s", status, sent)
        return {"run_id": run_id, "phase": 5, "status": status, "emails_sent": sent}

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 5] Reporting crashed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:reporting", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "phase": 5, "status": "failed", "error": str(exc)}


# ============================================================================
# PHASE 5b — Scheduled Notification Tasks (Daily / Weekly / Issue Alerts)
# ============================================================================

@celery_app.task(name="orchestrator.send_daily_summary", **_RETRY_KW)
def send_daily_summary_task() -> dict[str, Any]:
    """Daily 8 AM progress summary."""
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
    """Hourly issue-alert sweep. Fires only if an issue exists."""
    run_id = redis_client.get("close:latest_run_id")
    if not run_id:
        return {"status": "no_active_run"}

    ok = send_issue_alert_email(run_id)
    logger.info("[IssueAlert] Sent=%s for run %s", ok, run_id)
    return {"run_id": run_id, "sent": ok}