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
from app.agents.reporting import send_executive_summary_email

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
# Gemini free tier: 15 requests per minute per project. Our Phase-1 dispatch
# bursts 24 LLM calls (8 companies × 3 agents). Without pacing we hit 429
# immediately. Two knobs:
#
#   CHORD_STAGGER_SECONDS   → sleep BETWEEN company chords.
#   AGENT_STAGGER_SECONDS   → sleep INSIDE each agent task, before the LLM.
#
# Rule of thumb: 15s chord stagger alone gives ~3 calls/15s = 12/min worst
# case. Setting AGENT_STAGGER to 0 avoids blocking worker slots without
# affecting the RPM profile — the real throttle is at dispatch time.
# ---------------------------------------------------------------------------
CHORD_STAGGER_SECONDS = 15
AGENT_STAGGER_SECONDS = 0


# ============================================================================
# PHASE 1 — Parallel group (3 agents per company)
# ============================================================================
# Each agent:
#   1. Runs the deterministic Python tool (SQL + Decimal math).
#   2. Passes the tool's output to Gemini for reasoning + narrative.
#   3. Returns a JSON-safe dict.
#
# The chord() dispatch in run_month_end_close() collects all three results
# and hands them to phase1_complete() only after all three finish.
# ============================================================================

@celery_app.task(name="agents.trial_balance")
def run_trial_balance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 1 agent — Trial Balance Validator.

    Delegates to `validate_trial_balance()` in app/agents/validator.py,
    which:
        - Loads the company's trial balance rows for the latest period.
        - Runs deterministic rules (balance equality, sign sanity, dupes).
        - Wraps the result in a Pydantic model.
        - Optionally consults Gemini for a controller-ready summary, with
          a deterministic fallback if the LLM is unavailable.
    """
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = validate_trial_balance(company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "trial_balance"
    return payload


@celery_app.task(name="agents.variance")
def run_variance_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 1 agent — Variance Analysis.

    Delegates to `run_variance_analysis()` in app/agents/variance.py.
    Compares actuals vs budget, flags accounts exceeding $50K or 10% variance.
    """
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_variance_analysis(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "variance"
    return payload


@celery_app.task(name="agents.cash_flow")
def run_cash_flow_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 1 agent — Cash Flow Reconciliation.

    Delegates to `run_cash_flow_reconciliation()` in app/agents/cash_flow.py.
    Reconciles GL cash movement against bank-statement movement and surfaces
    the top-N largest transactions for controller review.
    """
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_cash_flow_reconciliation(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "cash_flow"
    return payload


# ============================================================================
# PHASE 2 — Sequential chain per company
# ============================================================================
# Rationale for sequencing: revenue recognition depends on accruals being
# booked; expense categorization depends on revenue being recognized first.
# Running these in parallel would produce inconsistent intermediate state.
# ============================================================================

@celery_app.task(name="agents.accruals")
def run_accruals_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 2 Step 1 — Accrual Verification.

    Audits the accrual schedules table for staleness, orphans, and dupes.
    Must run before revenue recognition: unreconciled accruals distort the
    revenue base.
    """
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_accrual_verification(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "accruals"
    return payload


@celery_app.task(name="agents.revenue")
def run_revenue_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 2 Step 2 — Revenue Recognition (ASC 606).

    Validates allocation across performance obligations and detects stale
    milestones. Runs AFTER accruals so that deferred-revenue calculations
    use reconciled accrual figures.
    """
    if AGENT_STAGGER_SECONDS:
        time.sleep(AGENT_STAGGER_SECONDS)

    result = run_revenue_recognition(company_id=company_id, period=None)
    payload = result.model_dump()
    payload["run_id"] = run_id
    payload["agent"] = "revenue"
    return payload


@celery_app.task(name="agents.expenses")
def run_expenses_agent(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Phase 2 Step 3 — Expense Categorization.

    Audits expense accounts for miscategorization, unusual ratios, orphans.
    Last in the chain because it depends on final revenue classification.
    """
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

@celery_app.task(name="orchestrator.run_month_end_close")
def run_month_end_close(run_id: str | None = None) -> dict[str, Any]:
    """
    Fires the entire month-end close pipeline for one run_id.

    Called either by:
        - FastAPI's POST /api/v1/trigger-close
        - Celery Beat's daily 9 AM schedule

    Steps:
        1. Generate / accept a run_id.
        2. Initialize Redis state keys (status=running, counters=0).
        3. Fetch the list of all portfolio companies from Postgres.
        4. For each company, dispatch a Phase-1 chord with a staggered sleep
           between companies to respect Gemini's RPM budget.
    """
    run_id = run_id or str(uuid.uuid4())

    # ---- Initialize state ------------------------------------------------
    redis_client.set(f"close:{run_id}:status", "running", ex=PHASE_TIMEOUT)
    redis_client.set(f"close:{run_id}:phase2_count", 0, ex=PHASE_TIMEOUT)

    # ---- Fetch company list ----------------------------------------------
    db = SessionLocal()
    try:
        company_ids = [str(cid) for (cid, ) in db.query(Company.id).all()]
    finally:
        db.close()

    if not company_ids:
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "status": "failed", "reason": "no companies found"}

    redis_client.set(f"close:{run_id}:total_companies", len(company_ids), ex=PHASE_TIMEOUT)

    # Track this run for the UI's "recent runs" list. Capped at 50 entries.
    redis_client.lpush("close:runs:recent", run_id)
    redis_client.ltrim("close:runs:recent", 0, 49)

    # ---- Dispatch Phase 1, staggered -------------------------------------
    # The stagger exists ONLY to keep LLM calls under Gemini's RPM cap.
    # `idx > 0` skips the sleep for the very first company — no point
    # delaying the start of the pipeline.
    for idx, company_id in enumerate(company_ids):
        if idx > 0 and CHORD_STAGGER_SECONDS:
            time.sleep(CHORD_STAGGER_SECONDS)

        # chord(group(3 agents)) fires the 3 agents in parallel and hands
        # their combined results to phase1_complete once all three finish.
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

@celery_app.task(name="orchestrator.phase1_complete")
def phase1_complete(results: list[Any], run_id: str, company_id: str) -> dict[str, Any]:
    """
    Chord callback: fires when all three Phase-1 agents finish for one company.

    Responsibilities:
        1. Mark Phase 1 as done in Redis for this company.
        2. Inspect the three results for non-PASSED statuses and record them
           under `phase1_failures:{company_id}` (audit trail — Phase 2 still
           runs regardless, so the controller sees the full picture).
        3. Dispatch the Phase-2 sequential chain for this company.

    Args:
        results:  Injected by Celery — the list of return values from the
                  three agents in the chord's group.
        run_id:   The close run identifier.
        company_id: The company this callback belongs to.
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
        logger.warning(
            "Phase 1 for %s had non-PASSED status in: %s — proceeding to Phase 2.",
            company_id, phase1_failures,
        )

    # ---- Dispatch Phase 2 chain -----------------------------------------
    # `.si()` = immutable signature → each step does NOT receive the previous
    # step's return value as its first argument. We pass run_id/company_id
    # explicitly at each step.
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

@celery_app.task(name="orchestrator.phase2_complete")
def phase2_complete(run_id: str, company_id: str) -> dict[str, Any]:
    """
    Chain tail: fires after the three Phase-2 agents finish for one company.

    Uses an atomic Redis INCR counter to detect when ALL companies have
    finished Phase 2. The last company to complete triggers Phase 3, and a
    `SET NX` lock guarantees exactly one worker wins the race even if two
    companies finish within milliseconds of each other.
    """
    redis_client.set(f"close:{run_id}:phase2:{company_id}", "done", ex=PHASE_TIMEOUT)

    # INCR is atomic — no race between concurrent chain tails.
    count = int(redis_client.incr(f"close:{run_id}:phase2_count"))
    total = int(redis_client.get(f"close:{run_id}:total_companies") or 0)

    if count >= total:
        # SET NX with a shared key acts as a distributed mutex. Only one
        # worker's `locked` return value will be truthy.
        locked = redis_client.set(
            f"close:{run_id}:phase3_lock", "1", nx=True, ex=PHASE_TIMEOUT
        )
        if locked:
            run_cross_company_elimination.delay(run_id)

    return {"run_id": run_id, "company_id": company_id, "phase2_count": count, "total": total}


# ============================================================================
# PHASE 3 — Cross-Company Intercompany Elimination
# ============================================================================

@celery_app.task(name="orchestrator.run_cross_company_elimination")
def run_cross_company_elimination(run_id: str) -> dict[str, Any]:
    """
    Phase 3 — Cross-Company Intercompany Elimination.

    Fires exactly once per run, after every company completed Phase 2.

    Responsibilities:
        - Reconcile intercompany flows across the entire portfolio.
        - Flag mismatches (directional asymmetry, duplicate txns, orphans).
        - Persist the full EliminationResult JSON for Phase 4 + UI.
        - Hand off to Phase 4 (consolidation) — this task does NOT set the
          final 'completed' status; Phase 4 owns that.
    """
    logger.info("[Phase 3] Intercompany elimination started for run %s", run_id)

    try:
        result = run_elimination(period=None)
        payload = result.model_dump()
        payload["run_id"] = run_id
        payload["phase"] = 3

        # Persist full structured result for Phase 4 and the UI.
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

        # ---- Hand off to Phase 4 -----------------------------------------
        run_consolidation_task.delay(run_id)
        return payload

    except Exception as exc:  # noqa: BLE001
        logger.exception("[Phase 3] Elimination failed for run %s: %s", run_id, exc)
        redis_client.set(f"close:{run_id}:phase3", "failed", ex=PHASE_TIMEOUT)
        redis_client.set(f"close:{run_id}:status", "failed", ex=PHASE_TIMEOUT)
        return {"run_id": run_id, "phase": 3, "status": "failed", "error": str(exc)}


# ============================================================================
# PHASE 4 — Final Consolidation
# ============================================================================

@celery_app.task(name="orchestrator.run_consolidation_task")
def run_consolidation_task(run_id: str) -> dict[str, Any]:
    """
    Phase 4 — Final Consolidation.

    Rolls up all companies' trial balances into a group P&L, applies the
    Phase-3 intercompany asymmetry as a conservative EBITDA haircut, and
    marks the run 'completed'.

    Reads the Phase-3 asymmetry directly from Redis (`close:{run_id}:phase3:result`)
    so that the consolidation is deterministic and audit-traceable.
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

        # ---- Fire-and-forget reporting (failure here does not roll back) --
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


# ============================================================================
# PHASE 5 — Reporting & Communication
# ============================================================================

@celery_app.task(name="orchestrator.run_reporting_task")
def run_reporting_task(run_id: str) -> dict[str, Any]:
    """
    Phase 5 — Reporting & Communication.

    Sends the executive summary email to stakeholders. Runs AFTER the run is
    marked 'completed' — a failure here does NOT roll back the pipeline.

    Writes:
        close:{run_id}:reporting        → 'done' | 'failed'
        close:{run_id}:reporting_email  → recipient address (for audit)
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