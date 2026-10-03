from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

from app.db.database import settings

celery_app = Celery(
    "month_end_close",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.agents.orchestrator"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
)


# =============================================================================
# AUTONOMY TOGGLE
# =============================================================================
# ENABLE_AUTONOMOUS_SCHEDULE=0  → Beat does NOT auto-fire the close. Manual
#                                  trigger via POST /api/v1/trigger-close only.
#                                  (Default; useful for testing and demos.)
# ENABLE_AUTONOMOUS_SCHEDULE=1  → Beat fires:
#                                    - a full close daily at 9:00 AM UTC
#                                    - a formal month-end close on the 1st
#                                      of every month at 9:30 AM UTC
#                                  Notifications always fire regardless.
#
# Why daily + monthly?
#   - Daily close: continuous-close pattern; validates the pipeline every day.
#   - Month-end: the formal close that produces the LP-facing package.
#   Both are idempotent — the UI tracks each run by its own run_id.
# =============================================================================



# =============================================================================
# BEAT SCHEDULE
# =============================================================================
_beat_schedule: dict = {
    # ---- Daily summary email, 8:00 AM UTC --------------------------------
    # Always active. This is the "Daily Summary" 
    # a morning progress update to stakeholders.
    "daily-summary-8am": {
        "task": "orchestrator.send_daily_summary",
        "schedule": crontab(hour=8, minute=0),
    },

    # ---- Weekly stakeholder report, Monday 8:00 AM UTC -------------------
    # Always active. This is the "Stakeholder Report" 
    "weekly-report-monday-8am": {
        "task": "orchestrator.send_weekly_report",
        "schedule": crontab(day_of_week=1, hour=8, minute=0),
    },

    # ---- Issue alert sweep, daily at 12:00 PM UTC ------------------------
    # Replaces the previous hourly sweep. Fires once mid-day; the task itself
    # only sends an alert if the current run has an active issue (phase-3
    # mismatch or escalation flag). No-op otherwise.
    "issue-alert-daily-noon": {
        "task": "orchestrator.send_issue_alert",
        "schedule": crontab(hour=12, minute=0),
    },
}
_beat_enabled = os.getenv("BEAT_ENABLED", "0").strip() == "1"
_enable_autonomous = _beat_enabled and os.getenv("ENABLE_AUTONOMOUS_SCHEDULE", "0").strip() == "1"

if _enable_autonomous:
    # ---- Full close, daily at 9:00 AM UTC --------------------------------
    # Continuous-close cadence: one full validation→consolidation cycle per day.
    _beat_schedule["month-end-close-daily-9am"] = {
        "task": "orchestrator.run_month_end_close",
        "schedule": crontab(hour=9, minute=0),
    }

    # ---- Formal month-end close, 1st of month at 9:30 AM UTC -------------
    # The stakeholder-facing close that produces the LP package + completion
    # email. Runs slightly after the daily so the two don't collide.
    _beat_schedule["month-end-close-day-1"] = {
        "task": "orchestrator.run_month_end_close",
        "schedule": crontab(day_of_month=1, hour=9, minute=30),
    }


celery_app.conf.beat_schedule = _beat_schedule
if not _beat_enabled:
    # Hard-disable: no task, scheduled or otherwise, will fire from Beat.
    celery_app.conf.beat_schedule = {}
    import logging as _logging
    _logging.getLogger(__name__).info(
        "[Beat] DISABLED (BEAT_ENABLED=0). Trigger manually via POST /api/v1/trigger-close."
    )

# =============================================================================
# TASK ROUTES
# =============================================================================
celery_app.conf.task_routes = {
    "orchestrator.*": {"queue": "orchestrator"},
    "agents.*": {"queue": "agents"},
}

# =============================================================================
# GLOBAL LLM RATE LIMITER — installed once per worker process
# =============================================================================
# Enforces GEMINI_MAX_RPM (default 12) calls / 60s ACROSS all workers via a
# shared Redis counter. Prevents 429 quota errors and gives predictable
# pipeline latency.
from app.core.rate_limit import install_rate_limiter
install_rate_limiter()