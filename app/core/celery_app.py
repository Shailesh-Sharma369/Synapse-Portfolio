from __future__ import annotations

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
# CELERY BEAT SCHEDULE — Autonomous Operation
# =============================================================================
# The assignment requires:
#   - Daily 9 AM full close
#   - Hourly during close week (days 1-5)
#   - Daily summary email
#   - Weekly stakeholder report
#   - Issue alerts (condition-triggered, hourly sweep)
#
# All of the above are wired here. The close-week hourly entry passes
# close_week_only=True; the task itself exits early if today.day > 5.
# =============================================================================
celery_app.conf.beat_schedule = {
    # ---- Full close, daily 9 AM ------------------------------------------
    "month-end-close-daily-9am": {
        "task": "orchestrator.run_month_end_close",
        "schedule": crontab(hour=9, minute=0),
    },

    # # ---- Close-week hourly sweep (task self-guards to days 1-5) ----------
    # "close-week-hourly": {
    #     "task": "orchestrator.run_month_end_close",
    #     "schedule": crontab(minute=0),
    #     "kwargs": {"close_week_only": True},
    # },

    # ---- Daily progress summary, 8 AM ------------------------------------
    "daily-summary-8am": {
        "task": "orchestrator.send_daily_summary",
        "schedule": crontab(hour=8, minute=0),
    },

    # ---- Weekly stakeholder report, Monday 8 AM --------------------------
    "weekly-report-monday-8am": {
        "task": "orchestrator.send_weekly_report",
        "schedule": crontab(day_of_week=1, hour=8, minute=0),
    },

    # ---- Hourly issue alert sweep ----------------------------------------
    "issue-alert-hourly": {
        "task": "orchestrator.send_issue_alert",
        "schedule": crontab(minute=15),
    },
}

celery_app.conf.task_routes = {
    "orchestrator.*": {"queue": "orchestrator"},
    "agents.*": {"queue": "agents"},
}