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

celery_app.conf.beat_schedule = {
    "month-end-close-daily-9am": {
        "task": "orchestrator.run_month_end_close",
        "schedule": crontab(hour=9, minute=0),
    },
}

celery_app.conf.task_routes = {
    "orchestrator.*": {"queue": "orchestrator"},
    "agents.*": {"queue": "agents"},
}