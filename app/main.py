# app/main.py (replace top imports + endpoint)

from __future__ import annotations

import uuid
from typing import Any

import redis
from fastapi import FastAPI
from sqlalchemy import text

from app.agents.orchestrator import run_month_end_close
from app.core.celery_app import celery_app   
from app.db.database import engine, settings

app = FastAPI(title="Month-End Close Orchestrator", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    postgres_ok = False
    redis_ok = False

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        postgres_ok = False

    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "healthy" if (postgres_ok and redis_ok) else "degraded",
        "postgres": postgres_ok,
        "redis": redis_ok,
    }


@app.post("/api/v1/trigger-close")
def trigger_close() -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    run_month_end_close.delay(run_id)
    # ✅ NEW: publish latest run pointer for the zero-click dashboard
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True)
        r.set("close:latest_run_id", run_id, ex=86400)  # 24h TTL
    except Exception:
        pass
    return {"run_id": run_id, "status": "queued"}