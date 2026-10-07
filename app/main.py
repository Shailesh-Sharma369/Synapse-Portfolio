import os
os.environ.setdefault("AGNO_TELEMETRY", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("LITELLM_TELEMETRY", "false")

import uuid
from typing import Any

import redis
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from app.agents.orchestrator import run_month_end_close
from app.core.celery_app import celery_app   
from app.db.database import engine, settings

app = FastAPI(title="Month-End Close Orchestrator", version="0.1.0")
from app.core.rate_limit import install_rate_limiter
install_rate_limiter()


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
        "bind": "127.0.0.1 (host-loopback only)",
        "telemetry": "disabled",
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


@app.get("/api/v1/privacy/audit/{run_id}")
def privacy_audit(run_id: str) -> dict[str, Any]:
    try:
        from app.core.privacy import describe_for_audit

        return describe_for_audit(run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc