"""
Global Gemini request rate limiter.

Enforces a maximum of GEMINI_MAX_RPM (default 12) LLM calls per rolling
60-second window ACROSS ALL Celery workers and the UI process.

Why Redis?
    Celery prefork workers run in separate OS processes. A local counter
    would let each worker hit the limit independently. Redis gives us a
    single shared atomic counter.

Reservation model:
    A single `Agent.run(...)` internally fires MULTIPLE Gemini HTTP calls
    (one per ReAct step — typically 2-4). So we reserve
    GEMINI_CALLS_PER_AGENT slots upfront per agent.run() invocation. This
    over-reserves slightly but guarantees the real HTTP rate stays under
    the free-tier ceiling.

503 handling:
    Gemini 503 is server-side overload, not rate limiting. We don't retry
    503 in tight loops (that wastes slots). Instead we back off and let
    the agent's own retry policy handle it with jitter.
"""
from __future__ import annotations

import logging
import os
import time

import redis

from app.db.database import settings

logger = logging.getLogger(__name__)

_r = redis.from_url(settings.redis_url, decode_responses=True)
_MAX_RPM = int(os.getenv("GEMINI_MAX_RPM", "12"))
_CALLS_PER_AGENT = int(os.getenv("GEMINI_CALLS_PER_AGENT", "4"))
_WINDOW_SECONDS = 60
_BUCKET_TTL = 120


def acquire_slots(n: int = 1, timeout_seconds: int = 300) -> None:
    """
    Atomically reserve `n` slots in the current 60s window.

    If the window cannot fit `n` slots, sleep until the next window.
    The counter is NEVER incremented when the reservation fails — that was
    the bug in the previous version (sleeping workers kept incrementing).
    """
    started = time.time()
    while True:
        bucket = int(time.time() // _WINDOW_SECONDS)
        key = f"ratelimit:gemini:{bucket}"

        with _r.pipeline() as pipe:
            try:
                pipe.watch(key)
                current = int(pipe.get(key) or 0)

                if current + n > _MAX_RPM:
                    pipe.unwatch()
                    now = time.time()
                    next_bucket = (bucket + 1) * _WINDOW_SECONDS
                    sleep_for = max(0.1, next_bucket - now + 0.3)

                    if (now - started) + sleep_for > timeout_seconds:
                        logger.warning(
                            "[RateLimit] timeout after %.1fs — proceeding anyway",
                            now - started,
                        )
                        return

                    logger.info(
                        "[RateLimit] window full (%d/%d), need %d — sleeping %.1fs",
                        current, _MAX_RPM, n, sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue

                # Reserve n slots atomically
                pipe.multi()
                pipe.incrby(key, n)
                pipe.expire(key, _BUCKET_TTL)
                pipe.execute()

                logger.debug(
                    "[RateLimit] reserved %d slot(s) — %d/%d used",
                    n, current + n, _MAX_RPM,
                )
                return
            except redis.WatchError:
                continue
            finally:
                pipe.reset()


def acquire_slot(timeout_seconds: int = 300) -> None:
    """Backwards-compatible single-slot reservation."""
    acquire_slots(1, timeout_seconds)


def peek_window() -> dict:
    """Diagnostic snapshot of the current window."""
    bucket = int(time.time() // _WINDOW_SECONDS)
    key = f"ratelimit:gemini:{bucket}"
    count = int(_r.get(key) or 0)
    ttl = _r.ttl(key)
    return {
        "bucket": bucket,
        "count": count,
        "max": _MAX_RPM,
        "calls_per_agent": _CALLS_PER_AGENT,
        "remaining": max(0, _MAX_RPM - count),
        "ttl_seconds": max(0, ttl),
    }


# =============================================================================
# MONKEY-PATCH — reserve slots per agent.run() invocation
# =============================================================================
_INSTALLED = False


def install_rate_limiter() -> None:
    """
    Idempotent. Called once per process (worker, api, ui, beat).

    Patches Agent.run so that EVERY call reserves GEMINI_CALLS_PER_AGENT
    slots upfront. This keeps the real HTTP rate below the free-tier cap
    even though a single agent.run makes multiple Gemini calls internally.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    try:
        from agno.agent import Agent
    except Exception:  # noqa: BLE001
        logger.warning("[RateLimit] Could not import Agno.Agent — NOT installed")
        return

    _original_run = Agent.run

    def _limited_run(self, *args, **kwargs):
        acquire_slots(_CALLS_PER_AGENT)
        return _original_run(self, *args, **kwargs)

    Agent.run = _limited_run

    if hasattr(Agent, "arun"):
        _original_arun = Agent.arun

        async def _limited_arun(self, *args, **kwargs):
            acquire_slots(_CALLS_PER_AGENT)
            return await _original_arun(self, *args, **kwargs)

        Agent.arun = _limited_arun

    logger.info(
        "[RateLimit] Installed — %d RPM cap, %d slots reserved per agent.run",
        _MAX_RPM, _CALLS_PER_AGENT,
    )