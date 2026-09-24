"""
Orchestrator Agent (Agno) — Phase 0 + Phase 6.

This is the "brain" that complements the Celery orchestrator. It makes two
kinds of decisions that the Celery graph can't:

    PRE-FLIGHT  — before dispatching a close, sanity-check the portfolio:
                  Are all companies seeded? Is any company already mid-close?
                  Should we defer this run?

    POST-FLIGHT — after Phase 4 completes, review the run status and decide:
                  Escalate to human? Send alert email? Mark as clean?

The Celery orchestrator remains the execution engine; this agent is the
decision layer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agno.agent import Agent
from app.core.llm import get_model  
from app.core.llm import get_model
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.database import SessionLocal, settings
from app.db.models import Company

logger = logging.getLogger(__name__)
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


# =============================================================================
# TOOLS — what the orchestrator agent can query
# =============================================================================

def list_portfolio_companies() -> list[dict[str, Any]]:
    """Return all companies in the portfolio."""
    db = SessionLocal()
    try:
        rows = db.scalars(select(Company).order_by(Company.id)).all()
        return [{"id": c.id, "name": c.name, "industry": c.industry} for c in rows]
    finally:
        db.close()


def get_run_status(run_id: str) -> dict[str, Any]:
    """Fetch the current close-run status from Redis."""
    import redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    return {
        "run_id": run_id,
        "status": r.get(f"close:{run_id}:status") or "unknown",
        "phase3_status": r.get(f"close:{run_id}:phase3_status") or "not_run",
        "phase4_status": r.get(f"close:{run_id}:phase4_status") or "not_run",
        "escalation": r.get(f"close:{run_id}:escalation"),
        "phase2_count": r.get(f"close:{run_id}:phase2_count") or "0",
        "total_companies": r.get(f"close:{run_id}:total_companies") or "0",
    }


def list_escalations(run_id: str) -> dict[str, Any]:
    """Return any escalation flags or phase-level failures."""
    import redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    out = {"failures": [], "escalation": r.get(f"close:{run_id}:escalation")}
    for key in r.scan_iter(match=f"close:{run_id}:phase1_failures:*"):
        company = key.rsplit(":", 1)[-1]
        out["failures"].append({"company_id": company, "agents": r.get(key)})
    return out


# =============================================================================
# SCHEMAS
# =============================================================================

class PreflightDecision(BaseModel):
    proceed: bool = Field(..., description="True if the run should proceed.")
    reason: str = Field(..., description="1-2 sentence reasoning.")
    companies_to_run: list[str] = Field(default_factory=list)


class PostflightDecision(BaseModel):
    status_summary: str = Field(..., description="1-2 sentence status.")
    escalate_to_human: bool = Field(..., description="True if human review required.")
    escalation_reason: str | None = None
    send_alert_email: bool = Field(False)


# =============================================================================
# AGENTS
# =============================================================================

def _build_preflight_agent() -> Agent:
    return Agent(
        name="Orchestrator Pre-Flight",
        model=get_model(),
        tools=[list_portfolio_companies, get_run_status],
        instructions=[
            "You are the Orchestrator Pre-Flight Agent for a PE month-end close system.",
            "Decide whether a close run should proceed based on portfolio + current state.",
            "",
            "RULES:",
            "1. Call list_portfolio_companies to see the portfolio.",
            "2. If a run_id is provided, call get_run_status to check state.",
            "3. proceed=True if the portfolio has companies and no run is already active.",
            "4. companies_to_run = all portfolio company IDs (unless some are excluded).",
            "",
            "Output ONLY the structured JSON schema.",
        ],
        output_schema=PreflightDecision,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=5,
    )


def _build_postflight_agent() -> Agent:
    return Agent(
        name="Orchestrator Post-Flight",
        model=get_model(),
        tools=[get_run_status, list_escalations],
        instructions=[
            "You are the Orchestrator Post-Flight Agent for a PE month-end close system.",
            "Review a completed (or failed) run and decide escalation.",
            "",
            "RULES:",
            "1. Call get_run_status(run_id) to see overall status.",
            "2. Call list_escalations(run_id) to see failures.",
            "3. escalate_to_human=True if: status != completed, OR any escalation exists,",
            "   OR phase3 has mismatches.",
            "4. send_alert_email=True if escalate_to_human=True.",
            "5. Write status_summary in 1-2 sentences.",
            "",
            "Output ONLY the structured JSON schema.",
        ],
        output_schema=PostflightDecision,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=5,
    )


def run_preflight(run_id: str | None = None) -> PreflightDecision:
    """Pre-flight check before dispatching a close run."""
    try:
        agent = _build_preflight_agent()
    except RuntimeError as exc:
        logger.warning("Pre-flight model init failed (%s) — proceeding.", exc)
        return PreflightDecision(proceed=True, reason=f"LLM unavailable ({exc}); proceeding.",
                                 companies_to_run=[])
    try:
        prompt = f"Pre-flight check. run_id={run_id!r}. Decide if the close should proceed."
        response = agent.run(prompt)
        content = response.content
        if isinstance(content, dict):
            return PreflightDecision(**content)
        if isinstance(content, str) and content.strip().startswith("{"):
            return PreflightDecision(**json.loads(content))
        raise ValueError(f"Unexpected response: {content}")
    except Exception as exc:
        logger.warning("Pre-flight LLM failed (%s) — defaulting to proceed.", type(exc).__name__)
        return PreflightDecision(proceed=True, reason="Pre-flight LLM unavailable; proceeding.",
                                 companies_to_run=[])

def run_postflight(run_id: str) -> PostflightDecision:
    try:
        agent = _build_postflight_agent()
    except RuntimeError as exc:
        logger.warning("Post-flight model init failed (%s).", exc)
        return PostflightDecision(status_summary=f"LLM unavailable ({exc}).",
                                  escalate_to_human=False)
    try:
        agent = _build_postflight_agent()
        prompt = f"Post-flight review for run_id={run_id!r}. Decide escalation."
        response = agent.run(prompt)
        content = response.content
        if isinstance(content, dict):
            return PostflightDecision(**content)
        if isinstance(content, str) and content.strip().startswith("{"):
            return PostflightDecision(**json.loads(content))
        raise ValueError(f"Unexpected response: {content}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Post-flight LLM failed (%s) — no escalation.", type(exc).__name__)
        return PostflightDecision(status_summary=f"LLM unavailable ({type(exc).__name__}).",
                                  escalate_to_human=False)