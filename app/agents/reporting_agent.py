"""
Reporting & Communication Agent (Agno) — Phase 5, decision layer.

The existing `reporting.py` handles HTML rendering and Resend API calls.
This agent sits on top and DECIDES:
    - Which email types to send (daily/weekly/completion/issue-alert)
    - Who should receive them
    - What priority
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field

from app.db.database import settings

logger = logging.getLogger(__name__)
_AGENT_DEBUG = os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")


def get_run_state(run_id: str) -> dict[str, Any]:
    """Fetch run state from Redis — used by the reporting agent to decide."""
    import redis
    r = redis.from_url(settings.redis_url, decode_responses=True)
    return {
        "run_id": run_id,
        "status": r.get(f"close:{run_id}:status") or "unknown",
        "phase3_status": r.get(f"close:{run_id}:phase3_status") or "not_run",
        "phase4_status": r.get(f"close:{run_id}:phase4_status") or "not_run",
        "has_escalation": r.get(f"close:{run_id}:escalation") is not None,
        "reporting_done": r.get(f"close:{run_id}:reporting") or "no",
    }


class EmailPlan(BaseModel):
    emails_to_send: list[str] = Field(
        ...,
        description="List from: 'completion', 'issue_alert', 'daily_summary', 'weekly_report'.",
    )
    priority: str = Field(..., description="'HIGH', 'MEDIUM', 'LOW'.")
    reasoning: str = Field(..., description="1-2 sentence rationale.")


def _build_agent() -> Agent:
    return Agent(
        name="Reporting Decision Agent",
        model=Gemini(id="gemini-3.1-flash-lite", api_key=settings.gemini_api_key),
        tools=[get_run_state],
        instructions=[
            "You decide which emails to send for a PE month-end close run.",
            "",
            "RULES:",
            "1. Call get_run_state(run_id) first.",
            "2. If status == 'completed' and no escalation:",
            "     emails_to_send = ['completion'], priority='LOW'",
            "3. If has_escalation OR phase3_status == 'MISMATCHES_FOUND':",
            "     emails_to_send = ['completion', 'issue_alert'], priority='HIGH'",
            "4. If status == 'running':",
            "     emails_to_send = ['daily_summary'], priority='MEDIUM'",
            "5. Only send 'weekly_report' on Mondays (SKIP unless explicitly requested).",
            "6. Never send duplicate emails — check reporting_done flag.",
            "",
            "Output ONLY the structured JSON schema.",
        ],
        output_schema=EmailPlan,
        markdown=False,
        use_json_mode=True,
        debug_mode=_AGENT_DEBUG,
        tool_call_limit=4,
    )


def plan_emails(run_id: str) -> EmailPlan:
    """Decide which emails to send based on run state."""
    if not settings.gemini_api_key:
        return EmailPlan(emails_to_send=["completion"], priority="MEDIUM",
                         reasoning="LLM unavailable; defaulting to completion email.")
    try:
        agent = _build_agent()
        prompt = f"Plan emails for run_id={run_id!r}. Call get_run_state first."
        response = agent.run(prompt)
        content = response.content
        if isinstance(content, dict):
            return EmailPlan(**content)
        if isinstance(content, str) and content.strip().startswith("{"):
            return EmailPlan(**json.loads(content))
        raise ValueError(f"Unexpected response: {content}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reporting planner failed (%s) — defaulting.", type(exc).__name__)
        return EmailPlan(emails_to_send=["completion"], priority="MEDIUM",
                         reasoning=f"LLM unavailable ({type(exc).__name__}).")