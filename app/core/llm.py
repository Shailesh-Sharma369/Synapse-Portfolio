"""
Centralized LLM factory.

Reads `settings.llm_provider` and `settings.llm_model_id` and returns the
appropriate Agno model instance. Adding a new provider = one `if` branch.

Supported providers:
    - "gemini" (default) — Google Gemini via agno.models.google.Gemini
    - "claude"            — Anthropic Claude via agno.models.anthropic.Claude

Failures:
    - Missing API key for the selected provider → raise RuntimeError with a
      clear message so the calling agent can fall back to deterministic output.
    - Unknown provider → raise RuntimeError.
"""
from __future__ import annotations

import logging

from app.db.database import settings

logger = logging.getLogger(__name__)


def get_model():
    """
    Return an Agno model instance based on settings.llm_provider.

    Usage in an agent:
        from app.core.llm import get_model
        Agent(name="...", model=get_model(), tools=[...], ...)
    """
    provider = (settings.llm_provider or "gemini").strip().lower()
    model_id = (settings.llm_model_id or "").strip()

    if not model_id:
        raise RuntimeError(
            "llm_model_id is empty. Set LLM_MODEL_ID in .env "
            "(e.g., gemini-3.5-flash-lite or claude-3-5-sonnet-latest)."
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Cannot initialise Gemini model."
            )
        from agno.models.google import Gemini
        logger.debug("[LLM] Using Gemini model=%s", model_id)
        return Gemini(id=model_id, api_key=settings.gemini_api_key)

    if provider == "claude":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set but LLM_PROVIDER=claude. "
                "Set ANTHROPIC_API_KEY in .env or switch LLM_PROVIDER=gemini."
            )
        # Lazy import so Gemini-only deployments don't need the anthropic SDK.
        try:
            from agno.models.anthropic import Claude
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK not installed. Add `anthropic` to requirements.txt."
            ) from exc
        logger.debug("[LLM] Using Claude model=%s", model_id)
        return Claude(id=model_id, api_key=settings.anthropic_api_key)

    raise RuntimeError(
        f"Unknown LLM_PROVIDER='{provider}'. Supported: 'gemini', 'claude'."
    )