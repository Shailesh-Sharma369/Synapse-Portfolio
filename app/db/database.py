from __future__ import annotations

from typing import Generator

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    # ---- Gemini Settings ------------------------------------
    gemini_api_key: str = ""
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    database_url: str = "postgresql://admin:password123@localhost:5432/month_end_close"
    resend_api_key: str = ""
    from_email: str = "onboarding@resend.dev"
    to_email: str = "shaileshmsharma1@gmail.com"

    # ---- NEW: LLM provider configuration ---------------------------------
    # Swap between "gemini" and "claude" without changing code.
    llm_provider: str = "gemini" #or "claude"
    llm_model_id: str = "gemini-3.5-flash-lite"

    # Anthropic is optional. Only required if llm_provider == "claude".
    anthropic_api_key: str = ""

    # ---- NEW: Pacing / rate limits ---------------------------------------
    # Chord-level stagger: delay between dispatching each company's Phase-1
    # chord. Free tier: 15-20s. Paid tier: 0-2s.
    chord_stagger_seconds: int = 25

    # Agent-level stagger: delay inside each agent task before the LLM call.
    # Rarely needed since the rate limiter handles the RPM cap.
    agent_stagger_seconds: int = 0


settings = Settings()

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()