from __future__ import annotations

import inspect
import json
import logging
import re
import threading
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MASKER_CACHE: dict[str, DataMasker] = {}
_MASKER_LOCK = threading.Lock()


class DataMasker:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self._map_key = f"close:{run_id}:privacy:map"
        self._lock_key = f"close:{run_id}:privacy:lock"
        self._real_to_token: dict[str, str] = {}
        self._token_to_real: dict[str, str] = {}
        self._token_to_display: dict[str, str] = {}
        self._mask_regex: re.Pattern[str] | None = None
        self._rehydrate_regex: re.Pattern[str] | None = None
        self._redis = None
        self._redis_ok = False

        redis_url = self._get_redis_url()
        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                self._redis_ok = True
                self._load_from_redis()
            except Exception as exc:
                logger.warning(
                    "Privacy masker Redis unavailable; using in-memory map: %s", exc
                )
        else:
            logger.warning(
                "Privacy masker: settings/Redis URL unavailable; using in-memory map."
            )

        if self._redis_ok and self._redis is not None:
            try:
                with self._redis.lock(self._lock_key, timeout=10, blocking_timeout=10):
                    self._load_from_redis()
                    self._prime_from_portfolio()
            except Exception as exc:
                logger.warning(
                    "Privacy masker lock failed; priming without lock: %s", exc
                )
                self._prime_from_portfolio()
        else:
            self._prime_from_portfolio()

    # ------------------------------------------------------------------
    # Lazy DB access — import only when needed, never at module import.
    # This lets the module load on a host without a live Postgres.
    # ------------------------------------------------------------------
    @staticmethod
    def _get_redis_url() -> str | None:
        try:
            from app.db.database import settings

            return getattr(settings, "redis_url", None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Privacy] settings import failed (%s) — no Redis.",
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _alpha_token(idx: int) -> str:
        letters = ""
        while idx > 0:
            idx, remainder = divmod(idx - 1, 26)
            letters = chr(ord("A") + remainder) + letters
        return letters

    def _load_from_redis(self) -> None:
        if not self._redis_ok or self._redis is None:
            return
        for real, raw in self._redis.hgetall(self._map_key).items():
            try:
                data = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                data = {"token": raw}
            token = data["token"]
            self._real_to_token[real] = token
            self._token_to_real.setdefault(token, real)
            if data.get("display"):
                self._token_to_display[token] = data["display"]

    def _persist_to_redis(self) -> None:
        if not self._redis_ok or self._redis is None:
            return
        try:
            mapping = {
                real: json.dumps({
                    "token": token,
                    "display": self._token_to_display.get(token),
                })
                for real, token in self._real_to_token.items()
            }
            if mapping:
                self._redis.hset(self._map_key, mapping=mapping)
            self._redis.expire(self._map_key, 3600)
        except Exception as exc:
            self._redis_ok = False
            logger.warning(
                "Privacy masker Redis persistence failed; using in-memory map: %s", exc
            )

    def _assign(self, real: str, token: str, display: str | None = None) -> str:
        self._real_to_token[real] = token
        self._token_to_real.setdefault(token, real)
        if display is not None:
            self._token_to_display[token] = display
        self._mask_regex = None
        self._rehydrate_regex = None
        self._persist_to_redis()
        return token

    def _prime_from_portfolio(self) -> None:
        # Lazy import — this is where psycopg2 gets loaded on hosts that
        # can load it. On hosts where the DLL is blocked (corporate WDAC /
        # AppLocker), we log and continue with an empty map. Tokens will
        # be assigned on demand via ``token_for``.
        try:
            from app.db.database import SessionLocal
            from app.db.models import Company, RevenueContract
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Privacy] DB layer unavailable (%s: %s). "
                "Portfolio priming skipped — tokens will be assigned lazily.",
                type(exc).__name__,
                exc,
            )
            return

        try:
            db = SessionLocal()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Privacy] DB session failed: %s", exc)
            return

        try:
            from sqlalchemy import select

            for company_id, company_name in db.execute(
                select(Company.id, Company.name)
            ).all():
                token = self.token_for(str(company_id))
                self._assign(str(company_name), token, str(company_name))
            for customer in db.scalars(
                select(RevenueContract.customer).distinct()
            ).all():
                self.token_for(str(customer))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Privacy] Portfolio priming query failed: %s", exc)
        finally:
            db.close()

    def token_for(self, real: str) -> str:
        existing = self._real_to_token.get(real)
        if existing is not None:
            return existing
        if re.fullmatch(r"[a-z0-9_]+", real):
            idx = 1
            while f"Entity_{self._alpha_token(idx)}" in self._token_to_real:
                idx += 1
            return self._assign(real, f"Entity_{self._alpha_token(idx)}")
        idx = 1
        while f"Customer_{idx}" in self._token_to_real:
            idx += 1
        return self._assign(real, f"Customer_{idx}")

    # ------------------------------------------------------------------
    # MASKING (real -> token)
    # ------------------------------------------------------------------
    def mask_text(self, text: str) -> str:
        if self._mask_regex is None and self._real_to_token:
            keys = sorted(self._real_to_token, key=len, reverse=True)
            self._mask_regex = re.compile("|".join(re.escape(key) for key in keys))
        if self._mask_regex is None:
            return text
        return self._mask_regex.sub(
            lambda match: self._real_to_token[match.group(0)], text
        )

    def mask_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.mask_text(value)
        if isinstance(value, dict):
            return {key: self.mask_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.mask_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.mask_value(item) for item in value)
        return value

    # ------------------------------------------------------------------
    # REHYDRATION — display-name variant (for the LLM's OUTPUT)
    # ------------------------------------------------------------------
    def rehydrate_text(self, text: str) -> str:
        if self._rehydrate_regex is None and self._token_to_real:
            keys = sorted(self._token_to_real, key=len, reverse=True)
            self._rehydrate_regex = re.compile("|".join(re.escape(key) for key in keys))
        if self._rehydrate_regex is None:
            return text
        return self._rehydrate_regex.sub(
            lambda match: self._token_to_display.get(
                match.group(0), self._token_to_real[match.group(0)]
            ),
            text,
        )

    def rehydrate_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.rehydrate_text(value)
        if isinstance(value, dict):
            return {key: self.rehydrate_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.rehydrate_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.rehydrate_value(item) for item in value)
        return value

    # ------------------------------------------------------------------
    # REHYDRATION — ID variant (for TOOL ARGUMENTS)
    # Tools query by company_id slug, not by human name.
    # ------------------------------------------------------------------
    def rehydrate_arg_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        if self._rehydrate_regex is None and self._token_to_real:
            keys = sorted(self._token_to_real, key=len, reverse=True)
            self._rehydrate_regex = re.compile("|".join(re.escape(key) for key in keys))
        if self._rehydrate_regex is None:
            return text
        return self._rehydrate_regex.sub(
            lambda match: self._token_to_real[match.group(0)], text
        )

    def rehydrate_arg_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.rehydrate_arg_text(value)
        if isinstance(value, dict):
            return {key: self.rehydrate_arg_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.rehydrate_arg_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.rehydrate_arg_value(item) for item in value)
        return value

    # ------------------------------------------------------------------
    # TOOL WRAPPING
    # ------------------------------------------------------------------
    def wrap_tool(self, fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(
                *self.rehydrate_arg_value(args),
                **self.rehydrate_arg_value(kwargs),
            )
            return self.mask_value(result)

        try:
            wrapper.__signature__ = inspect.signature(fn)
        except (TypeError, ValueError):
            pass
        return wrapper

    def wrap_tools(self, tools: list[Callable]) -> list[Callable]:
        return [self.wrap_tool(tool) for tool in tools]


def get_masker(run_id: str) -> DataMasker:
    with _MASKER_LOCK:
        if run_id not in _MASKER_CACHE:
            _MASKER_CACHE[run_id] = DataMasker(run_id)
        return _MASKER_CACHE[run_id]


def drop_masker(run_id: str) -> None:
    with _MASKER_LOCK:
        _MASKER_CACHE.pop(run_id, None)


def describe_for_audit(run_id: str) -> dict:
    masker = get_masker(run_id)
    return {
        "run_id": run_id,
        "tokens_issued": len(masker._token_to_real),
        "redis_backed": bool(masker._redis_ok),
        "map_key": str(masker._map_key),
    }