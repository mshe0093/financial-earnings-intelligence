"""Centralized configuration — all secrets parsed via Pydantic v2 BaseSettings.

Usage:
    from earnings_intel.config import get_settings
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value()

Security:
    - All API keys use ``SecretStr`` — automatically redacted in repr, logs,
      tracebacks, and serialized output.
    - SEC User-Agent format is validated against the EDGAR Fair Access pattern.
    - The ``.env`` file is loaded automatically but MUST be git-ignored.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded exclusively from environment / .env file.

    All fields map 1-to-1 with environment variable names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Secrets (SecretStr — redacted in repr, logs, tracebacks) ────────────
    gemini_api_key: SecretStr
    fred_api_key: SecretStr

    # ── SEC Fair Access ────────────────────────────────────────────────────
    sec_user_agent: str

    # ── Database ───────────────────────────────────────────────────────────
    duckdb_path: str = "data/earnings.duckdb"

    # ── Operational ────────────────────────────────────────────────────────
    log_level: str = "INFO"
    edgar_rate_limit_rps: int = 8

    @field_validator("sec_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, v: str) -> str:
        """Enforce SEC EDGAR Fair Access User-Agent format.

        Required format: ``"AppName contact@email.com"``
        See: https://www.sec.gov/os/webmaster-faq#code-support
        """
        pattern = r"^.+\s+\S+@\S+\.\S+$"
        if not re.match(pattern, v):
            raise ValueError(
                "sec_user_agent must match 'AppName user@domain.com' format "
                "per SEC EDGAR Fair Access policy."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a valid Python logging level name."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("edgar_rate_limit_rps")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """SEC allows ≤10 req/s; we enforce a conservative ceiling."""
        if not 1 <= v <= 10:
            raise ValueError(f"edgar_rate_limit_rps must be between 1 and 10 (SEC limit), got {v}")
        return v

    @property
    def resolved_duckdb_path(self) -> Path:
        """Return the DuckDB path resolved relative to the project root."""
        p = Path(self.duckdb_path)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Factory — returns a cached singleton.

    Uses ``lru_cache`` so the .env file is read at most once per process.
    """
    return Settings()  # type: ignore[call-arg]
