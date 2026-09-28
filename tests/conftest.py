"""Shared test fixtures for the earnings-intel test suite.

Provides:
    - ``mock_env``: Patches environment variables with safe test values.
    - ``test_settings``: A ``Settings`` instance loaded from mock env vars.
    - ``memory_db``: An in-memory DuckDB connection with the full schema applied.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import duckdb
import pytest

from earnings_intel.config import Settings
from earnings_intel.db.connection import get_memory_connection
from earnings_intel.db.schema import initialize_schema

# ── Safe test values — NEVER real credentials ──────────────────────────────

_TEST_ENV = {
    "GEMINI_API_KEY": "test-gemini-key-not-real",
    "FRED_API_KEY": "test-fred-key-not-real",
    "SEC_USER_AGENT": "TestApp test@example.com",
    "DUCKDB_PATH": ":memory:",
    "LOG_LEVEL": "DEBUG",
    "EDGAR_RATE_LIMIT_RPS": "8",
}


@pytest.fixture()
def mock_env() -> Generator[dict[str, str], None, None]:
    """Patch ``os.environ`` with safe test values for the duration of a test."""
    with patch.dict("os.environ", _TEST_ENV, clear=False):
        yield _TEST_ENV


@pytest.fixture()
def test_settings(mock_env: dict[str, str]) -> Settings:
    """Return a ``Settings`` instance loaded from mocked env vars.

    Bypasses the ``lru_cache`` on ``get_settings()`` by constructing directly.
    """
    return Settings()  # type: ignore[call-arg]


@pytest.fixture()
def memory_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield an in-memory DuckDB connection with all tables created."""
    with get_memory_connection() as conn:
        initialize_schema(conn)
        yield conn
