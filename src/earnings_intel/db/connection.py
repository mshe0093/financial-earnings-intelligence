"""DuckDB connection factory.

Provides a context-managed connection to the project's DuckDB database file.
The database file is created lazily — the parent directory is ensured to exist
on first access.

Usage:
    from earnings_intel.db.connection import get_connection

    with get_connection() as conn:
        conn.execute("SELECT * FROM filings_metadata LIMIT 5")

    # Or with explicit settings:
    from earnings_intel.config import get_settings
    with get_connection(get_settings()) as conn:
        ...
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path

import duckdb

from earnings_intel.config import Settings, get_settings


def _ensure_parent_dir(db_path: Path) -> None:
    """Create parent directories for the database file if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def get_connection(
    settings: Settings | None = None,
    *,
    read_only: bool = False,
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a DuckDB connection, closing it on exit.

    Args:
        settings: Application settings. Defaults to the cached singleton.
        read_only: Open the database in read-only mode (for concurrent readers).

    Yields:
        An open ``DuckDBPyConnection``.
    """
    if settings is None:
        settings = get_settings()

    db_path = settings.resolved_duckdb_path
    _ensure_parent_dir(db_path)

    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


@contextlib.contextmanager
def get_memory_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield an in-memory DuckDB connection — useful for tests.

    Yields:
        An open ``DuckDBPyConnection`` backed by ``:memory:``.
    """
    conn = duckdb.connect(":memory:")
    try:
        yield conn
    finally:
        conn.close()
