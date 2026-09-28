"""CIK normalization, ticker-to-CIK mapping, and SEC directory lookup.

Enforces zero-padded 10-digit CIK representation everywhere in the system.
Provides DuckDB caching for ticker→CIK mappings bootstrapped from the SEC's
``company_tickers.json`` endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb
import httpx

from earnings_intel.edgar.models import CompanyInfo

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def normalize_cik(raw_cik: str | int) -> str:
    """Normalize CIK to a zero-padded 10-digit string.

    Args:
        raw_cik: CIK as an integer, unpadded string, or whitespace-padded string.

    Returns:
        A 10-digit string with leading zeros.

    Examples:
        >>> normalize_cik(320193)
        '0000320193'
        >>> normalize_cik(" 320193 ")
        '0000320193'
        >>> normalize_cik("0000320193")
        '0000320193'
    """
    return str(raw_cik).strip().zfill(10)


async def fetch_company_tickers(
    client: httpx.AsyncClient,
) -> dict[str, CompanyInfo]:
    """Download and parse the SEC's official company_tickers.json.

    The SEC endpoint returns a JSON object of numbered objects:
    ``{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}``

    Args:
        client: An authenticated/configured ``httpx.AsyncClient`` with User-Agent set.

    Returns:
        Mapping of ``{UPPERCASE_TICKER: CompanyInfo}``.
    """
    response = await client.get(SEC_TICKERS_URL)
    response.raise_for_status()
    raw_data: dict[str, dict[str, Any]] = response.json()

    mapping: dict[str, CompanyInfo] = {}
    for entry in raw_data.values():
        ticker = str(entry["ticker"]).strip().upper()
        mapping[ticker] = CompanyInfo(
            cik=normalize_cik(entry["cik_str"]),
            ticker=ticker,
            title=str(entry["title"]).strip(),
        )

    logger.info("Loaded %d ticker-to-CIK mappings from SEC", len(mapping))
    return mapping


def lookup_cik(ticker: str, mapping: dict[str, CompanyInfo]) -> str | None:
    """Look up a normalized CIK by ticker symbol (case-insensitive).

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "aapl").
        mapping: Dictionary from ``fetch_company_tickers`` or DB cache.

    Returns:
        Zero-padded 10-digit CIK string, or ``None`` if not found.
    """
    info = mapping.get(ticker.strip().upper())
    return info.cik if info else None


def store_ticker_mapping(
    conn: duckdb.DuckDBPyConnection,
    mapping: dict[str, CompanyInfo],
) -> int:
    """Persist ticker→CIK mapping into DuckDB for fast offline lookup.

    Creates the ``ticker_cik_mapping`` table if it does not exist.
    Uses ``INSERT OR REPLACE`` for idempotent upserts.

    Args:
        conn: An open DuckDB connection.
        mapping: Dictionary of ``{ticker: CompanyInfo}``.

    Returns:
        Number of rows inserted/updated.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_cik_mapping (
            ticker       VARCHAR PRIMARY KEY,
            cik          VARCHAR(10) NOT NULL,
            company_name VARCHAR NOT NULL,
            updated_at   TIMESTAMP DEFAULT current_timestamp
        );
    """)

    rows = [(info.ticker, info.cik, info.title) for info in mapping.values()]
    conn.executemany(
        """
        INSERT OR REPLACE INTO ticker_cik_mapping (ticker, cik, company_name)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    logger.info("Stored %d ticker mappings in DuckDB", len(rows))
    return len(rows)


def load_ticker_mapping_from_db(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, CompanyInfo]:
    """Load previously stored ticker→CIK mapping from DuckDB.

    Returns:
        Mapping of ``{UPPERCASE_TICKER: CompanyInfo}``, or empty dict if unpopulated.
    """
    try:
        rows = conn.execute("SELECT ticker, cik, company_name FROM ticker_cik_mapping").fetchall()
        return {row[0]: CompanyInfo(ticker=row[0], cik=row[1], title=row[2]) for row in rows}
    except duckdb.CatalogException:
        return {}
