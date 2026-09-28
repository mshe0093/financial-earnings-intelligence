"""DuckDB schema definitions and migrations.

All analytical data lives in a single DuckDB file.  This module owns the DDL
and exposes two public entry-points:

* ``initialize_schema(conn)`` — creates all tables and indexes (idempotent).
* ``SCHEMA_VERSION``          — bumped whenever the DDL changes.

Tables:
    filings_metadata   — raw SEC EDGAR filing metadata
    extracted_signals   — 12 structured metrics from Gemini extraction
    price_series        — historical daily OHLCV prices
    macro_indicators    — FRED macroeconomic time-series
"""

from __future__ import annotations

import duckdb

SCHEMA_VERSION: int = 1

# ── DDL Statements ─────────────────────────────────────────────────────────

_DDL_FILINGS_METADATA = """
CREATE TABLE IF NOT EXISTS filings_metadata (
    filing_id        VARCHAR PRIMARY KEY,   -- '{cik}_{accession_number}'
    cik              VARCHAR(10) NOT NULL,  -- Zero-padded 10-digit CIK
    ticker           VARCHAR(10),
    company_name     VARCHAR NOT NULL,
    form_type        VARCHAR(10) NOT NULL,  -- '10-K' or '10-Q'
    filing_date      DATE NOT NULL,
    period_of_report DATE NOT NULL,
    accession_number VARCHAR NOT NULL,
    primary_doc_url  VARCHAR NOT NULL,
    mda_text_hash    VARCHAR,               -- SHA-256 of extracted MD&A text
    ingested_at      TIMESTAMP DEFAULT current_timestamp,

    -- Constraints
    CONSTRAINT valid_form CHECK (form_type IN ('10-K', '10-Q')),
    CONSTRAINT valid_cik  CHECK (length(cik) = 10)
);
"""

_DDL_FILINGS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_filings_ticker_date
    ON filings_metadata (ticker, filing_date);

CREATE INDEX IF NOT EXISTS idx_filings_cik
    ON filings_metadata (cik);
"""

_DDL_EXTRACTED_SIGNALS = """
CREATE TABLE IF NOT EXISTS extracted_signals (
    signal_id                VARCHAR PRIMARY KEY,   -- '{filing_id}_v{version}'
    filing_id                VARCHAR NOT NULL REFERENCES filings_metadata(filing_id),
    extraction_model         VARCHAR NOT NULL,       -- e.g. 'gemini-2.5-flash'
    extraction_timestamp     TIMESTAMP DEFAULT current_timestamp,

    -- 12 Structured Metrics
    revenue_guidance         VARCHAR,      -- 'raise' | 'maintain' | 'lower' | 'none'
    revenue_guidance_detail  VARCHAR,      -- Free-text elaboration
    eps_guidance             VARCHAR,      -- 'raise' | 'maintain' | 'lower' | 'none'
    eps_guidance_detail      VARCHAR,
    margin_outlook           VARCHAR,      -- 'expanding' | 'stable' | 'contracting' | 'none'
    capex_direction          VARCHAR,      -- 'increasing' | 'stable' | 'decreasing' | 'none'
    management_sentiment     FLOAT,        -- [-1.0, 1.0] composite score
    forward_language_ratio   FLOAT,        -- Ratio of forward-looking to total sentences
    risk_factor_count        INTEGER,      -- Number of distinct risk factors mentioned
    key_risk_topics          VARCHAR[],    -- Array of top risk themes
    guidance_confidence      FLOAT,        -- [0.0, 1.0] model confidence
    restructuring_signals    BOOLEAN,      -- Restructuring/layoffs/impairment mentioned

    -- Provenance
    mda_char_count           INTEGER,
    prompt_tokens_used       INTEGER,
    completion_tokens_used   INTEGER
);
"""

_DDL_SIGNALS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_signals_filing
    ON extracted_signals (filing_id);
"""

_DDL_PRICE_SERIES = """
CREATE TABLE IF NOT EXISTS price_series (
    ticker       VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open_price   DOUBLE,
    high_price   DOUBLE,
    low_price    DOUBLE,
    close_price  DOUBLE NOT NULL,
    adj_close    DOUBLE NOT NULL,
    volume       BIGINT,
    source       VARCHAR DEFAULT 'yfinance',
    fetched_at   TIMESTAMP DEFAULT current_timestamp,

    PRIMARY KEY (ticker, trade_date)
);
"""

_DDL_PRICE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_price_ticker_date
    ON price_series (ticker, trade_date);
"""

_DDL_MACRO_INDICATORS = """
CREATE TABLE IF NOT EXISTS macro_indicators (
    series_id    VARCHAR NOT NULL,     -- e.g. 'DFF', 'T10Y2Y', 'VIXCLS'
    obs_date     DATE NOT NULL,
    value        DOUBLE,
    fetched_at   TIMESTAMP DEFAULT current_timestamp,

    PRIMARY KEY (series_id, obs_date)
);
"""

_DDL_BACKTEST_RUNS = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id                VARCHAR PRIMARY KEY,
    run_timestamp         TIMESTAMP DEFAULT current_timestamp,
    benchmark             VARCHAR NOT NULL,
    window_days           INTEGER NOT NULL,
    total_trades          INTEGER NOT NULL,
    winning_trades        INTEGER NOT NULL,
    losing_trades         INTEGER NOT NULL,
    hit_rate              FLOAT NOT NULL,
    mean_return           FLOAT NOT NULL,
    mean_abnormal_return  FLOAT NOT NULL,
    annualized_return     FLOAT NOT NULL,
    annualized_volatility FLOAT NOT NULL,
    sharpe_ratio          FLOAT NOT NULL,
    max_drawdown          FLOAT NOT NULL,
    calmar_ratio          FLOAT NOT NULL
);
"""

_DDL_BACKTEST_TRADES = """
CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id          VARCHAR PRIMARY KEY,
    run_id            VARCHAR NOT NULL REFERENCES backtest_runs(run_id),
    filing_id         VARCHAR NOT NULL REFERENCES filings_metadata(filing_id),
    ticker            VARCHAR NOT NULL,
    filing_date       DATE NOT NULL,
    signal            VARCHAR NOT NULL,
    signal_reason     VARCHAR,
    entry_date        DATE NOT NULL,
    exit_date         DATE NOT NULL,
    holding_days      INTEGER NOT NULL,
    stock_return      FLOAT NOT NULL,
    benchmark_return  FLOAT NOT NULL,
    strategy_return   FLOAT NOT NULL,
    abnormal_return   FLOAT NOT NULL
);
"""

_DDL_BACKTEST_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_backtest_trades_run
    ON backtest_trades (run_id);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_ticker
    ON backtest_trades (ticker);
"""

# Schema version tracking table
_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS _schema_version (
    version      INTEGER NOT NULL,
    applied_at   TIMESTAMP DEFAULT current_timestamp
);
"""

# Ordered list — dependencies first (filings before signals and backtest trades).
_ALL_DDL: list[str] = [
    _DDL_SCHEMA_VERSION,
    _DDL_FILINGS_METADATA,
    _DDL_FILINGS_INDEXES,
    _DDL_EXTRACTED_SIGNALS,
    _DDL_SIGNALS_INDEXES,
    _DDL_PRICE_SERIES,
    _DDL_PRICE_INDEXES,
    _DDL_MACRO_INDICATORS,
    _DDL_BACKTEST_RUNS,
    _DDL_BACKTEST_TRADES,
    _DDL_BACKTEST_INDEXES,
]

_EXPECTED_TABLES: set[str] = {
    "filings_metadata",
    "extracted_signals",
    "price_series",
    "macro_indicators",
    "backtest_runs",
    "backtest_trades",
    "_schema_version",
}


# ── Public API ─────────────────────────────────────────────────────────────


def initialize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables and indexes (idempotent).

    Executes each DDL statement inside a transaction.  Safe to call
    on every application startup — ``CREATE ... IF NOT EXISTS`` ensures
    no-ops on subsequent runs.

    Args:
        conn: An open DuckDB connection.
    """
    conn.begin()
    try:
        for ddl in _ALL_DDL:
            conn.execute(ddl)

        # Record schema version if not already present.
        existing = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        if existing is None or existing[0] is None or existing[0] < SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO _schema_version (version) VALUES (?)",
                [SCHEMA_VERSION],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_current_version(conn: duckdb.DuckDBPyConnection) -> int | None:
    """Return the latest applied schema version, or ``None`` if uninitialized.

    Args:
        conn: An open DuckDB connection.

    Returns:
        The schema version integer, or ``None``.
    """
    try:
        result = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        return result[0] if result else None
    except duckdb.CatalogException:
        return None


def verify_schema(conn: duckdb.DuckDBPyConnection) -> dict[str, bool]:
    """Check that all expected tables exist.

    Returns:
        A mapping of ``{table_name: exists}`` for every expected table.
    """
    existing_tables: set[str] = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    return {table: table in existing_tables for table in sorted(_EXPECTED_TABLES)}
