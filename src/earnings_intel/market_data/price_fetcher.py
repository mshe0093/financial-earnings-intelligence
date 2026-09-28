"""Historical stock price fetcher using yfinance with DuckDB caching.

Implements smart caching:
1. Check the local ``price_series`` DuckDB table first.
2. If cached range covers the requested window, return from DB immediately.
3. If not, download only missing date ranges from yfinance and upsert to DuckDB.
4. Returns a clean, consistent Pandas DataFrame for quantitative analysis.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import duckdb
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetchError(Exception):
    """Raised when historical price data could not be fetched or is empty."""


def fetch_prices(
    ticker: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download daily OHLCV prices from yfinance.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'SPY').
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        DataFrame with columns:
        ``[ticker, trade_date, open_price, high_price, low_price, close_price, adj_close, volume]``

    Raises:
        PriceFetchError: If download fails, ticker is invalid, or no data returned.
    """
    clean_ticker = ticker.strip().upper()
    # yfinance end date is exclusive, so add 1 day to include end_date
    yf_end = end_date + timedelta(days=1)

    try:
        df = yf.download(
            tickers=clean_ticker,
            start=start_date.isoformat(),
            end=yf_end.isoformat(),
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise PriceFetchError(f"yfinance download failed for {clean_ticker}: {e}") from e

    if df.empty:
        raise PriceFetchError(
            f"No price data returned by yfinance for {clean_ticker} "
            f"between {start_date} and {end_date}."
        )

    # Flatten MultiIndex columns if present (yfinance v0.2+ often returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to turn DatetimeIndex into a regular column
    df = df.reset_index()

    # Map column names standardly
    col_mapping = {
        "Date": "trade_date",
        "Open": "open_price",
        "High": "high_price",
        "Low": "low_price",
        "Close": "close_price",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_mapping)

    # Add ticker column
    df["ticker"] = clean_ticker

    # Ensure trade_date is date objects (not timestamps)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # Select and order standard columns
    standard_cols = [
        "ticker",
        "trade_date",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "adj_close",
        "volume",
    ]
    available_cols = [c for c in standard_cols if c in df.columns]
    result = df[available_cols].copy()

    # Drop any rows with NaN in close_price or adj_close
    result = result.dropna(subset=["close_price", "adj_close"])

    logger.info(
        "Downloaded %d price bars for %s (%s to %s)",
        len(result),
        clean_ticker,
        start_date,
        end_date,
    )
    return result


def get_cached_date_range(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
) -> tuple[date, date] | None:
    """Query DuckDB for the min and max trade dates cached for a ticker.

    Returns:
        ``(min_date, max_date)`` tuple, or ``None`` if no prices are cached.
    """
    clean_ticker = ticker.strip().upper()
    try:
        row = conn.execute(
            """
            SELECT MIN(trade_date), MAX(trade_date)
            FROM price_series
            WHERE ticker = ?
            """,
            [clean_ticker],
        ).fetchone()
        if row and row[0] is not None and row[1] is not None:
            return (row[0], row[1])
        return None
    except duckdb.CatalogException:
        return None


def store_prices(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """Upsert price DataFrame into the DuckDB ``price_series`` table.

    Args:
        conn: An open DuckDB connection.
        df: DataFrame conforming to ``price_series`` schema.

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        return 0

    # Register DataFrame as a temporary view and perform upsert
    conn.register("_temp_price_import", df)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO price_series
                (ticker, trade_date, open_price, high_price, low_price,
                 close_price, adj_close, volume)
            SELECT
                ticker, trade_date, open_price, high_price, low_price,
                close_price, adj_close, volume
            FROM _temp_price_import
        """)
        row_count = len(df)
        logger.info("Cached %d price records for %s", row_count, df["ticker"].iloc[0])
        return row_count
    finally:
        conn.unregister("_temp_price_import")


def fetch_and_cache_prices(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Smart price fetcher with cache check.

    If the requested range is not fully covered in DuckDB, downloads the
    full range, updates the cache, and returns the query result.

    Args:
        conn: An open DuckDB connection.
        ticker: Ticker symbol.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame of prices sorted by trade_date ascending.
    """
    clean_ticker = ticker.strip().upper()
    cached_range = get_cached_date_range(conn, clean_ticker)

    # If cache is missing or does not fully cover the window, fetch from yfinance
    needs_fetch = False
    if cached_range is None:
        needs_fetch = True
    else:
        cached_min, cached_max = cached_range
        # Allow 4-day tolerance for weekends/holidays
        if start_date < cached_min - timedelta(days=4) or end_date > cached_max + timedelta(
            days=4
        ):
            needs_fetch = True

    if needs_fetch:
        logger.info(
            "Cache miss/partial hit for %s (%s to %s). Fetching from yfinance...",
            clean_ticker,
            start_date,
            end_date,
        )
        fresh_df = fetch_prices(clean_ticker, start_date, end_date)
        store_prices(conn, fresh_df)

    # Always return from DuckDB for consistent typing and ordering
    return conn.execute(
        """
        SELECT ticker, trade_date, open_price, high_price, low_price,
               close_price, adj_close, volume
        FROM price_series
        WHERE ticker = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date ASC
        """,
        [clean_ticker, start_date, end_date],
    ).df()
