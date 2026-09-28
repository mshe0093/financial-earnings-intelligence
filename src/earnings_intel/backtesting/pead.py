"""Post-Earnings Announcement Drift (PEAD) computation and Cumulative Abnormal Return (CAR).

Quantifies stock price reaction and abnormal drift following an SEC earnings filing
benchmarked against market performance (e.g. S&P 500 / SPY) over a holding window.
"""

from __future__ import annotations

import logging
from datetime import date

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InsufficientPriceDataError(Exception):
    """Raised when there is insufficient historical price data to compute PEAD."""


class PEADResult(BaseModel):
    """Result of Post-Earnings Announcement Drift (PEAD) computation."""

    filing_id: str = Field(description="Unique SEC filing ID")
    ticker: str = Field(description="Stock ticker symbol")
    filing_date: date = Field(description="Date the filing was submitted to the SEC")
    window_days: int = Field(default=30, description="Holding period in trading days")
    start_date: date = Field(description="First trading day of the holding window")
    end_date: date = Field(description="Final trading day of the holding window")
    raw_return: float = Field(description="Total stock return over the window")
    benchmark_return: float = Field(description="Benchmark (e.g. SPY) return over the same window")
    abnormal_return: float = Field(
        description="Abnormal return: raw_return minus benchmark_return"
    )
    car_series: list[float] = Field(
        default_factory=list,
        description="Daily Cumulative Abnormal Return (CAR) progression",
    )
    trading_dates: list[date] = Field(
        default_factory=list,
        description="Trading dates evaluated during the window",
    )


def compute_pead_from_dataframes(
    filing_id: str,
    filing_date: date,
    ticker: str,
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    window_days: int = 30,
) -> PEADResult:
    """Compute 30-day Post-Earnings Announcement Drift directly from price DataFrames.

    Args:
        filing_id: Identifier for the filing (e.g. '{cik}_{accession}').
        filing_date: SEC filing date.
        ticker: Ticker symbol (e.g. 'AAPL').
        ticker_df: DataFrame with at least ['trade_date', 'adj_close'].
        benchmark_df: Benchmark DataFrame with at least ['trade_date', 'adj_close'].
        window_days: Number of trading days post-filing to evaluate (default 30).

    Returns:
        `PEADResult` containing returns, CAR series, and trading dates.

    Raises:
        InsufficientPriceDataError: If price data has fewer than 2 trading days.
    """
    # Standardize column names
    t_df = ticker_df[["trade_date", "adj_close"]].copy()
    b_df = benchmark_df[["trade_date", "adj_close"]].copy()

    # Ensure trade_date is date objects
    t_df["trade_date"] = pd.to_datetime(t_df["trade_date"]).dt.date
    b_df["trade_date"] = pd.to_datetime(b_df["trade_date"]).dt.date

    # Filter for trading days on or after filing date
    t_df = t_df[t_df["trade_date"] >= filing_date].sort_values("trade_date")
    b_df = b_df[b_df["trade_date"] >= filing_date].sort_values("trade_date")

    # Inner join on trade_date to align exact trading sessions
    merged = pd.merge(
        t_df,
        b_df,
        on="trade_date",
        suffixes=("_stock", "_bench"),
    ).sort_values("trade_date")

    if len(merged) < 2:
        raise InsufficientPriceDataError(
            f"Filing {filing_id} for {ticker} on {filing_date} has only {len(merged)} "
            "common trading day(s); minimum 2 required."
        )

    # Slice to event window (day 0 + window_days trading sessions)
    effective_window = min(len(merged) - 1, window_days)
    window_slice = merged.iloc[: effective_window + 1].copy()

    dates: list[date] = list(window_slice["trade_date"])
    start_date = dates[0]
    end_date = dates[-1]

    # Calculate daily percentage returns
    window_slice["stock_ret"] = window_slice["adj_close_stock"].pct_change()
    window_slice["bench_ret"] = window_slice["adj_close_bench"].pct_change()
    window_slice["abnormal_ret"] = window_slice["stock_ret"] - window_slice["bench_ret"]

    # Day 0 CAR is 0.0; subsequent days accumulate daily abnormal return
    car_series = [0.0]
    for ar in window_slice["abnormal_ret"].iloc[1:]:
        val = 0.0 if pd.isna(ar) else float(ar)
        car_series.append(round(car_series[-1] + val, 6))

    # Total cumulative returns over the full window
    stock_p0 = float(window_slice["adj_close_stock"].iloc[0])
    stock_pN = float(window_slice["adj_close_stock"].iloc[-1])
    bench_p0 = float(window_slice["adj_close_bench"].iloc[0])
    bench_pN = float(window_slice["adj_close_bench"].iloc[-1])

    raw_return = (stock_pN / stock_p0) - 1.0 if stock_p0 > 0 else 0.0
    benchmark_return = (bench_pN / bench_p0) - 1.0 if bench_p0 > 0 else 0.0
    abnormal_return = raw_return - benchmark_return

    return PEADResult(
        filing_id=filing_id,
        ticker=ticker.upper(),
        filing_date=filing_date,
        window_days=effective_window,
        start_date=start_date,
        end_date=end_date,
        raw_return=round(raw_return, 6),
        benchmark_return=round(benchmark_return, 6),
        abnormal_return=round(abnormal_return, 6),
        car_series=car_series,
        trading_dates=dates,
    )


def compute_pead(
    filing_id: str,
    filing_date: date,
    ticker: str,
    conn: duckdb.DuckDBPyConnection,
    benchmark_ticker: str = "SPY",
    window_days: int = 30,
) -> PEADResult:
    """Query DuckDB price_series table and compute PEAD.

    Args:
        filing_id: Unique filing identifier.
        filing_date: SEC filing date.
        ticker: Target company stock ticker symbol.
        conn: Open DuckDB connection containing `price_series`.
        benchmark_ticker: Market benchmark symbol (default: 'SPY').
        window_days: Holding window in trading days (default: 30).

    Returns:
        `PEADResult` model.

    Raises:
        InsufficientPriceDataError: If price history is unavailable or sparse.
    """
    clean_ticker = ticker.strip().upper()
    clean_bench = benchmark_ticker.strip().upper()

    ticker_df = conn.execute(
        """
        SELECT trade_date, adj_close
        FROM price_series
        WHERE ticker = ? AND trade_date >= ?
        ORDER BY trade_date ASC
        """,
        [clean_ticker, filing_date],
    ).df()

    bench_df = conn.execute(
        """
        SELECT trade_date, adj_close
        FROM price_series
        WHERE ticker = ? AND trade_date >= ?
        ORDER BY trade_date ASC
        """,
        [clean_bench, filing_date],
    ).df()

    if ticker_df.empty:
        raise InsufficientPriceDataError(
            f"No price records found in DuckDB for ticker {clean_ticker} "
            f"on or after {filing_date}."
        )

    if bench_df.empty:
        raise InsufficientPriceDataError(
            f"No price records found in DuckDB for benchmark {clean_bench} "
            f"on or after {filing_date}."
        )

    return compute_pead_from_dataframes(
        filing_id=filing_id,
        filing_date=filing_date,
        ticker=clean_ticker,
        ticker_df=ticker_df,
        benchmark_df=bench_df,
        window_days=window_days,
    )
