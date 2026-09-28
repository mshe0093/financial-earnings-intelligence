"""Tests for Post-Earnings Announcement Drift (PEAD) and CAR calculation."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from earnings_intel.backtesting.pead import (
    InsufficientPriceDataError,
    PEADResult,
    compute_pead,
    compute_pead_from_dataframes,
)


def _generate_synthetic_prices(
    start_date: date,
    num_days: int,
    initial_price: float,
    daily_gain_pct: float,
) -> pd.DataFrame:
    """Generate deterministic daily price series."""
    dates: list[date] = []
    prices: list[float] = []
    curr = initial_price

    cur_date = start_date
    while len(dates) < num_days:
        # Skip weekends to mimic trading days
        if cur_date.weekday() < 5:
            dates.append(cur_date)
            prices.append(round(curr, 4))
            curr = curr * (1.0 + daily_gain_pct)
        cur_date += timedelta(days=1)

    return pd.DataFrame({"trade_date": dates, "adj_close": prices})


class TestPEADFromDataFrames:
    """Tests for compute_pead_from_dataframes."""

    def test_known_abnormal_return_and_car(self) -> None:
        """Stock gains 1% daily while benchmark is flat (0%)."""
        filing_date = date(2023, 1, 10)
        # 11 trading days: Day 0 + 10-day window
        ticker_df = _generate_synthetic_prices(filing_date, 11, 100.0, 0.01)
        bench_df = _generate_synthetic_prices(filing_date, 11, 400.0, 0.00)

        result = compute_pead_from_dataframes(
            filing_id="filing_test_1",
            filing_date=filing_date,
            ticker="AAPL",
            ticker_df=ticker_df,
            benchmark_df=bench_df,
            window_days=10,
        )

        assert isinstance(result, PEADResult)
        assert result.ticker == "AAPL"
        assert result.filing_id == "filing_test_1"
        assert result.window_days == 10
        assert len(result.trading_dates) == 11
        assert len(result.car_series) == 11

        # Day 0 CAR is always 0.0
        assert result.car_series[0] == 0.0

        # Daily abnormal returns should be ~0.01 every day
        for i in range(1, 11):
            assert result.car_series[i] == pytest.approx(i * 0.01, abs=1e-3)

        # Raw return is 1.01^10 - 1 = ~0.104622
        expected_raw = (1.01**10) - 1.0
        assert result.raw_return == pytest.approx(expected_raw, rel=1e-3)
        assert result.benchmark_return == pytest.approx(0.0, abs=1e-5)
        assert result.abnormal_return == pytest.approx(expected_raw, rel=1e-3)

    def test_underperforming_benchmark(self) -> None:
        """Stock is flat while benchmark gains 2% daily."""
        filing_date = date(2023, 3, 1)
        ticker_df = _generate_synthetic_prices(filing_date, 6, 50.0, 0.00)
        bench_df = _generate_synthetic_prices(filing_date, 6, 300.0, 0.02)

        result = compute_pead_from_dataframes(
            filing_id="filing_test_under",
            filing_date=filing_date,
            ticker="XYZ",
            ticker_df=ticker_df,
            benchmark_df=bench_df,
            window_days=5,
        )

        assert result.raw_return == pytest.approx(0.0, abs=1e-5)
        expected_bench = (1.02**5) - 1.0
        assert result.benchmark_return == pytest.approx(expected_bench, rel=1e-3)
        assert result.abnormal_return < 0.0
        assert result.car_series[-1] == pytest.approx(-5 * 0.02, abs=1e-3)

    def test_ignores_dates_before_filing(self) -> None:
        """Prices before filing_date must not affect the window calculation."""
        filing_date = date(2023, 5, 10)
        start_date = filing_date - timedelta(days=20)
        ticker_df = _generate_synthetic_prices(start_date, 30, 100.0, 0.01)
        bench_df = _generate_synthetic_prices(start_date, 30, 400.0, 0.00)

        result = compute_pead_from_dataframes(
            filing_id="filing_pre_test",
            filing_date=filing_date,
            ticker="MSFT",
            ticker_df=ticker_df,
            benchmark_df=bench_df,
            window_days=5,
        )

        assert result.start_date >= filing_date

    def test_insufficient_data_raises_error(self) -> None:
        """Fewer than 2 overlapping trading days post-filing must raise InsufficientPriceDataError."""
        filing_date = date(2023, 6, 1)
        ticker_df = pd.DataFrame({"trade_date": [filing_date], "adj_close": [100.0]})
        bench_df = pd.DataFrame({"trade_date": [filing_date], "adj_close": [400.0]})

        with pytest.raises(InsufficientPriceDataError, match="minimum 2 required"):
            compute_pead_from_dataframes(
                filing_id="filing_short",
                filing_date=filing_date,
                ticker="TEST",
                ticker_df=ticker_df,
                benchmark_df=bench_df,
                window_days=10,
            )


class TestPEADDuckDB:
    """Tests for compute_pead directly querying DuckDB tables."""

    def test_compute_pead_with_duckdb(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Query price_series table in DuckDB and compute PEAD."""
        filing_date = date(2023, 2, 1)

        # Populate price_series for NVDA and SPY
        dates = [filing_date + timedelta(days=i) for i in range(10)]
        for i, d in enumerate(dates):
            memory_db.execute(
                """
                INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES ('NVDA', ?, ?, ?)
                """,
                [d, 100.0 + i, 100.0 + i],
            )
            memory_db.execute(
                """
                INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES ('SPY', ?, ?, ?)
                """,
                [d, 400.0 + (i * 0.5), 400.0 + (i * 0.5)],
            )

        result = compute_pead(
            filing_id="filing_nvda",
            filing_date=filing_date,
            ticker="NVDA",
            conn=memory_db,
            benchmark_ticker="SPY",
            window_days=5,
        )

        assert result.ticker == "NVDA"
        assert result.window_days == 5
        assert result.raw_return > 0.0

    def test_missing_ticker_in_duckdb_raises(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Missing ticker in DuckDB price_series raises InsufficientPriceDataError."""
        with pytest.raises(InsufficientPriceDataError, match=r"No price records found.*UNKNOWN"):
            compute_pead(
                filing_id="f1",
                filing_date=date(2023, 1, 1),
                ticker="UNKNOWN",
                conn=memory_db,
            )

    def test_missing_benchmark_in_duckdb_raises(
        self, memory_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Missing benchmark in DuckDB price_series raises InsufficientPriceDataError."""
        memory_db.execute(
            """
            INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
            VALUES ('EXISTING', '2023-01-01', 10.0, 10.0), ('EXISTING', '2023-01-02', 11.0, 11.0)
            """
        )
        with pytest.raises(InsufficientPriceDataError, match=r"No price records found.*SPY"):
            compute_pead(
                filing_id="f1",
                filing_date=date(2023, 1, 1),
                ticker="EXISTING",
                conn=memory_db,
                benchmark_ticker="SPY",
            )
