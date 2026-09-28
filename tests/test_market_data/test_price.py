"""Tests for historical price fetcher with DuckDB caching."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from earnings_intel.market_data.price_fetcher import (
    PriceFetchError,
    fetch_and_cache_prices,
    fetch_prices,
    get_cached_date_range,
    store_prices,
)


@pytest.fixture()
def mock_yf_dataframe() -> pd.DataFrame:
    """Synthetic price DataFrame mimicking yfinance output."""
    dates = pd.date_range(start="2023-11-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [170.0, 171.0, 172.0, 173.0, 174.0],
            "High": [172.0, 173.0, 174.0, 175.0, 176.0],
            "Low": [169.0, 170.0, 171.0, 172.0, 173.0],
            "Close": [171.0, 172.0, 173.0, 174.0, 175.0],
            "Adj Close": [171.0, 172.0, 173.0, 174.0, 175.0],
            "Volume": [50_000_000, 52_000_000, 48_000_000, 55_000_000, 60_000_000],
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


class TestPriceFetcher:
    """Verify price downloading, normalization, and caching."""

    def test_fetch_prices_normalizes_columns(
        self,
        mock_yf_dataframe: pd.DataFrame,
    ) -> None:
        with patch("yfinance.download", return_value=mock_yf_dataframe):
            df = fetch_prices("AAPL", date(2023, 11, 1), date(2023, 11, 7))

        assert not df.empty
        assert "ticker" in df.columns
        assert "trade_date" in df.columns
        assert "close_price" in df.columns
        assert df["ticker"].iloc[0] == "AAPL"
        assert len(df) == 5

    def test_fetch_prices_empty_raises(self) -> None:
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(PriceFetchError):
                fetch_prices("NONEXISTENT", date(2023, 1, 1), date(2023, 1, 10))

    def test_store_and_query_cache(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        mock_yf_dataframe: pd.DataFrame,
    ) -> None:
        with patch("yfinance.download", return_value=mock_yf_dataframe):
            df = fetch_prices("AAPL", date(2023, 11, 1), date(2023, 11, 7))

        stored = store_prices(memory_db, df)
        assert stored == 5

        # Query range from cache
        rng = get_cached_date_range(memory_db, "AAPL")
        assert rng is not None
        min_d, max_d = rng
        assert min_d == date(2023, 11, 1)
        assert max_d == date(2023, 11, 7)

    def test_get_cached_date_range_missing_ticker(
        self,
        memory_db: duckdb.DuckDBPyConnection,
    ) -> None:
        rng = get_cached_date_range(memory_db, "NO_DATA_TICKER")
        assert rng is None

    def test_fetch_and_cache_prices_smart_flow(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        mock_yf_dataframe: pd.DataFrame,
    ) -> None:
        with patch("yfinance.download", return_value=mock_yf_dataframe) as mock_dl:
            # First call -> cache miss, downloads
            df1 = fetch_and_cache_prices(memory_db, "AAPL", date(2023, 11, 1), date(2023, 11, 7))
            assert len(df1) == 5
            assert mock_dl.call_count == 1

            # Second call with same range -> cache hit, NO second download
            df2 = fetch_and_cache_prices(memory_db, "AAPL", date(2023, 11, 1), date(2023, 11, 7))
            assert len(df2) == 5
            assert mock_dl.call_count == 1  # Still 1! Hit cache
