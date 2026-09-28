"""Tests for FRED API client and macro indicator storage."""

from __future__ import annotations

from unittest.mock import AsyncMock

import duckdb
import httpx
import pytest

from earnings_intel.config import Settings
from earnings_intel.market_data.fred_client import (
    FredClient,
    store_fred_observations,
)


@pytest.fixture()
def fred_settings(test_settings: Settings) -> Settings:
    return test_settings


class TestFredClient:
    """Verify FRED client parsing and null handling."""

    @pytest.mark.asyncio
    async def test_get_series_parses_values_and_dots(self, fred_settings: Settings) -> None:
        client = FredClient(fred_settings)

        # Mock FRED API response — note '.' represents missing/holiday value
        mock_response = httpx.Response(
            status_code=200,
            json={
                "realtime_start": "2023-01-01",
                "realtime_end": "2023-01-03",
                "observation_start": "2023-01-01",
                "observation_end": "2023-01-03",
                "units": "Lin",
                "output_type": 1,
                "file_type": "json",
                "order_by": "observation_date",
                "sort_order": "asc",
                "count": 3,
                "offset": 0,
                "limit": 100000,
                "observations": [
                    {"date": "2023-01-01", "value": "."},  # Missing / holiday
                    {"date": "2023-01-02", "value": "4.33"},  # Valid float
                    {"date": "2023-01-03", "value": "4.35"},  # Valid float
                ],
            },
            request=httpx.Request("GET", "https://api.stlouisfed.org/fred/series/observations"),
        )
        client._client.get = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        result = await client.get_series("DFF")
        assert result.series_id == "DFF"
        assert result.count == 3
        assert result.observations[0].value is None  # Handled '.' correctly
        assert result.observations[1].value == 4.33
        assert result.observations[2].value == 4.35
        await client.close()

    def test_store_observations_in_duckdb(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        fred_settings: Settings,
    ) -> None:
        from datetime import date

        from earnings_intel.market_data.fred_client import (
            FredObservation,
            FredSeriesResult,
        )

        sample_result = FredSeriesResult(
            series_id="DFF",
            observations=[
                FredObservation(obs_date=date(2023, 1, 1), value=None),
                FredObservation(obs_date=date(2023, 1, 2), value=4.33),
            ],
            count=2,
        )

        rows = store_fred_observations(memory_db, [sample_result])
        assert rows == 2

        # Query back
        res = memory_db.execute(
            "SELECT obs_date, value FROM macro_indicators WHERE series_id = 'DFF' ORDER BY obs_date"
        ).fetchall()
        assert len(res) == 2
        assert res[0][1] is None
        assert res[1][1] == pytest.approx(4.33)
