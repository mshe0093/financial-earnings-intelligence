"""FRED (Federal Reserve Economic Data) API client for macroeconomic indicators.

Fetches key time-series (Fed Funds Rate, 10Y-2Y spread, VIX, CPI, Unemployment)
and stores them in the DuckDB ``macro_indicators`` table for contextual analysis.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import duckdb
import httpx
from pydantic import BaseModel, Field

from earnings_intel.config import Settings, get_settings

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Key macro series relevant to market context and earnings drift analysis
DEFAULT_MACRO_SERIES: list[str] = [
    "DFF",  # Federal Funds Effective Rate (daily)
    "T10Y2Y",  # 10-Year minus 2-Year Treasury Constant Maturity (daily yield curve)
    "VIXCLS",  # CBOE Volatility Index (daily)
    "CPIAUCSL",  # Consumer Price Index for All Urban Consumers (monthly)
    "UNRATE",  # Civilian Unemployment Rate (monthly)
]


class FredObservation(BaseModel):
    """A single observation data point from a FRED series."""

    obs_date: date
    value: float | None = None


class FredSeriesResult(BaseModel):
    """Full time-series download result from FRED."""

    series_id: str
    observations: list[FredObservation] = Field(default_factory=list)
    count: int = 0


class FredClient:
    """Async client for the Federal Reserve Economic Data (FRED) API.

    Usage:
        async with FredClient() as client:
            result = await client.get_series("DFF", start_date="2023-01-01")
            with get_connection() as conn:
                stored = store_fred_observations(conn, [result])
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._api_key = self._settings.fred_api_key.get_secret_value()
        self._client = httpx.AsyncClient(
            base_url=FRED_BASE_URL,
            timeout=30.0,
            follow_redirects=True,
        )

    async def get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> FredSeriesResult:
        """Download observations for a specific FRED series.

        Args:
            series_id: FRED series identifier (e.g. 'DFF', 'T10Y2Y').
            start_date: YYYY-MM-DD format start date.
            end_date: YYYY-MM-DD format end date.

        Returns:
            ``FredSeriesResult`` with parsed observations.
        """
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        response = await self._client.get("/series/observations", params=params)
        response.raise_for_status()
        data = response.json()

        raw_obs = data.get("observations", [])
        observations: list[FredObservation] = []

        for item in raw_obs:
            raw_val = item.get("value", "")
            # FRED represents missing/holiday values with a single period '.'
            val: float | None = None
            if raw_val and raw_val != ".":
                try:
                    val = float(raw_val)
                except ValueError:
                    val = None

            observations.append(
                FredObservation(
                    obs_date=date.fromisoformat(item["date"]),
                    value=val,
                )
            )

        logger.info(
            "Fetched %d observations for FRED series %s",
            len(observations),
            series_id,
        )
        return FredSeriesResult(
            series_id=series_id,
            observations=observations,
            count=len(observations),
        )

    async def get_multiple_series(
        self,
        series_ids: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FredSeriesResult]:
        """Fetch multiple FRED series sequentially to respect the 120 req/min rate limit."""
        target_ids = series_ids or DEFAULT_MACRO_SERIES
        results: list[FredSeriesResult] = []
        for sid in target_ids:
            try:
                res = await self.get_series(sid, start_date=start_date, end_date=end_date)
                results.append(res)
            except Exception as e:
                logger.error("Failed to fetch FRED series %s: %s", sid, e)
        return results

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self._client.aclose()

    async def __aenter__(self) -> FredClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def store_fred_observations(
    conn: duckdb.DuckDBPyConnection,
    results: list[FredSeriesResult],
) -> int:
    """Store FRED series observations into the DuckDB ``macro_indicators`` table.

    Uses ``INSERT OR REPLACE`` for idempotent upserts.

    Args:
        conn: An open DuckDB connection.
        results: List of ``FredSeriesResult`` objects.

    Returns:
        Total number of rows inserted/updated.
    """
    total_rows = 0
    rows: list[tuple[str, date, float | None]] = []

    for series_result in results:
        for obs in series_result.observations:
            rows.append((series_result.series_id, obs.obs_date, obs.value))

    if rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO macro_indicators (series_id, obs_date, value)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        total_rows = len(rows)

    logger.info("Stored %d macro indicator observations in DuckDB", total_rows)
    return total_rows
