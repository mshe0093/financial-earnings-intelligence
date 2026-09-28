"""Tests for EdgarClient — mock HTTP transport, rate limiting, and error handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from earnings_intel.config import Settings
from earnings_intel.edgar.client import (
    EdgarClient,
    EdgarNotFoundError,
    EdgarRateLimitError,
    EdgarServerError,
)


@pytest.fixture()
def client_settings(test_settings: Settings) -> Settings:
    return test_settings


class TestEdgarClientHeaders:
    """Verify SEC Fair Access headers are correctly injected."""

    def test_user_agent_header_set(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)
        assert client._headers["User-Agent"] == "TestApp test@example.com"
        assert "gzip" in client._headers["Accept-Encoding"]


class TestEdgarClientThrottledGet:
    """Verify HTTP behavior with mocked transport."""

    @pytest.mark.asyncio
    async def test_successful_request(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)

        # Mock the underlying httpx client's get method
        mock_response = httpx.Response(
            status_code=200,
            json={"status": "ok"},
            request=httpx.Request("GET", "https://www.sec.gov/test"),
        )
        client._client.get = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        resp = await client._throttled_get("https://www.sec.gov/test")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        await client.close()

    @pytest.mark.asyncio
    async def test_429_raises_rate_limit_error(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)

        mock_response = httpx.Response(
            status_code=429,
            request=httpx.Request("GET", "https://www.sec.gov/test"),
        )
        client._client.get = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(EdgarRateLimitError):
                await client._throttled_get("https://www.sec.gov/test")
        await client.close()

    @pytest.mark.asyncio
    async def test_404_raises_not_found_error(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)

        mock_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://www.sec.gov/missing"),
        )
        client._client.get = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        with pytest.raises(EdgarNotFoundError):
            await client._throttled_get("https://www.sec.gov/missing")
        await client.close()

    @pytest.mark.asyncio
    async def test_500_raises_server_error(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)

        mock_response = httpx.Response(
            status_code=503,
            request=httpx.Request("GET", "https://www.sec.gov/fail"),
        )
        client._client.get = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(EdgarServerError):
                await client._throttled_get("https://www.sec.gov/fail")
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, client_settings: Settings) -> None:
        async with EdgarClient(client_settings) as client:
            assert client is not None
            assert not client._client.is_closed
        assert client._client.is_closed


class TestSubmissionsParsing:
    """Verify parsing of SEC company submissions response."""

    @pytest.mark.asyncio
    async def test_get_filings_list_filters_forms(self, client_settings: Settings) -> None:
        client = EdgarClient(client_settings)

        # Mock the submissions JSON response
        mock_submissions = {
            "name": "Apple Inc.",
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "10-Q", "10-K"],
                    "accessionNumber": [
                        "0000320193-23-000106",
                        "0000320193-23-000099",
                        "0000320193-23-000077",
                        "0000320193-22-000108",
                    ],
                    "filingDate": ["2023-11-03", "2023-09-15", "2023-08-04", "2022-10-28"],
                    "reportDate": ["2023-09-30", "2023-09-15", "2023-07-01", "2022-09-24"],
                    "primaryDocument": [
                        "aapl-20230930.htm",
                        "form8k.htm",
                        "aapl-20230701.htm",
                        "aapl-20220924.htm",
                    ],
                }
            },
        }

        client.get_company_submissions = AsyncMock(  # type: ignore[method-assign]
            return_value=mock_submissions
        )

        filings = await client.get_filings_list(
            "0000320193",
            form_types=("10-K", "10-Q"),
            ticker="AAPL",
        )

        # Should include 10-K and 10-Q, excluding 8-K -> exactly 3 filings
        assert len(filings) == 3
        assert filings[0].form_type == "10-K"
        assert filings[1].form_type == "10-Q"
        assert filings[2].form_type == "10-K"
        assert filings[0].ticker == "AAPL"
        assert filings[0].company_name == "Apple Inc."
        assert "Archives/edgar/data/320193" in filings[0].primary_doc_url
        await client.close()
