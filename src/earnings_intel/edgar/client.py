"""Async HTTP client for SEC EDGAR with rate-limiting and Fair Access compliance.

Enforces SEC EDGAR Fair Access policy:
- Ceiling of ≤10 requests per second (default: 8 rps)
- Mandatory ``User-Agent`` header on every request: "AppName contact@email.com"
- Exponential backoff on HTTP 429 (Too Many Requests) and 5xx errors
- Full filing download from EDGAR Archives
- Submissions history via SEC Data API
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import duckdb
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from earnings_intel.config import Settings, get_settings
from earnings_intel.edgar.cik import normalize_cik
from earnings_intel.edgar.models import FilingMetadata

logger = logging.getLogger(__name__)

EDGAR_BASE_URL = "https://www.sec.gov"
EDGAR_DATA_URL = "https://data.sec.gov"
EDGAR_ARCHIVES_URL = f"{EDGAR_BASE_URL}/Archives/edgar/data"


# ── Exceptions ─────────────────────────────────────────────────────────────


class EdgarError(Exception):
    """Base exception for SEC EDGAR operations."""


class EdgarRateLimitError(EdgarError):
    """Raised when SEC returns HTTP 429 (Too Many Requests)."""


class EdgarServerError(EdgarError):
    """Raised on SEC 5xx server errors."""


class EdgarNotFoundError(EdgarError):
    """Raised when an accession number, CIK, or filing document is 404."""


# ── Client Implementation ─────────────────────────────────────────────────


class EdgarClient:
    """Async HTTP client for SEC EDGAR with Fair Access rate limiting.

    Usage:
        async with EdgarClient() as client:
            filings = await client.get_company_filings("AAPL")
            html_bytes = await client.get_filing_document(
                cik="0000320193",
                accession_number="0000320193-23-000106",
                primary_doc="aapl-20230930.htm",
            )
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._rate_limit = self._settings.edgar_rate_limit_rps
        self._min_interval = 1.0 / self._rate_limit
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

        self._headers = {
            "User-Agent": self._settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }

        # Create client with HTTP/2 support and generous timeout
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """Underlying httpx client (useful for passing to sub-functions)."""
        return self._client

    async def _throttle(self) -> None:
        """Enforce inter-request spacing to stay below the RPS ceiling."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                delay = self._min_interval - elapsed
                await asyncio.sleep(delay)
            self._last_request_time = time.monotonic()

    @retry(
        retry=retry_if_exception_type((EdgarRateLimitError, EdgarServerError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def _throttled_get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Execute a rate-limited GET request with retry on 429/5xx."""
        await self._throttle()

        req_headers = dict(self._headers)
        if headers:
            req_headers.update(headers)

        # Fix Host header for data.sec.gov endpoints
        if "data.sec.gov" in url:
            req_headers["Host"] = "data.sec.gov"
        elif "efts.sec.gov" in url:
            req_headers["Host"] = "efts.sec.gov"
        else:
            req_headers["Host"] = "www.sec.gov"

        start_time = time.monotonic()
        try:
            response = await self._client.get(url, params=params, headers=req_headers)
            elapsed = time.monotonic() - start_time
            logger.debug(
                "GET %s -> %d (%.2fs)",
                url,
                response.status_code,
                elapsed,
            )

            if response.status_code == 429:
                logger.warning("SEC EDGAR 429 Rate Limit hit on %s. Retrying...", url)
                raise EdgarRateLimitError(f"Rate limited by SEC EDGAR: {url}")
            if response.status_code == 404:
                raise EdgarNotFoundError(f"Resource not found on SEC EDGAR: {url}")
            if response.status_code >= 500:
                logger.warning("SEC EDGAR %d server error on %s", response.status_code, url)
                raise EdgarServerError(f"SEC server error ({response.status_code}): {url}")

            response.raise_for_status()
            return response

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise EdgarRateLimitError(str(e)) from e
            if e.response.status_code >= 500:
                raise EdgarServerError(str(e)) from e
            if e.response.status_code == 404:
                raise EdgarNotFoundError(str(e)) from e
            raise EdgarError(f"HTTP error: {e}") from e

    async def get_company_submissions(self, raw_cik: str | int) -> dict[str, Any]:
        """Fetch the full submissions history for a company from SEC Data API.

        Endpoint: ``https://data.sec.gov/submissions/CIK{10-digit-cik}.json``

        Args:
            raw_cik: CIK (integer or string, padded or unpadded).

        Returns:
            Parsed JSON dictionary of submissions metadata.
        """
        cik = normalize_cik(raw_cik)
        url = f"{EDGAR_DATA_URL}/submissions/CIK{cik}.json"
        response = await self._throttled_get(url)
        return response.json()

    async def get_filings_list(
        self,
        raw_cik: str | int,
        *,
        form_types: tuple[str, ...] = ("10-K", "10-Q"),
        max_filings: int = 50,
        ticker: str | None = None,
    ) -> list[FilingMetadata]:
        """Fetch and parse filing metadata for specified form types.

        Extracts recent filings from the ``submissions/CIK{cik}.json`` endpoint.

        Args:
            raw_cik: Company CIK.
            form_types: Form types to filter by (e.g. ('10-K', '10-Q')).
            max_filings: Maximum number of filings to return.
            ticker: Optional ticker symbol to attach to metadata.

        Returns:
            List of ``FilingMetadata`` objects sorted newest-first.
        """
        cik = normalize_cik(raw_cik)
        data = await self.get_company_submissions(cik)

        company_name = data.get("name", "Unknown")
        recent = data.get("filings", {}).get("recent", {})

        if not recent:
            logger.warning("No recent filings found for CIK %s", cik)
            return []

        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results: list[FilingMetadata] = []
        cik_int = str(int(cik))  # Unpadded for archive URL paths

        for i, form in enumerate(forms):
            if form not in form_types:
                continue

            acc_raw = accession_numbers[i]
            acc_nodash = acc_raw.replace("-", "")
            primary_doc = primary_docs[i]
            primary_url = f"{EDGAR_ARCHIVES_URL}/{cik_int}/{acc_nodash}/{primary_doc}"

            from datetime import date as d_date

            filing_d = d_date.fromisoformat(filing_dates[i])
            period_d = d_date.fromisoformat(report_dates[i]) if report_dates[i] else filing_d

            results.append(
                FilingMetadata(
                    filing_id=f"{cik}_{acc_raw}",
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    form_type=form,
                    filing_date=filing_d,
                    period_of_report=period_d,
                    accession_number=acc_raw,
                    primary_doc_url=primary_url,
                )
            )

            if len(results) >= max_filings:
                break

        logger.info(
            "Found %d filings matching %s for CIK %s (%s)",
            len(results),
            form_types,
            cik,
            company_name,
        )
        return results

    async def get_filing_document(
        self,
        raw_cik: str | int,
        accession_number: str,
        primary_doc: str,
    ) -> bytes:
        """Download raw filing HTML document from EDGAR Archives.

        Args:
            raw_cik: CIK (padded or unpadded).
            accession_number: e.g. "0000320193-23-000106"
            primary_doc: e.g. "aapl-20230930.htm"

        Returns:
            Raw bytes of the HTML document.
        """
        cik_int = str(int(normalize_cik(raw_cik)))
        acc_nodash = accession_number.replace("-", "")
        url = f"{EDGAR_ARCHIVES_URL}/{cik_int}/{acc_nodash}/{primary_doc}"
        response = await self._throttled_get(url)
        return response.content

    async def download_filing_by_url(self, primary_doc_url: str) -> bytes:
        """Download filing document directly by its full URL."""
        response = await self._throttled_get(primary_doc_url)
        return response.content

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        await self._client.aclose()

    async def __aenter__(self) -> EdgarClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def store_filing_metadata(
    conn: duckdb.DuckDBPyConnection,
    filing: FilingMetadata,
) -> None:
    """Store or update a single filing's metadata in DuckDB."""
    conn.execute(
        """
        INSERT OR REPLACE INTO filings_metadata (
            filing_id, cik, ticker, company_name, form_type,
            filing_date, period_of_report, accession_number,
            primary_doc_url, mda_text_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            filing.filing_id,
            filing.cik,
            filing.ticker,
            filing.company_name,
            filing.form_type,
            filing.filing_date,
            filing.period_of_report,
            filing.accession_number,
            filing.primary_doc_url,
            filing.mda_text_hash,
        ],
    )


def store_filings_metadata(
    conn: duckdb.DuckDBPyConnection,
    filings: list[FilingMetadata],
) -> int:
    """Batch store or update filing metadata records in DuckDB."""
    if not filings:
        return 0
    rows = [
        (
            f.filing_id,
            f.cik,
            f.ticker,
            f.company_name,
            f.form_type,
            f.filing_date,
            f.period_of_report,
            f.accession_number,
            f.primary_doc_url,
            f.mda_text_hash,
        )
        for f in filings
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO filings_metadata (
            filing_id, cik, ticker, company_name, form_type,
            filing_date, period_of_report, accession_number,
            primary_doc_url, mda_text_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)
