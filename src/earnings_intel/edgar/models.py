"""Pydantic v2 data models for SEC EDGAR filing metadata and search responses."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FilingMetadata(BaseModel):
    """Metadata for a single SEC filing (10-K or 10-Q).

    Represents a row in the ``filings_metadata`` DuckDB table.
    """

    filing_id: str = Field(description="Unique ID: '{cik}_{accession_number}'")
    cik: str = Field(description="Zero-padded 10-digit CIK")
    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    company_name: str = Field(description="Official company name")
    form_type: str = Field(description="Filing form type: '10-K' or '10-Q'")
    filing_date: date = Field(description="Date filed with the SEC")
    period_of_report: date = Field(description="Fiscal period end date")
    accession_number: str = Field(description="SEC accession number (unique per filing)")
    primary_doc_url: str = Field(description="Full URL to the primary HTML document")
    mda_text_hash: str | None = Field(default=None, description="SHA-256 of extracted MD&A")
    ingested_at: datetime | None = Field(default=None, description="Ingestion timestamp")

    @field_validator("cik")
    @classmethod
    def validate_cik(cls, v: str) -> str:
        """CIK must be exactly 10 digits (zero-padded)."""
        stripped = v.strip()
        if not re.match(r"^\d{10}$", stripped):
            raise ValueError(f"CIK must be exactly 10 digits, got '{v}'")
        return stripped

    @field_validator("form_type")
    @classmethod
    def validate_form_type(cls, v: str) -> str:
        """Only 10-K and 10-Q filings are supported."""
        upper = v.strip().upper()
        if upper not in ("10-K", "10-Q"):
            raise ValueError(f"form_type must be '10-K' or '10-Q', got '{v}'")
        return upper


class CompanyInfo(BaseModel):
    """Entry from SEC company_tickers.json mapping."""

    cik: str = Field(description="Zero-padded 10-digit CIK")
    ticker: str = Field(description="Stock ticker symbol (uppercase)")
    title: str = Field(description="Company name as registered with the SEC")


class EdgarSearchResult(BaseModel):
    """Container for EDGAR search / submission query results."""

    total_hits: int = 0
    filings: list[FilingMetadata] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)
