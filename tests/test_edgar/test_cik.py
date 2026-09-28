"""Tests for CIK normalization, ticker lookup, and DuckDB storage."""

from __future__ import annotations

import duckdb
import pytest

from earnings_intel.edgar.cik import (
    lookup_cik,
    normalize_cik,
    store_ticker_mapping,
)
from earnings_intel.edgar.models import CompanyInfo, FilingMetadata


class TestCikNormalization:
    """Verify normalize_cik pads correctly to 10 digits."""

    def test_normalize_cik_from_int(self) -> None:
        assert normalize_cik(320193) == "0000320193"

    def test_normalize_cik_from_string(self) -> None:
        assert normalize_cik("320193") == "0000320193"

    def test_normalize_cik_already_padded(self) -> None:
        assert normalize_cik("0000320193") == "0000320193"

    def test_normalize_cik_strips_whitespace(self) -> None:
        assert normalize_cik("  320193  ") == "0000320193"

    def test_normalize_cik_single_digit(self) -> None:
        assert normalize_cik(1) == "0000000001"


class TestTickerLookup:
    """Verify ticker to CIK mapping logic."""

    @pytest.fixture()
    def sample_mapping(self) -> dict[str, CompanyInfo]:
        return {
            "AAPL": CompanyInfo(cik="0000320193", ticker="AAPL", title="Apple Inc."),
            "MSFT": CompanyInfo(cik="0000789019", ticker="MSFT", title="Microsoft Corp"),
            "GOOGL": CompanyInfo(cik="0001652044", ticker="GOOGL", title="Alphabet Inc."),
        }

    def test_lookup_cik_found(self, sample_mapping: dict[str, CompanyInfo]) -> None:
        assert lookup_cik("AAPL", sample_mapping) == "0000320193"

    def test_lookup_cik_case_insensitive(self, sample_mapping: dict[str, CompanyInfo]) -> None:
        assert lookup_cik("aapl", sample_mapping) == "0000320193"
        assert lookup_cik("Msft", sample_mapping) == "0000789019"

    def test_lookup_cik_not_found(self, sample_mapping: dict[str, CompanyInfo]) -> None:
        assert lookup_cik("UNKNOWN", sample_mapping) is None

    def test_store_and_retrieve_mapping(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        sample_mapping: dict[str, CompanyInfo],
    ) -> None:
        """Verify DuckDB persistence of the ticker mapping."""
        stored = store_ticker_mapping(memory_db, sample_mapping)
        assert stored == 3

        row = memory_db.execute(
            "SELECT cik, company_name FROM ticker_cik_mapping WHERE ticker = 'AAPL'"
        ).fetchone()
        assert row is not None
        assert row[0] == "0000320193"
        assert "Apple" in row[1]


class TestFilingMetadataValidation:
    """Verify FilingMetadata Pydantic model validation rules."""

    def test_valid_filing_metadata(self) -> None:
        from datetime import date

        f = FilingMetadata(
            filing_id="0000320193_0000320193-23-000106",
            cik="0000320193",
            ticker="AAPL",
            company_name="Apple Inc",
            form_type="10-K",
            filing_date=date(2023, 11, 3),
            period_of_report=date(2023, 9, 30),
            accession_number="0000320193-23-000106",
            primary_doc_url="https://example.com/filing.htm",
        )
        assert f.cik == "0000320193"
        assert f.form_type == "10-K"

    def test_invalid_cik_raises(self) -> None:
        from datetime import date

        with pytest.raises(Exception):  # noqa: B017
            FilingMetadata(
                filing_id="test",
                cik="123",  # Not 10 digits
                company_name="Test",
                form_type="10-K",
                filing_date=date(2023, 1, 1),
                period_of_report=date(2023, 1, 1),
                accession_number="acc",
                primary_doc_url="http://example.com",
            )

    def test_invalid_form_type_raises(self) -> None:
        from datetime import date

        with pytest.raises(Exception):  # noqa: B017
            FilingMetadata(
                filing_id="test",
                cik="0000320193",
                company_name="Test",
                form_type="8-K",  # Only 10-K and 10-Q allowed
                filing_date=date(2023, 1, 1),
                period_of_report=date(2023, 1, 1),
                accession_number="acc",
                primary_doc_url="http://example.com",
            )
