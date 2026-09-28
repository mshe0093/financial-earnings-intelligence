"""Tests for earnings_intel.db — connection factory and schema initialization."""

from __future__ import annotations

import duckdb
import pytest

from earnings_intel.db.connection import get_memory_connection
from earnings_intel.db.schema import (
    SCHEMA_VERSION,
    get_current_version,
    initialize_schema,
    verify_schema,
)


class TestSchemaInitialization:
    """Verify that initialize_schema creates all expected tables."""

    def test_all_tables_created(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Every expected table should exist after initialization."""
        status = verify_schema(memory_db)
        for table_name, exists in status.items():
            assert exists, f"Table '{table_name}' was not created"

    def test_schema_version_recorded(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """The schema version should be written to _schema_version."""
        version = get_current_version(memory_db)
        assert version == SCHEMA_VERSION

    def test_idempotent_initialization(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Calling initialize_schema twice should not raise or duplicate rows."""
        # First call already happened in the fixture. Call again:
        initialize_schema(memory_db)

        # Version table should still have exactly the expected version.
        version = get_current_version(memory_db)
        assert version == SCHEMA_VERSION

        # All tables still exist.
        status = verify_schema(memory_db)
        assert all(status.values())


class TestFilingsMetadataTable:
    """Verify constraints on the filings_metadata table."""

    def test_insert_valid_filing(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """A valid row should insert successfully."""
        memory_db.execute("""
            INSERT INTO filings_metadata
                (filing_id, cik, ticker, company_name, form_type,
                 filing_date, period_of_report, accession_number, primary_doc_url)
            VALUES
                ('0000320193_0000320193-23-000106', '0000320193', 'AAPL',
                 'Apple Inc', '10-K', '2023-11-03', '2023-09-30',
                 '0000320193-23-000106',
                 'https://www.sec.gov/Archives/edgar/data/320193/filing.htm')
        """)
        row = memory_db.execute("SELECT filing_id, ticker FROM filings_metadata").fetchone()
        assert row is not None
        assert row[0] == "0000320193_0000320193-23-000106"
        assert row[1] == "AAPL"

    def test_invalid_form_type_rejected(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """The CHECK constraint should reject form types other than 10-K/10-Q."""
        with pytest.raises(duckdb.ConstraintException):
            memory_db.execute("""
                INSERT INTO filings_metadata
                    (filing_id, cik, ticker, company_name, form_type,
                     filing_date, period_of_report, accession_number, primary_doc_url)
                VALUES
                    ('test_bad_form', '0000320193', 'AAPL', 'Apple Inc', '8-K',
                     '2023-11-03', '2023-09-30', 'acc-num',
                     'https://example.com/filing.htm')
            """)

    def test_invalid_cik_length_rejected(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """The CHECK constraint should reject CIKs that aren't exactly 10 chars."""
        with pytest.raises(duckdb.ConstraintException):
            memory_db.execute("""
                INSERT INTO filings_metadata
                    (filing_id, cik, ticker, company_name, form_type,
                     filing_date, period_of_report, accession_number, primary_doc_url)
                VALUES
                    ('test_bad_cik', '320193', 'AAPL', 'Apple Inc', '10-K',
                     '2023-11-03', '2023-09-30', 'acc-num',
                     'https://example.com/filing.htm')
            """)

    def test_duplicate_filing_id_rejected(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Primary key should prevent duplicate filing_id."""
        insert_sql = """
            INSERT INTO filings_metadata
                (filing_id, cik, ticker, company_name, form_type,
                 filing_date, period_of_report, accession_number, primary_doc_url)
            VALUES
                ('dup_test', '0000320193', 'AAPL', 'Apple Inc', '10-K',
                 '2023-11-03', '2023-09-30', 'acc-num',
                 'https://example.com/filing.htm')
        """
        memory_db.execute(insert_sql)
        with pytest.raises(duckdb.ConstraintException):
            memory_db.execute(insert_sql)


class TestPriceSeriesTable:
    """Verify the price_series table structure."""

    def test_insert_and_query_price(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Basic insert and retrieve should work."""
        memory_db.execute("""
            INSERT INTO price_series (ticker, trade_date, close_price, adj_close, volume)
            VALUES ('AAPL', '2023-11-03', 176.65, 176.65, 79829246)
        """)
        row = memory_db.execute(
            "SELECT ticker, close_price FROM price_series WHERE ticker = 'AAPL'"
        ).fetchone()
        assert row is not None
        assert row[1] == pytest.approx(176.65)

    def test_composite_primary_key(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """Duplicate (ticker, trade_date) should be rejected."""
        insert_sql = """
            INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
            VALUES ('SPY', '2023-11-03', 435.00, 435.00)
        """
        memory_db.execute(insert_sql)
        with pytest.raises(duckdb.ConstraintException):
            memory_db.execute(insert_sql)


class TestConnectionFactory:
    """Verify the connection factory utility."""

    def test_memory_connection_works(self) -> None:
        """get_memory_connection should yield a usable connection."""
        with get_memory_connection() as conn:
            result = conn.execute("SELECT 42 AS answer").fetchone()
            assert result is not None
            assert result[0] == 42

    def test_memory_connection_closes(self) -> None:
        """Connection should be closed after exiting the context manager."""
        with get_memory_connection() as conn:
            pass
        # After exit, calling execute should raise.
        with pytest.raises(Exception):  # noqa: B017
            conn.execute("SELECT 1")
