"""Integration tests for FastAPI REST API endpoints."""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, timedelta
from unittest.mock import patch

import duckdb
import pytest
from fastapi.testclient import TestClient

from earnings_intel.api.main import create_app
from earnings_intel.api.routes import get_db
from earnings_intel.extraction.schemas import (
    CapexDirection,
    EarningsSignals,
    GuidanceDirection,
    MarginOutlook,
)


@pytest.fixture
def client(memory_db: duckdb.DuckDBPyConnection) -> Generator[TestClient, None, None]:
    """TestClient configured with in-memory database dependency override."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: memory_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_db(memory_db: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """Populate in-memory DB with known filings, signals, and price series."""
    base_date = date(2023, 2, 1)

    # 1. Filings
    memory_db.execute(
        """
        INSERT INTO filings_metadata (
            filing_id, cik, ticker, company_name, form_type,
            filing_date, period_of_report, accession_number, primary_doc_url
        ) VALUES
        ('f_aapl_10k', '0000320193', 'AAPL', 'Apple Inc.', '10-K', ?, ?, 'acc_1', 'https://sec.gov/1'),
        ('f_tsla_10q', '0001318605', 'TSLA', 'Tesla, Inc.', '10-Q', ?, ?, 'acc_2', 'https://sec.gov/2')
        """,
        [base_date, base_date, base_date + timedelta(days=10), base_date + timedelta(days=10)],
    )

    # 2. Extracted Signals for AAPL
    memory_db.execute(
        """
        INSERT INTO extracted_signals (
            signal_id, filing_id, extraction_model, revenue_guidance,
            revenue_guidance_detail, eps_guidance, eps_guidance_detail,
            margin_outlook, capex_direction, management_sentiment,
            forward_language_ratio, risk_factor_count, guidance_confidence,
            restructuring_signals
        ) VALUES (
            'sig_aapl_1', 'f_aapl_10k', 'gemini-2.5-flash', 'raise',
            'Strong iPhone demand', 'raise', '', 'expanding', 'stable',
            0.50, 0.45, 2, 0.95, false
        )
        """
    )

    # 3. Prices for AAPL, TSLA, SPY, QQQ, IWM
    dates = [base_date + timedelta(days=i) for i in range(25)]
    for i, d in enumerate(dates):
        memory_db.execute(
            """
            INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
            VALUES
            ('AAPL', ?, ?, ?),
            ('TSLA', ?, ?, ?),
            ('SPY', ?, ?, ?),
            ('QQQ', ?, ?, ?),
            ('IWM', ?, ?, ?)
            """,
            [
                d,
                150.0 + i,
                150.0 + i,
                d,
                200.0 - i,
                200.0 - i,
                d,
                400.0 + (i * 0.5),
                400.0 + (i * 0.5),
                d,
                350.0 + (i * 0.8),
                350.0 + (i * 0.8),
                d,
                180.0 + (i * 0.2),
                180.0 + (i * 0.2),
            ],
        )

    return memory_db


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_check_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns 200 with database connection info."""
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        assert isinstance(data["filing_count"], int)
        assert isinstance(data["price_count"], int)


class TestFilingsEndpoints:
    """Tests for filing exploration and signals retrieval."""

    def test_list_filings_empty(self, client: TestClient) -> None:
        """Empty database returns zero filings."""
        res = client.get("/api/filings")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["filings"] == []

    def test_list_filings_with_filters(
        self, client: TestClient, seeded_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Filter filings by ticker and form type."""
        # Query all
        res = client.get("/api/filings")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 2
        assert len(data["filings"]) == 2

        # Filter ticker
        res_aapl = client.get("/api/filings?ticker=AAPL")
        data_aapl = res_aapl.json()
        assert data_aapl["total"] == 1
        assert data_aapl["filings"][0]["ticker"] == "AAPL"
        assert data_aapl["filings"][0]["has_signals"] is True
        assert data_aapl["filings"][0]["management_sentiment"] == pytest.approx(0.50)

        # Filter form_type
        res_10q = client.get("/api/filings?form_type=10-Q")
        data_10q = res_10q.json()
        assert data_10q["total"] == 1
        assert data_10q["filings"][0]["ticker"] == "TSLA"
        assert data_10q["filings"][0]["has_signals"] is False

    def test_get_filing_signals_found(
        self, client: TestClient, seeded_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Retrieve full 12 metrics for a filing."""
        res = client.get("/api/filings/f_aapl_10k/signals")
        assert res.status_code == 200
        data = res.json()
        assert data["filing_id"] == "f_aapl_10k"
        assert data["revenue_guidance"] == "raise"
        assert data["margin_outlook"] == "expanding"
        assert data["management_sentiment"] == pytest.approx(0.50)

    def test_get_filing_signals_not_found(self, client: TestClient) -> None:
        """Non-existent filing signals return 404."""
        res = client.get("/api/filings/non_existent_filing/signals")
        assert res.status_code == 404
        assert "No extracted signals found" in res.json()["detail"]


class TestExtractEndpoint:
    """Tests for on-demand extraction trigger."""

    def test_extract_filing_not_found(self, client: TestClient) -> None:
        """Trigger extraction on unknown filing returns 404."""
        res = client.post("/api/filings/unknown_id/extract")
        assert res.status_code == 404

    def test_extract_filing_success(
        self, client: TestClient, seeded_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Trigger on-demand extraction on an existing filing with mocked Gemini client."""
        mock_signals = EarningsSignals(
            revenue_guidance=GuidanceDirection.LOWER,
            revenue_guidance_detail="Macro softness",
            eps_guidance=GuidanceDirection.LOWER,
            eps_guidance_detail="",
            margin_outlook=MarginOutlook.CONTRACTING,
            capex_direction=CapexDirection.DECREASING,
            management_sentiment=-0.40,
            forward_language_ratio=0.30,
            risk_factor_count=4,
            key_risk_topics=["competition"],
            guidance_confidence=0.85,
            restructuring_signals=True,
        )

        with patch("earnings_intel.api.routes.GeminiExtractionClient") as mock_client_cls:
            mock_inst = mock_client_cls.return_value
            mock_inst._default_model = "gemini-2.5-flash"
            mock_inst.extract_signals.return_value = (
                mock_signals,
                {"prompt": 10, "completion": 20},
            )

            res = client.post("/api/filings/f_tsla_10q/extract")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["filing_id"] == "f_tsla_10q"
            assert data["signals"]["revenue_guidance"] == "lower"
            assert data["signals"]["restructuring_signals"] is True


class TestBacktestEndpoints:
    """Tests for PEAD and Portfolio backtesting routes."""

    def test_ticker_pead_unknown_ticker_returns_404(self, client: TestClient) -> None:
        """Unknown ticker returns 404."""
        res = client.get("/api/backtests/UNKNOWN_TICKER")
        assert res.status_code == 404

    def test_ticker_pead_with_benchmarks(
        self, client: TestClient, seeded_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Compute PEAD against default SPY and alternative QQQ/IWM benchmarks."""
        # Default SPY
        res_spy = client.get("/api/backtests/AAPL?benchmark=SPY&window_days=10")
        assert res_spy.status_code == 200
        pead_list = res_spy.json()
        assert len(pead_list) == 1
        assert pead_list[0]["ticker"] == "AAPL"
        assert pead_list[0]["raw_return"] > 0.0

        # Alternative benchmark QQQ
        res_qqq = client.get("/api/backtests/AAPL?benchmark=QQQ&window_days=10")
        assert res_qqq.status_code == 200
        assert len(res_qqq.json()) == 1

        # Alternative benchmark IWM
        res_iwm = client.get("/api/backtests/AAPL?benchmark=IWM&window_days=10")
        assert res_iwm.status_code == 200
        assert len(res_iwm.json()) == 1

    def test_portfolio_backtest_with_parameters(
        self, client: TestClient, seeded_db: duckdb.DuckDBPyConnection
    ) -> None:
        """Run portfolio backtest with configurable benchmark and filters."""
        res = client.get(
            "/api/backtests/portfolio?benchmark=SPY&window_days=10&min_sentiment=0.3&allow_short=true"
        )
        assert res.status_code == 200
        data = res.json()
        assert data["benchmark"] == "SPY"
        assert data["window_days"] == 10
        assert "metrics" in data
        assert "trades" in data
        assert len(data["trades"]) >= 1
        assert data["trades"][0]["ticker"] == "AAPL"
        assert data["trades"][0]["signal"] == "LONG"
