"""Tests for the BacktestEngine, SignalFilter, and DuckDB storage."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from earnings_intel.backtesting.engine import (
    BacktestEngine,
    BacktestReport,
    SignalFilter,
    TradeSignal,
)
from earnings_intel.extraction.schemas import (
    CapexDirection,
    ExtractedSignalRecord,
    GuidanceDirection,
    MarginOutlook,
)


@pytest.fixture
def sample_signal_record() -> ExtractedSignalRecord:
    """Fixture providing a valid ExtractedSignalRecord."""
    return ExtractedSignalRecord(
        signal_id="sig_test_1",
        filing_id="filing_test_1",
        extraction_model="gemini-2.5-flash",
        revenue_guidance=GuidanceDirection.RAISE,
        revenue_guidance_detail="Raised full year revenue expectation",
        eps_guidance=GuidanceDirection.RAISE,
        eps_guidance_detail="",
        margin_outlook=MarginOutlook.EXPANDING,
        capex_direction=CapexDirection.STABLE,
        management_sentiment=0.45,
        forward_language_ratio=0.52,
        risk_factor_count=3,
        key_risk_topics=["competition"],
        guidance_confidence=0.90,
        restructuring_signals=False,
    )


class TestSignalFilter:
    """Tests for strategy signal filtering logic."""

    def test_long_rule_revenue_and_sentiment(
        self, sample_signal_record: ExtractedSignalRecord
    ) -> None:
        """Revenue guidance raised + sentiment > 0.3 -> LONG."""
        flt = SignalFilter(min_sentiment=0.3)
        signal, reason = flt.evaluate(sample_signal_record)
        assert signal == TradeSignal.LONG
        assert "Revenue guidance raised" in reason

    def test_long_rule_forward_ratio_and_margin(self) -> None:
        """Forward ratio > 0.4 + expanding margins -> LONG."""
        flt = SignalFilter(min_forward_ratio=0.4)
        data = {
            "revenue_guidance": "maintain",
            "management_sentiment": 0.1,
            "forward_language_ratio": 0.45,
            "margin_outlook": "expanding",
            "restructuring_signals": False,
        }
        signal, reason = flt.evaluate(data)
        assert signal == TradeSignal.LONG
        assert "High forward ratio" in reason

    def test_short_rule_revenue_lower_and_sentiment(self) -> None:
        """Revenue guidance lowered + negative sentiment (<-0.3) -> SHORT."""
        flt = SignalFilter(min_sentiment=0.3)
        data = {
            "revenue_guidance": "lower",
            "management_sentiment": -0.40,
            "forward_language_ratio": 0.2,
            "margin_outlook": "contracting",
            "restructuring_signals": False,
        }
        signal, reason = flt.evaluate(data)
        assert signal == TradeSignal.SHORT
        assert "Revenue guidance lowered" in reason

    def test_short_rule_restructuring_and_contracting(self) -> None:
        """Restructuring signals + contracting margins -> SHORT."""
        flt = SignalFilter()
        data = {
            "revenue_guidance": "maintain",
            "management_sentiment": -0.1,
            "forward_language_ratio": 0.15,
            "margin_outlook": "contracting",
            "restructuring_signals": True,
        }
        signal, reason = flt.evaluate(data)
        assert signal == TradeSignal.SHORT
        assert "Restructuring signals present" in reason

    def test_no_shorts_disables_shorting(self) -> None:
        """When allow_short=False, short signals become NEUTRAL."""
        flt = SignalFilter(allow_short=False)
        data = {
            "revenue_guidance": "lower",
            "management_sentiment": -0.40,
            "margin_outlook": "contracting",
            "restructuring_signals": True,
        }
        signal, _ = flt.evaluate(data)
        assert signal == TradeSignal.NEUTRAL

    def test_confidence_threshold_rejection(self) -> None:
        """Low guidance confidence drops signal to NEUTRAL."""
        flt = SignalFilter(require_guidance_confidence=0.85)
        data = {
            "revenue_guidance": "raise",
            "management_sentiment": 0.50,
            "guidance_confidence": 0.60,
        }
        signal, reason = flt.evaluate(data)
        assert signal == TradeSignal.NEUTRAL
        assert "Confidence" in reason

    def test_neutral_fallback(self) -> None:
        """Signal that meets neither condition is NEUTRAL."""
        flt = SignalFilter()
        data = {
            "revenue_guidance": "maintain",
            "management_sentiment": 0.05,
            "forward_language_ratio": 0.2,
            "margin_outlook": "stable",
            "restructuring_signals": False,
        }
        signal, _ = flt.evaluate(data)
        assert signal == TradeSignal.NEUTRAL


class TestBacktestEngineInMemory:
    """Tests for BacktestEngine using run_on_records."""

    def test_run_on_records_executes_trades(self) -> None:
        """Engine processes filings and price DataFrames to compute metrics."""
        base_date = date(2023, 1, 15)
        dates = [base_date + timedelta(days=i) for i in range(40)]

        # Price data: SPY flat, AAPL up 10%, TSLA down 10%
        spy_prices = [400.0 for _ in dates]
        aapl_prices = [100.0 * (1.0 + (0.0025 * i)) for i in range(len(dates))]
        tsla_prices = [200.0 * (1.0 - (0.0025 * i)) for i in range(len(dates))]

        price_dfs = {
            "SPY": pd.DataFrame({"trade_date": dates, "adj_close": spy_prices}),
            "AAPL": pd.DataFrame({"trade_date": dates, "adj_close": aapl_prices}),
            "TSLA": pd.DataFrame({"trade_date": dates, "adj_close": tsla_prices}),
        }

        filings = [
            # Long AAPL (should be profitable)
            {
                "filing_id": "filing_aapl",
                "ticker": "AAPL",
                "filing_date": base_date,
                "revenue_guidance": "raise",
                "margin_outlook": "expanding",
                "management_sentiment": 0.40,
                "forward_language_ratio": 0.45,
                "restructuring_signals": False,
            },
            # Short TSLA (stock dropped, short position should be profitable!)
            {
                "filing_id": "filing_tsla",
                "ticker": "TSLA",
                "filing_date": base_date,
                "revenue_guidance": "lower",
                "margin_outlook": "contracting",
                "management_sentiment": -0.35,
                "forward_language_ratio": 0.15,
                "restructuring_signals": True,
            },
            # Neutral filing (ignored)
            {
                "filing_id": "filing_neutral",
                "ticker": "AAPL",
                "filing_date": base_date,
                "revenue_guidance": "maintain",
                "management_sentiment": 0.0,
                "forward_language_ratio": 0.1,
                "margin_outlook": "stable",
                "restructuring_signals": False,
            },
        ]

        engine = BacktestEngine(benchmark="SPY", window_days=20)
        report = engine.run_on_records(filings, price_dfs)

        assert isinstance(report, BacktestReport)
        assert len(report.trades) == 2

        # Both trades should be winning
        assert report.trades[0].ticker == "AAPL"
        assert report.trades[0].signal == TradeSignal.LONG
        assert report.trades[0].strategy_return > 0.0

        assert report.trades[1].ticker == "TSLA"
        assert report.trades[1].signal == TradeSignal.SHORT
        assert report.trades[1].strategy_return > 0.0

        assert report.metrics.total_trades == 2
        assert report.metrics.winning_trades == 2
        assert report.metrics.hit_rate == 1.0


class TestBacktestEngineDuckDB:
    """Tests for BacktestEngine querying DuckDB tables and persisting results."""

    def test_run_and_save_report_in_duckdb(self, memory_db: duckdb.DuckDBPyConnection) -> None:
        """End-to-end backtest querying and persisting to DuckDB."""
        filing_date = date(2023, 2, 1)

        # 1. Insert filing metadata
        memory_db.execute(
            """
            INSERT INTO filings_metadata (
                filing_id, cik, ticker, company_name, form_type,
                filing_date, period_of_report, accession_number, primary_doc_url
            ) VALUES (
                'f_nvda_2023', '0001045810', 'NVDA', 'NVIDIA CORP', '10-K',
                ?, ?, 'acc_123', 'https://sec.gov/doc'
            )
            """,
            [filing_date, filing_date],
        )

        # 2. Insert extracted signals (Long signal)
        memory_db.execute(
            """
            INSERT INTO extracted_signals (
                signal_id, filing_id, extraction_model, revenue_guidance,
                revenue_guidance_detail, eps_guidance, eps_guidance_detail,
                margin_outlook, capex_direction, management_sentiment,
                forward_language_ratio, risk_factor_count, guidance_confidence,
                restructuring_signals
            ) VALUES (
                'sig_nvda_1', 'f_nvda_2023', 'gemini-2.5-flash', 'raise',
                'Strong AI demand', 'raise', '', 'expanding', 'increasing',
                0.55, 0.48, 2, 0.95, false
            )
            """
        )

        # 3. Insert price series for NVDA and SPY
        for i in range(15):
            d = filing_date + timedelta(days=i)
            memory_db.execute(
                """
                INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES ('NVDA', ?, ?, ?)
                """,
                [d, 150.0 + (i * 2.0), 150.0 + (i * 2.0)],
            )
            memory_db.execute(
                """
                INSERT INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES ('SPY', ?, ?, ?)
                """,
                [d, 410.0 + (i * 0.5), 410.0 + (i * 0.5)],
            )

        # 4. Run BacktestEngine
        engine = BacktestEngine(conn=memory_db, benchmark="SPY", window_days=10)
        report = engine.run()

        assert len(report.trades) == 1
        assert report.trades[0].ticker == "NVDA"
        assert report.trades[0].signal == TradeSignal.LONG
        assert report.metrics.total_trades == 1
        assert report.metrics.winning_trades == 1

        # 5. Persist report to DuckDB
        saved_id = engine.save_report(report)
        assert saved_id == report.run_id

        # Verify DB records
        row_run = memory_db.execute(
            "SELECT run_id, total_trades, hit_rate FROM backtest_runs WHERE run_id = ?",
            [saved_id],
        ).fetchone()
        assert row_run is not None
        assert row_run[0] == saved_id
        assert row_run[1] == 1
        assert row_run[2] == 1.0

        row_trade = memory_db.execute(
            "SELECT trade_id, ticker, signal FROM backtest_trades WHERE run_id = ?",
            [saved_id],
        ).fetchone()
        assert row_trade is not None
        assert row_trade[1] == "NVDA"
        assert row_trade[2] == "LONG"
