"""Unit tests for Gradio UI construction and Plotly visualization generation."""

from __future__ import annotations

from datetime import date

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import pytest

from earnings_intel.api.ui import (
    build_metrics_summary_markdown,
    create_drawdown_chart,
    create_equity_curve_chart,
    create_pead_car_chart,
    create_sentiment_price_chart,
    create_ui,
    get_demo_backtest_data,
)
from earnings_intel.backtesting.engine import TradeRecord, TradeSignal
from earnings_intel.backtesting.metrics import PortfolioMetrics


class TestPlotlyFigures:
    """Verify Plotly chart creation and trace structures."""

    def test_equity_curve_chart_valid(self) -> None:
        """Create equity curve figure with data."""
        eq = [1.0, 1.05, 1.02, 1.10]
        fig = create_equity_curve_chart(eq, "SPY")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].name == "Earnings Intel Strategy"
        assert list(fig.data[0].y) == eq

    def test_equity_curve_chart_empty(self) -> None:
        """Empty equity curve falls back safely to [1.0]."""
        fig = create_equity_curve_chart([], "SPY")
        assert isinstance(fig, go.Figure)
        assert list(fig.data[0].y) == [1.0]

    def test_drawdown_chart_valid(self) -> None:
        """Create underwater drawdown chart."""
        eq = [1.0, 1.2, 0.9, 1.1]
        fig = create_drawdown_chart(eq)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].fill == "tozeroy"
        # Minimum value in drawdown should be -25%
        assert float(min(fig.data[0].y)) == pytest.approx(-0.25, abs=1e-4)

    def test_pead_car_chart_valid(self) -> None:
        """Create PEAD Cumulative Abnormal Return event study chart."""
        trades = [
            TradeRecord(
                trade_id="t1",
                filing_id="f1",
                ticker="AAPL",
                filing_date=date(2023, 1, 1),
                signal=TradeSignal.LONG,
                entry_date=date(2023, 1, 2),
                exit_date=date(2023, 1, 15),
                holding_days=10,
                stock_return=0.05,
                benchmark_return=0.01,
                strategy_return=0.05,
                abnormal_return=0.04,
                car_series=[0.0, 0.01, 0.02, 0.04],
            ),
            TradeRecord(
                trade_id="t2",
                filing_id="f2",
                ticker="TSLA",
                filing_date=date(2023, 1, 1),
                signal=TradeSignal.SHORT,
                entry_date=date(2023, 1, 2),
                exit_date=date(2023, 1, 15),
                holding_days=10,
                stock_return=-0.06,
                benchmark_return=0.01,
                strategy_return=0.06,
                abnormal_return=0.07,
                car_series=[0.0, 0.02, 0.05, 0.07],
            ),
        ]
        fig = create_pead_car_chart(trades, window_days=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # One long trace, one short trace
        trace_names = [d.name for d in fig.data]
        assert any("Long Signals" in name for name in trace_names)
        assert any("Short Signals" in name for name in trace_names)

    def test_sentiment_price_chart_dual_axis(self) -> None:
        """Dual-axis chart displays both stock price and sentiment markers."""
        dates = [date(2023, 1, 10), date(2023, 4, 15)]
        sentiments = [0.45, -0.20]
        price_df = pd.DataFrame(
            {
                "trade_date": [date(2023, 1, 10), date(2023, 2, 10), date(2023, 4, 15)],
                "adj_close": [150.0, 160.0, 155.0],
            }
        )

        fig = create_sentiment_price_chart("AAPL", dates, sentiments, price_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.data[0].yaxis in ("y", "y1")
        assert fig.data[1].yaxis == "y2"


class TestDemoDataAndMarkdown:
    """Verify demo data generator and markdown formatting."""

    def test_get_demo_backtest_data(self) -> None:
        """Generates valid BacktestReport and DataFrame for demo exploration."""
        report, df_trades = get_demo_backtest_data(
            benchmark="QQQ",
            window_days=15,
            min_sentiment=0.25,
            min_forward_ratio=0.35,
            allow_short=True,
        )
        assert report.benchmark == "QQQ"
        assert report.window_days == 15
        assert isinstance(df_trades, pd.DataFrame)
        assert not df_trades.empty
        assert "Ticker" in df_trades.columns
        assert "Strategy Return" in df_trades.columns

    def test_build_metrics_summary_markdown(self) -> None:
        """Formats KPI summary table in Markdown."""
        m = PortfolioMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            hit_rate=0.60,
            mean_return=0.035,
            mean_abnormal_return=0.021,
            annualized_return=0.185,
            annualized_volatility=0.12,
            sharpe_ratio=1.45,
            max_drawdown=-0.08,
            calmar_ratio=2.31,
            equity_curve=[1.0, 1.05, 1.10],
        )
        md = build_metrics_summary_markdown(m, "SPY")
        assert "1.45" in md
        assert "60.0%" in md
        assert "-8.00%" in md
        assert "SPY" in md


class TestGradioAppConstruction:
    """Verify Gradio application initialization."""

    def test_create_ui_returns_blocks(self) -> None:
        """Gradio app builds without errors."""
        ui = create_ui()
        assert isinstance(ui, gr.Blocks)
