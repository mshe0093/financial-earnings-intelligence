"""Quantitative backtesting engine for SEC earnings signals.

Evaluates extracted Gemini signals against historical market prices to quantify
Post-Earnings Announcement Drift (PEAD), Cumulative Abnormal Return (CAR),
and portfolio performance metrics (Sharpe ratio, hit rate, max drawdown, Calmar ratio).
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from earnings_intel.backtesting.metrics import (
    PortfolioMetrics,
    calculate_portfolio_metrics,
)
from earnings_intel.backtesting.pead import (
    InsufficientPriceDataError,
    compute_pead,
    compute_pead_from_dataframes,
)
from earnings_intel.extraction.schemas import (
    ExtractedSignalRecord,
    GuidanceDirection,
    MarginOutlook,
)

logger = logging.getLogger(__name__)


class TradeSignal(StrEnum):
    """Trading action signal determined from filing analysis."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalFilter(BaseModel):
    """Configurable signal filtering criteria conforming to Section 7.3."""

    min_sentiment: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Sentiment threshold (+X for long, -X for short)",
    )
    min_forward_ratio: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum forward-language ratio for expansion long entry",
    )
    require_guidance_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum guidance confidence score required to act on signal",
    )
    allow_short: bool = Field(
        default=True,
        description="Whether short entries are enabled in the strategy",
    )

    def evaluate(self, signal: ExtractedSignalRecord | dict[str, Any]) -> tuple[TradeSignal, str]:
        """Evaluate signal against strategy rules.

        Rules from Section 7.3:
        * Long 1: Revenue guidance raised AND sentiment > +min_sentiment.
        * Long 2: Forward ratio > min_forward_ratio AND expanding margins.
        * Short 1: Revenue guidance lowered AND sentiment < -min_sentiment.
        * Short 2: Restructuring signals present AND contracting margins.

        Args:
            signal: ExtractedSignalRecord instance or dictionary with signal fields.

        Returns:
            Tuple of (TradeSignal, explanation_string).
        """
        # Convert to dictionary or attributes
        if isinstance(signal, ExtractedSignalRecord):
            rev_guidance = signal.revenue_guidance.value
            margin = signal.margin_outlook.value
            sentiment = float(signal.management_sentiment)
            forward_ratio = float(signal.forward_language_ratio)
            confidence = float(signal.guidance_confidence)
            restructuring = bool(signal.restructuring_signals)
        else:
            rev_raw = signal.get("revenue_guidance", "none")
            rev_guidance = (
                rev_raw.value if isinstance(rev_raw, GuidanceDirection) else str(rev_raw).lower()
            )
            margin_raw = signal.get("margin_outlook", "none")
            margin = (
                margin_raw.value
                if isinstance(margin_raw, MarginOutlook)
                else str(margin_raw).lower()
            )
            sentiment = float(signal.get("management_sentiment", 0.0) or 0.0)
            forward_ratio = float(signal.get("forward_language_ratio", 0.0) or 0.0)
            confidence = float(signal.get("guidance_confidence", 1.0) or 0.0)
            restructuring = bool(signal.get("restructuring_signals", False))

        if confidence < self.require_guidance_confidence:
            return (
                TradeSignal.NEUTRAL,
                f"Confidence {confidence:.2f} below threshold "
                f"{self.require_guidance_confidence:.2f}",
            )

        # Check Long Entry criteria
        if rev_guidance == GuidanceDirection.RAISE and sentiment > self.min_sentiment:
            return (
                TradeSignal.LONG,
                f"Revenue guidance raised with positive sentiment "
                f"({sentiment:+.2f} > {self.min_sentiment:.2f})",
            )
        if forward_ratio > self.min_forward_ratio and margin == MarginOutlook.EXPANDING:
            return (
                TradeSignal.LONG,
                f"High forward ratio ({forward_ratio:.2f} > {self.min_forward_ratio:.2f}) "
                f"with expanding margins",
            )

        # Check Short Entry criteria
        if self.allow_short:
            if rev_guidance == GuidanceDirection.LOWER and sentiment < -self.min_sentiment:
                return (
                    TradeSignal.SHORT,
                    f"Revenue guidance lowered with negative sentiment "
                    f"({sentiment:+.2f} < {-self.min_sentiment:.2f})",
                )
            if restructuring and margin == MarginOutlook.CONTRACTING:
                return (
                    TradeSignal.SHORT,
                    "Restructuring signals present with contracting margins",
                )

        return TradeSignal.NEUTRAL, "Signal did not satisfy long or short criteria"


class TradeRecord(BaseModel):
    """Detailed record of a single evaluated trade."""

    trade_id: str = Field(description="Unique trade execution identifier")
    filing_id: str = Field(description="Associated SEC filing identifier")
    ticker: str = Field(description="Stock ticker symbol")
    filing_date: date = Field(description="Filing submission date")
    signal: TradeSignal = Field(description="Executed signal: LONG or SHORT")
    signal_reason: str = Field(default="", description="Reason signal was triggered")
    entry_date: date = Field(description="Position entry date (first trading day post-filing)")
    exit_date: date = Field(description="Position exit date")
    holding_days: int = Field(description="Effective holding window in trading days")
    stock_return: float = Field(description="Raw stock price percentage return")
    benchmark_return: float = Field(description="Benchmark percentage return over same period")
    strategy_return: float = Field(
        description="Position strategy return (+stock_return for LONG, -stock_return for SHORT)"
    )
    abnormal_return: float = Field(
        description="Position abnormal return (+abnormal for LONG, -abnormal for SHORT)"
    )
    car_series: list[float] = Field(
        default_factory=list,
        description="Cumulative Abnormal Return series scaled by position direction",
    )


class BacktestReport(BaseModel):
    """Complete portfolio backtest summary and trade history."""

    run_id: str = Field(description="Unique backtest run identifier")
    run_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the backtest execution",
    )
    benchmark: str = Field(default="SPY", description="Market benchmark ticker used")
    window_days: int = Field(default=30, description="Target holding window in trading days")
    filter_config: SignalFilter = Field(description="SignalFilter parameters used")
    trades: list[TradeRecord] = Field(
        default_factory=list,
        description="List of all executed long/short trades",
    )
    metrics: PortfolioMetrics = Field(
        description="Consolidated portfolio risk and return statistics"
    )


class BacktestEngine:
    """Engine to backtest Gemini-extracted signals against historical market prices."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection | None = None,
        filter_config: SignalFilter | None = None,
        benchmark: str = "SPY",
        window_days: int = 30,
        risk_free_rate: float = 0.0,
    ) -> None:
        """Initialize BacktestEngine.

        Args:
            conn: DuckDB connection with `filings_metadata`, `extracted_signals`, `price_series`.
            filter_config: Signal filtering configuration (defaults to Section 7.3 rules).
            benchmark: Market benchmark ticker symbol (default: 'SPY').
            window_days: Holding window in trading days (default: 30).
            risk_free_rate: Annualized risk-free rate for Sharpe ratio (default: 0.0).
        """
        self.conn = conn
        self.filter_config = filter_config or SignalFilter()
        self.benchmark = benchmark.strip().upper()
        self.window_days = window_days
        self.risk_free_rate = risk_free_rate

    def evaluate_filing(
        self,
        filing_id: str,
        ticker: str,
        filing_date: date,
        signal_data: ExtractedSignalRecord | dict[str, Any],
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> TradeRecord | None:
        """Evaluate an individual filing signal and calculate trade returns if actionable.

        Args:
            filing_id: Filing identifier.
            ticker: Stock ticker.
            filing_date: Date of filing.
            signal_data: Extracted signals.
            conn: Optional DuckDB connection override.

        Returns:
            `TradeRecord` if signal is LONG or SHORT and price data exists; None if NEUTRAL
            or insufficient price history.
        """
        signal_type, reason = self.filter_config.evaluate(signal_data)
        if signal_type == TradeSignal.NEUTRAL:
            return None

        active_conn = conn or self.conn
        if active_conn is None:
            raise ValueError("DuckDB connection is required to compute price returns.")

        try:
            pead = compute_pead(
                filing_id=filing_id,
                filing_date=filing_date,
                ticker=ticker,
                conn=active_conn,
                benchmark_ticker=self.benchmark,
                window_days=self.window_days,
            )
        except InsufficientPriceDataError as e:
            logger.warning(
                "Skipping filing %s (%s): %s",
                filing_id,
                ticker,
                e,
            )
            return None

        # Calculate directional strategy returns
        is_long = signal_type == TradeSignal.LONG
        strat_return = pead.raw_return if is_long else -pead.raw_return
        ab_return = pead.abnormal_return if is_long else -pead.abnormal_return
        car_series = pead.car_series if is_long else [-val for val in pead.car_series]

        return TradeRecord(
            trade_id=f"trade_{uuid.uuid4().hex[:12]}",
            filing_id=filing_id,
            ticker=ticker.upper(),
            filing_date=filing_date,
            signal=signal_type,
            signal_reason=reason,
            entry_date=pead.start_date,
            exit_date=pead.end_date,
            holding_days=pead.window_days,
            stock_return=pead.raw_return,
            benchmark_return=pead.benchmark_return,
            strategy_return=round(strat_return, 6),
            abnormal_return=round(ab_return, 6),
            car_series=[round(c, 6) for c in car_series],
        )

    def run(
        self,
        tickers: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestReport:
        """Run backtest on filings stored in DuckDB.

        Args:
            tickers: Optional list of tickers to filter on.
            start_date: Optional earliest filing_date filter.
            end_date: Optional latest filing_date filter.

        Returns:
            Consolidated `BacktestReport`.
        """
        if self.conn is None:
            raise ValueError("DuckDB connection is required to run backtest.")

        query = """
            SELECT
                f.filing_id,
                f.ticker,
                f.filing_date,
                s.revenue_guidance,
                s.revenue_guidance_detail,
                s.eps_guidance,
                s.eps_guidance_detail,
                s.margin_outlook,
                s.capex_direction,
                s.management_sentiment,
                s.forward_language_ratio,
                s.risk_factor_count,
                s.guidance_confidence,
                s.restructuring_signals
            FROM filings_metadata f
            INNER JOIN extracted_signals s ON f.filing_id = s.filing_id
            WHERE f.ticker IS NOT NULL
        """
        params: list[Any] = []

        if tickers:
            clean_tickers = [t.strip().upper() for t in tickers]
            placeholders = ", ".join(["?"] * len(clean_tickers))
            query += f" AND f.ticker IN ({placeholders})"
            params.extend(clean_tickers)

        if start_date:
            query += " AND f.filing_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND f.filing_date <= ?"
            params.append(end_date)

        query += " ORDER BY f.filing_date ASC"

        df_filings = self.conn.execute(query, params).df()

        trades: list[TradeRecord] = []
        for _, row in df_filings.iterrows():
            f_id = str(row["filing_id"])
            t_ticker = str(row["ticker"])
            f_date = (
                row["filing_date"].date()
                if hasattr(row["filing_date"], "date")
                else pd.to_datetime(row["filing_date"]).date()
            )
            sig_dict = dict(row)

            trade = self.evaluate_filing(
                filing_id=f_id,
                ticker=t_ticker,
                filing_date=f_date,
                signal_data=sig_dict,
                conn=self.conn,
            )
            if trade is not None:
                trades.append(trade)

        strategy_returns = [t.strategy_return for t in trades]
        abnormal_returns = [t.abnormal_return for t in trades]
        metrics = calculate_portfolio_metrics(
            strategy_returns=strategy_returns,
            abnormal_returns=abnormal_returns,
            risk_free_rate=self.risk_free_rate,
            holding_period_days=self.window_days,
        )

        return BacktestReport(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            run_timestamp=datetime.utcnow(),
            benchmark=self.benchmark,
            window_days=self.window_days,
            filter_config=self.filter_config,
            trades=trades,
            metrics=metrics,
        )

    def run_on_records(
        self,
        filings_with_signals: list[dict[str, Any]],
        price_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestReport:
        """Run backtest on preloaded dictionaries and DataFrames without requiring SQL queries.

        Useful for unit tests and offline simulations.

        Args:
            filings_with_signals: List of filing dicts containing `filing_id`, `ticker`,
                `filing_date`, and signal metrics.
            price_dfs: Mapping of ticker to price DataFrame with columns
                `['trade_date', 'adj_close']`. Must include benchmark ticker (e.g. 'SPY').

        Returns:
            `BacktestReport`.
        """
        price_dfs = price_dfs or {}
        bench_df = price_dfs.get(self.benchmark)
        if bench_df is None and filings_with_signals:
            raise ValueError(f"Price DataFrame for benchmark '{self.benchmark}' must be provided.")

        trades: list[TradeRecord] = []
        for record in filings_with_signals:
            filing_id = str(record["filing_id"])
            ticker = str(record["ticker"]).strip().upper()
            raw_date = record["filing_date"]
            f_date = (
                raw_date.date() if hasattr(raw_date, "date") else pd.to_datetime(raw_date).date()
            )

            signal_type, reason = self.filter_config.evaluate(record)
            if signal_type == TradeSignal.NEUTRAL:
                continue

            ticker_df = price_dfs.get(ticker)
            if ticker_df is None:
                logger.warning("No price DataFrame provided for %s; skipping.", ticker)
                continue

            if bench_df is None:
                break

            try:
                pead = compute_pead_from_dataframes(
                    filing_id=filing_id,
                    filing_date=f_date,
                    ticker=ticker,
                    ticker_df=ticker_df,
                    benchmark_df=bench_df,
                    window_days=self.window_days,
                )
            except InsufficientPriceDataError as e:
                logger.warning("Insufficient prices for %s: %s", ticker, e)
                continue

            is_long = signal_type == TradeSignal.LONG
            strat_return = pead.raw_return if is_long else -pead.raw_return
            ab_return = pead.abnormal_return if is_long else -pead.abnormal_return
            car_series = pead.car_series if is_long else [-val for val in pead.car_series]

            trades.append(
                TradeRecord(
                    trade_id=f"trade_{uuid.uuid4().hex[:12]}",
                    filing_id=filing_id,
                    ticker=ticker,
                    filing_date=f_date,
                    signal=signal_type,
                    signal_reason=reason,
                    entry_date=pead.start_date,
                    exit_date=pead.end_date,
                    holding_days=pead.window_days,
                    stock_return=pead.raw_return,
                    benchmark_return=pead.benchmark_return,
                    strategy_return=round(strat_return, 6),
                    abnormal_return=round(ab_return, 6),
                    car_series=[round(c, 6) for c in car_series],
                )
            )

        strategy_returns = [t.strategy_return for t in trades]
        abnormal_returns = [t.abnormal_return for t in trades]
        metrics = calculate_portfolio_metrics(
            strategy_returns=strategy_returns,
            abnormal_returns=abnormal_returns,
            risk_free_rate=self.risk_free_rate,
            holding_period_days=self.window_days,
        )

        return BacktestReport(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            run_timestamp=datetime.utcnow(),
            benchmark=self.benchmark,
            window_days=self.window_days,
            filter_config=self.filter_config,
            trades=trades,
            metrics=metrics,
        )

    def save_report(
        self,
        report: BacktestReport,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> str:
        """Persist backtest run and executed trades into DuckDB.

        Args:
            report: BacktestReport instance to save.
            conn: DuckDB connection override (uses self.conn if None).

        Returns:
            The saved run_id string.
        """
        active_conn = conn or self.conn
        if active_conn is None:
            raise ValueError("DuckDB connection is required to persist backtest report.")

        active_conn.begin()
        try:
            active_conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, run_timestamp, benchmark, window_days, total_trades,
                    winning_trades, losing_trades, hit_rate, mean_return,
                    mean_abnormal_return, annualized_return, annualized_volatility,
                    sharpe_ratio, max_drawdown, calmar_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    report.run_id,
                    report.run_timestamp,
                    report.benchmark,
                    report.window_days,
                    report.metrics.total_trades,
                    report.metrics.winning_trades,
                    report.metrics.losing_trades,
                    report.metrics.hit_rate,
                    report.metrics.mean_return,
                    report.metrics.mean_abnormal_return,
                    report.metrics.annualized_return,
                    report.metrics.annualized_volatility,
                    report.metrics.sharpe_ratio,
                    report.metrics.max_drawdown,
                    report.metrics.calmar_ratio,
                ],
            )

            for t in report.trades:
                active_conn.execute(
                    """
                    INSERT INTO backtest_trades (
                        trade_id, run_id, filing_id, ticker, filing_date, signal,
                        signal_reason, entry_date, exit_date, holding_days,
                        stock_return, benchmark_return, strategy_return, abnormal_return
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        t.trade_id,
                        report.run_id,
                        t.filing_id,
                        t.ticker,
                        t.filing_date,
                        t.signal.value,
                        t.signal_reason,
                        t.entry_date,
                        t.exit_date,
                        t.holding_days,
                        t.stock_return,
                        t.benchmark_return,
                        t.strategy_return,
                        t.abnormal_return,
                    ],
                )
            active_conn.commit()
            return report.run_id
        except Exception:
            active_conn.rollback()
            raise
