"""CLI script to run quantitative backtests evaluating earnings signals against market prices.

Evaluates Gemini-extracted MD&A signals stored in DuckDB against historical prices,
computes 30-day Post-Earnings Announcement Drift (PEAD), Cumulative Abnormal Return (CAR),
and portfolio performance metrics (Sharpe ratio, hit rate, max drawdown, Calmar ratio).

Usage examples:
    python scripts/backtest.py --benchmark SPY --window 30
    python scripts/backtest.py --ticker AAPL --window 30
    python scripts/backtest.py --min-sentiment 0.25 --no-shorts
    python scripts/backtest.py --demo
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from earnings_intel.backtesting.engine import (
    BacktestEngine,
    BacktestReport,
    SignalFilter,
)
from earnings_intel.config import get_settings
from earnings_intel.db.connection import get_connection
from earnings_intel.db.schema import initialize_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("earnings_intel.backtest")


def create_demo_data() -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Generate realistic synthetic filings and prices for instant demonstration."""
    base_date = date(2023, 1, 15)
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX"]

    # 1. Generate 120 trading days of synthetic prices
    dates = [base_date + timedelta(days=i) for i in range(120)]
    price_dfs: dict[str, pd.DataFrame] = {}

    rng = np.random.default_rng(42)

    # Benchmark: slight steady upward trend
    bench_prices = [400.0]
    for _ in range(1, len(dates)):
        ret = rng.normal(0.0004, 0.008)
        bench_prices.append(round(bench_prices[-1] * (1.0 + ret), 2))
    price_dfs["SPY"] = pd.DataFrame({"trade_date": dates, "adj_close": bench_prices})

    for tick in tickers:
        p = 100.0 + rng.uniform(20.0, 80.0)
        t_prices = [p]
        for _ in range(1, len(dates)):
            ret = rng.normal(0.0006, 0.015)
            t_prices.append(round(t_prices[-1] * (1.0 + ret), 2))
        price_dfs[tick] = pd.DataFrame({"trade_date": dates, "adj_close": t_prices})

    # 2. Generate synthetic filings with varied signals
    filings = [
        {
            "filing_id": f"filing_{i}",
            "ticker": tickers[i % len(tickers)],
            "filing_date": base_date + timedelta(days=i * 8),
            "revenue_guidance": "raise" if i % 3 == 0 else ("lower" if i % 3 == 1 else "maintain"),
            "margin_outlook": "expanding" if i % 2 == 0 else "contracting",
            "management_sentiment": 0.45 if i % 3 == 0 else (-0.38 if i % 3 == 1 else 0.05),
            "forward_language_ratio": 0.48 if i % 2 == 0 else 0.22,
            "guidance_confidence": 0.88,
            "restructuring_signals": (i % 3 == 1),
        }
        for i in range(10)
    ]

    return filings, price_dfs


def print_backtest_report(report: BacktestReport) -> None:
    """Print a clean, professional summary table of backtest results."""
    m = report.metrics

    print("\n" + "=" * 70)
    print("      QUANTITATIVE EARNINGS INTELLIGENCE BACKTEST REPORT      ")
    print("=" * 70)
    print(f"Run ID:            {report.run_id}")
    print(f"Timestamp:         {report.run_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Benchmark:         {report.benchmark}")
    print(f"Holding Window:    {report.window_days} trading days")
    print(f"Min Sentiment:     {report.filter_config.min_sentiment:+.2f}")
    print(f"Min Forward Ratio: {report.filter_config.min_forward_ratio:.2f}")
    print(f"Allow Shorting:    {report.filter_config.allow_short}")
    print("-" * 70)
    print("PORTFOLIO PERFORMANCE & RISK METRICS:")
    print(f"  Total Trades:           {m.total_trades}")
    print(f"  Winning Trades:         {m.winning_trades} ({m.hit_rate * 100:.1f}% hit rate)")
    print(f"  Losing Trades:          {m.losing_trades}")
    print(f"  Mean Return:            {m.mean_return * 100:+.2f}%")
    print(f"  Mean Abnormal Return:   {m.mean_abnormal_return * 100:+.2f}% vs {report.benchmark}")
    print(f"  Annualized Return:      {m.annualized_return * 100:+.2f}%")
    print(f"  Annualized Volatility:  {m.annualized_volatility * 100:.2f}%")
    print(f"  Annualized Sharpe:      {m.sharpe_ratio:+.2f}")
    print(f"  Maximum Drawdown:       {m.max_drawdown * 100:.2f}%")
    print(f"  Calmar Ratio:           {m.calmar_ratio:.2f}")
    print("-" * 70)

    if report.trades:
        print("EXECUTED TRADES SUMMARY:")
        header = (
            f"  {'TICKER':<7} {'SIGNAL':<7} {'ENTRY DATE':<11} "
            f"{'STOCK RET':<10} {'STRAT RET':<10} {'ABNORMAL RET':<12}"
        )
        print(header)
        for t in report.trades[:15]:
            sig_str = t.signal.value
            stock_ret = f"{t.stock_return * 100:+.2f}%"
            strat_ret = f"{t.strategy_return * 100:+.2f}%"
            ab_ret = f"{t.abnormal_return * 100:+.2f}%"
            print(
                f"  {t.ticker:<7} {sig_str:<7} {t.entry_date.isoformat():<11} "
                f"{stock_ret:<10} {strat_ret:<10} {ab_ret:<12}"
            )
        if len(report.trades) > 15:
            print(f"  ... and {len(report.trades) - 15} more trades.")

    print("-" * 70)
    print("EQUITY CURVE PROGRESSION (normalized to 1.0):")
    sample_curve = report.metrics.equity_curve
    step = max(1, len(sample_curve) // 6)
    pts = [f"{sample_curve[i]:.3f}" for i in range(0, len(sample_curve), step)]
    if str(sample_curve[-1]) not in pts[-1]:
        pts.append(f"{sample_curve[-1]:.3f}")
    print("  " + " -> ".join(pts))
    print("=" * 70 + "\n")


def main() -> None:
    """CLI entrypoint for running earnings backtesting."""
    parser = argparse.ArgumentParser(
        description="Run quantitative backtests evaluating SEC earnings signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker (e.g. SPY).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Holding window in trading days.",
    )
    parser.add_argument(
        "--min-sentiment",
        type=float,
        default=0.3,
        help="Minimum absolute sentiment threshold (+X for Long, -X for Short).",
    )
    parser.add_argument(
        "--min-forward-ratio",
        type=float,
        default=0.4,
        help="Minimum forward-language ratio.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional single ticker filter (e.g. AAPL).",
    )
    parser.add_argument(
        "--no-shorts",
        action="store_true",
        help="Disable short positions (Long-only strategy).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist backtest run and trades into DuckDB.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic demo data to demonstrate engine functionality.",
    )

    args = parser.parse_args()

    filter_config = SignalFilter(
        min_sentiment=args.min_sentiment,
        min_forward_ratio=args.min_forward_ratio,
        allow_short=not args.no_shorts,
    )

    if args.demo:
        logger.info("Executing backtest on demonstration dataset...")
        demo_filings, demo_prices = create_demo_data()
        engine = BacktestEngine(
            conn=None,
            filter_config=filter_config,
            benchmark=args.benchmark,
            window_days=args.window,
        )
        report = engine.run_on_records(demo_filings, demo_prices)
        print_backtest_report(report)
        return

    # Normal mode: query DuckDB
    settings = get_settings()
    with get_connection(settings) as conn:
        initialize_schema(conn)

        filing_count = conn.execute(
            "SELECT count(*) FROM filings_metadata f "
            "INNER JOIN extracted_signals s ON f.filing_id = s.filing_id"
        ).fetchone()[0]

        price_count = conn.execute("SELECT count(*) FROM price_series").fetchone()[0]

        logger.info(
            "DuckDB contains %d extracted filings and %d price records.",
            filing_count,
            price_count,
        )

        if filing_count == 0 or price_count == 0:
            logger.warning(
                "Insufficient filings (%d) or prices (%d) in %s. "
                "Falling back to demonstration data to show engine output.",
                filing_count,
                price_count,
                settings.duckdb_path,
            )
            demo_filings, demo_prices = create_demo_data()
            engine = BacktestEngine(
                conn=None,
                filter_config=filter_config,
                benchmark=args.benchmark,
                window_days=args.window,
            )
            report = engine.run_on_records(demo_filings, demo_prices)
            print_backtest_report(report)
            return

        engine = BacktestEngine(
            conn=conn,
            filter_config=filter_config,
            benchmark=args.benchmark,
            window_days=args.window,
        )

        tickers = [args.ticker.strip().upper()] if args.ticker else None
        report = engine.run(tickers=tickers)

        if args.save:
            engine.save_report(report, conn=conn)
            logger.info("Saved backtest report %s to DuckDB.", report.run_id)

        print_backtest_report(report)


if __name__ == "__main__":
    main()
