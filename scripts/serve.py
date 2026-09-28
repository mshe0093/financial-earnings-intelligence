"""Launch script for the Financial Earnings Intelligence web server.

Serves the FastAPI backend and Gradio interactive dashboard locally.

Usage examples:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --reload
    python scripts/serve.py --demo
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

import duckdb
import numpy as np
import uvicorn

from earnings_intel.config import get_settings
from earnings_intel.db.connection import get_connection
from earnings_intel.db.schema import initialize_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("earnings_intel.serve")


def seed_demo_data_if_empty(conn: duckdb.DuckDBPyConnection) -> None:
    """Populate DuckDB with realistic demo filings, signals, and prices if empty."""
    f_count = conn.execute("SELECT count(*) FROM filings_metadata").fetchone()[0]
    p_count = conn.execute("SELECT count(*) FROM price_series").fetchone()[0]

    if f_count > 0 and p_count > 0:
        logger.info("DuckDB already populated with %d filings and %d prices.", f_count, p_count)
        return

    logger.info("Seeding demonstration dataset into DuckDB...")
    base_date = date(2023, 1, 10)
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
    benchmarks = ["SPY", "QQQ", "IWM"]

    # 1. Seed prices (150 trading days)
    rng = np.random.default_rng(42)
    dates = [base_date + timedelta(days=i) for i in range(150)]

    for bench in benchmarks:
        b_p0 = 400.0 if bench == "SPY" else (350.0 if bench == "QQQ" else 180.0)
        curr = b_p0
        for d in dates:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES (?, ?, ?, ?)
                """,
                [bench, d, round(curr, 2), round(curr, 2)],
            )
            curr *= 1.0 + rng.normal(0.0004, 0.008)

    for tick in tickers:
        curr = 120.0 + rng.uniform(10.0, 100.0)
        for d in dates:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_series (ticker, trade_date, close_price, adj_close)
                VALUES (?, ?, ?, ?)
                """,
                [tick, d, round(curr, 2), round(curr, 2)],
            )
            curr *= 1.0 + rng.normal(0.0006, 0.015)

    # 2. Seed filings & extracted signals
    sample_filings = [
        (
            "0000320193_0000320193-23-000106",
            "0000320193",
            "AAPL",
            "Apple Inc.",
            "10-K",
            0,
            "raise",
            0.42,
            0.45,
            "expanding",
            False,
        ),
        (
            "0000789019_0000789019-23-000085",
            "0000789019",
            "MSFT",
            "Microsoft Corp",
            "10-K",
            14,
            "raise",
            0.55,
            0.52,
            "expanding",
            False,
        ),
        (
            "0001045810_0001045810-23-000045",
            "0001045810",
            "NVDA",
            "NVIDIA CORP",
            "10-Q",
            28,
            "raise",
            0.68,
            0.58,
            "expanding",
            False,
        ),
        (
            "0001018724_0001018724-23-000050",
            "0001018724",
            "AMZN",
            "Amazon Com Inc",
            "10-Q",
            42,
            "maintain",
            0.15,
            0.35,
            "stable",
            False,
        ),
        (
            "0001652044_0001652044-23-000030",
            "0001652044",
            "GOOGL",
            "Alphabet Inc.",
            "10-Q",
            56,
            "raise",
            0.38,
            0.41,
            "expanding",
            False,
        ),
        (
            "0001318605_0001318605-23-000050",
            "0001318605",
            "TSLA",
            "Tesla, Inc.",
            "10-Q",
            70,
            "lower",
            -0.35,
            0.18,
            "contracting",
            True,
        ),
    ]

    for f_id, cik, tick, name, form, offset_days, rev_g, sent, fwd, marg, rest in sample_filings:
        f_date = base_date + timedelta(days=offset_days)
        conn.execute(
            """
            INSERT OR REPLACE INTO filings_metadata (
                filing_id, cik, ticker, company_name, form_type,
                filing_date, period_of_report, accession_number, primary_doc_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f_id,
                cik,
                tick,
                name,
                form,
                f_date,
                f_date,
                f_id.split("_")[1],
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{f_id}.htm",
            ],
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO extracted_signals (
                signal_id, filing_id, extraction_model, revenue_guidance,
                revenue_guidance_detail, eps_guidance, eps_guidance_detail,
                margin_outlook, capex_direction, management_sentiment,
                forward_language_ratio, risk_factor_count, guidance_confidence,
                restructuring_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"sig_{f_id}",
                f_id,
                "gemini-2.5-flash",
                rev_g,
                "Guidance revision details from management statement.",
                rev_g,
                "",
                marg,
                "stable",
                sent,
                fwd,
                3,
                0.92,
                rest,
            ],
        )

    logger.info("Demo dataset seeded successfully.")


def main() -> None:
    """CLI entrypoint to launch the web dashboard and REST server."""
    parser = argparse.ArgumentParser(
        description="Launch the Financial Earnings Intelligence web server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable code auto-reload.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Seed demo filings and prices into DuckDB if database is empty.",
    )

    args = parser.parse_args()

    settings = get_settings()
    with get_connection(settings) as conn:
        initialize_schema(conn)
        if args.demo:
            seed_demo_data_if_empty(conn)

    print("\n" + "=" * 70)
    print("      FINANCIAL EARNINGS INTELLIGENCE WEB SERVER LAUNCHED      ")
    print("=" * 70)
    print(f"  Interactive Dashboard UI: http://{args.host}:{args.port}/")
    print(f"  REST API Documentation:   http://{args.host}:{args.port}/docs")
    print(f"  Alternative OpenAPI:      http://{args.host}:{args.port}/redoc")
    print(f"  Health Check Endpoint:    http://{args.host}:{args.port}/api/health")
    print("=" * 70 + "\n")

    uvicorn.run(
        "earnings_intel.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
