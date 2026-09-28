"""CLI script to ingest SEC EDGAR 10-K and 10-Q filings into DuckDB.

Usage examples:
    python scripts/ingest.py --ticker AAPL --forms 10-K --years 5
    python scripts/ingest.py --ticker MSFT --forms 10-K 10-Q --years 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

from earnings_intel.config import get_settings
from earnings_intel.db.connection import get_connection
from earnings_intel.db.schema import initialize_schema
from earnings_intel.edgar.cik import (
    fetch_company_tickers,
    load_ticker_mapping_from_db,
    lookup_cik,
    normalize_cik,
    store_ticker_mapping,
)
from earnings_intel.edgar.client import EdgarClient, store_filing_metadata
from earnings_intel.edgar.parser import MDAExtractionError, extract_and_process

logger = logging.getLogger("earnings_intel.ingest")


async def run_ingestion(
    ticker: str,
    form_types: list[str],
    years: int = 5,
    max_filings: int | None = None,
    save_raw: bool = True,
    extract_mda: bool = True,
) -> int:
    """Run the ingestion pipeline for a given ticker.

    Args:
        ticker: Uppercase ticker symbol (e.g. 'AAPL').
        form_types: List of form types (e.g. ['10-K', '10-Q']).
        years: How many years of filings to ingest.
        max_filings: Optional limit on total filings.
        save_raw: Whether to save raw HTML to data/raw/.
        extract_mda: Whether to extract Item 7/Item 2 MD&A text.

    Returns:
        Number of filings successfully ingested.
    """
    settings = get_settings()
    ticker = ticker.strip().upper()
    cutoff_date = date.today() - timedelta(days=years * 365)

    with get_connection(settings) as conn:
        initialize_schema(conn)

        async with EdgarClient(settings) as client:
            # 1. Lookup or fetch CIK mapping
            mapping = load_ticker_mapping_from_db(conn)
            cik = lookup_cik(ticker, mapping)

            if not cik:
                logger.info("Ticker %s not found in local DB. Fetching SEC directory...", ticker)
                mapping = await fetch_company_tickers(client.client)
                store_ticker_mapping(conn, mapping)
                cik = lookup_cik(ticker, mapping)

            if not cik:
                logger.error("Could not find CIK for ticker %s in SEC database.", ticker)
                return 0

            cik = normalize_cik(cik)
            logger.info("Resolved %s -> CIK %s", ticker, cik)

            # 2. Query company filings list
            filings = await client.get_filings_list(
                cik,
                form_types=tuple(form_types),
                ticker=ticker,
                max_filings=max_filings or 100,
            )

            # Filter by cutoff date
            eligible_filings = [f for f in filings if f.filing_date >= cutoff_date]
            logger.info(
                "Found %d filings for %s filed on or after %s",
                len(eligible_filings),
                ticker,
                cutoff_date,
            )

            if not eligible_filings:
                return 0

            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)

            ingested_count = 0
            for filing in eligible_filings:
                logger.info(
                    "Processing %s %s (filed: %s, acc: %s)...",
                    filing.ticker,
                    filing.form_type,
                    filing.filing_date,
                    filing.accession_number,
                )

                try:
                    # Download raw document
                    html_bytes = await client.download_filing_by_url(filing.primary_doc_url)

                    # Save raw file if requested
                    if save_raw:
                        doc_name = filing.primary_doc_url.split("/")[-1]
                        raw_file = raw_dir / f"{filing.cik}_{filing.accession_number}_{doc_name}"
                        raw_file.write_bytes(html_bytes)

                    # Extract MD&A text
                    if extract_mda:
                        try:
                            mda_res = extract_and_process(html_bytes, form_type=filing.form_type)
                            filing.mda_text_hash = mda_res.text_hash
                            logger.info(
                                "Extracted MD&A: %d chars, hash: %s",
                                mda_res.char_count,
                                mda_res.text_hash[:8],
                            )
                        except MDAExtractionError as e:
                            logger.warning(
                                "Could not extract MD&A for %s: %s", filing.filing_id, e
                            )

                    # Persist metadata to DuckDB
                    store_filing_metadata(conn, filing)
                    ingested_count += 1

                except Exception as e:
                    logger.error("Failed processing filing %s: %s", filing.filing_id, e)

            logger.info(
                "Ingestion complete: %d of %d filings successfully stored for %s.",
                ingested_count,
                len(eligible_filings),
                ticker,
            )
            return ingested_count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR 10-K/10-Q filings into DuckDB.")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K"],
        help="Filing forms to ingest (default: 10-K). E.g. --forms 10-K 10-Q",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of historical years to ingest (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of filings to ingest",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving raw HTML files to data/raw/",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    count = asyncio.run(
        run_ingestion(
            ticker=args.ticker,
            form_types=args.forms,
            years=args.years,
            max_filings=args.limit,
            save_raw=not args.no_raw,
        )
    )
    print(f"Ingested {count} filings for {args.ticker.upper()}.")


if __name__ == "__main__":
    main()
