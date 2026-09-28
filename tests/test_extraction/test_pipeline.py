"""Unit and integration tests for the structured extraction pipeline and DuckDB persistence."""

from __future__ import annotations

from unittest.mock import MagicMock

import duckdb
import pytest

from earnings_intel.config import Settings
from earnings_intel.extraction.gemini_client import GeminiExtractionClient
from earnings_intel.extraction.pipeline import (
    extract_filing_signals,
    get_filing_signals,
    merge_chunk_signals,
)
from earnings_intel.extraction.schemas import (
    CapexDirection,
    EarningsSignals,
    GuidanceDirection,
    MarginOutlook,
)


@pytest.fixture()
def chunk_signals_sample() -> tuple[EarningsSignals, EarningsSignals]:
    """Two distinct EarningsSignals representing two chunks of an MD&A section."""
    chunk1 = EarningsSignals(
        revenue_guidance=GuidanceDirection.RAISE,
        revenue_guidance_detail="Cloud demand surged 20%.",
        eps_guidance=GuidanceDirection.MAINTAIN,
        eps_guidance_detail="Maintaining EPS expectations.",
        margin_outlook=MarginOutlook.EXPANDING,
        capex_direction=CapexDirection.INCREASING,
        management_sentiment=0.8,
        forward_language_ratio=0.5,
        risk_factor_count=3,
        key_risk_topics=["competition", "supply chain"],
        guidance_confidence=0.9,
        restructuring_signals=False,
    )
    chunk2 = EarningsSignals(
        revenue_guidance=GuidanceDirection.MAINTAIN,
        revenue_guidance_detail="",
        eps_guidance=GuidanceDirection.MAINTAIN,
        eps_guidance_detail="",
        margin_outlook=MarginOutlook.STABLE,
        capex_direction=CapexDirection.INCREASING,
        management_sentiment=0.4,
        forward_language_ratio=0.3,
        risk_factor_count=4,
        key_risk_topics=["supply chain", "currency risk", "regulatory"],
        guidance_confidence=0.7,
        restructuring_signals=True,
    )
    return chunk1, chunk2


class TestMergeChunkSignals:
    """Tests for multi-chunk aggregation logic."""

    def test_single_chunk_returns_directly(
        self,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, _ = chunk_signals_sample
        merged = merge_chunk_signals([(chunk1, 1000)])
        assert merged == chunk1

    def test_empty_list_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot merge empty"):
            merge_chunk_signals([])

    def test_weighted_merging_of_continuous_metrics(
        self,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, chunk2 = chunk_signals_sample
        # Chunk1 has weight 1000, Chunk2 has weight 3000 (total: 4000)
        # Expected sentiment: (0.8*1000 + 0.4*3000) / 4000 = (800 + 1200)/4000 = 0.5
        merged = merge_chunk_signals([(chunk1, 1000), (chunk2, 3000)])
        assert merged.management_sentiment == pytest.approx(0.5)
        # Expected forward_language_ratio: (0.5*1000 + 0.3*3000)/4000 = 1400/4000 = 0.35
        assert merged.forward_language_ratio == pytest.approx(0.35)
        # Expected confidence: (0.9*1000 + 0.7*3000)/4000 = 3000/4000 = 0.75
        assert merged.guidance_confidence == pytest.approx(0.75)

    def test_restructuring_flag_merged(
        self,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, chunk2 = chunk_signals_sample
        merged = merge_chunk_signals([(chunk1, 1000), (chunk2, 1000)])
        assert merged.restructuring_signals is True

    def test_key_risk_topics_deduplicated(
        self,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, chunk2 = chunk_signals_sample
        merged = merge_chunk_signals([(chunk1, 1000), (chunk2, 1000)])
        # Chunk1: ["competition", "supply chain"]
        # Chunk2: ["supply chain", "currency risk", "regulatory"]
        # Union without duplicates: ["competition", "supply chain", "currency risk", "regulatory"]
        assert merged.key_risk_topics == [
            "competition",
            "supply chain",
            "currency risk",
            "regulatory",
        ]
        assert merged.risk_factor_count == 4  # max(3, 4)

    def test_guidance_resolution(
        self,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, chunk2 = chunk_signals_sample
        merged = merge_chunk_signals([(chunk1, 1000), (chunk2, 1000)])
        # chunk1 had RAISE, chunk2 had MAINTAIN -> merged should be RAISE
        assert merged.revenue_guidance == GuidanceDirection.RAISE
        assert "Cloud demand surged 20%." in merged.revenue_guidance_detail


class TestExtractionPipeline:
    """Integration tests for extract_filing_signals with DuckDB."""

    def test_pipeline_with_mocked_client_and_db(
        self,
        memory_db: duckdb.DuckDBPyConnection,
        test_settings: Settings,
        chunk_signals_sample: tuple[EarningsSignals, EarningsSignals],
    ) -> None:
        chunk1, chunk2 = chunk_signals_sample

        # First insert parent filing into filings_metadata
        memory_db.execute(
            """
            INSERT INTO filings_metadata (
                filing_id, cik, ticker, company_name, form_type,
                filing_date, period_of_report, accession_number, primary_doc_url
            ) VALUES (
                '0000320193_0000320193-23-000106', '0000320193', 'AAPL', 'Apple Inc', '10-K',
                '2023-11-03', '2023-09-30', '0000320193-23-000106', 'https://sec.gov/doc'
            )
            """
        )

        mock_client = MagicMock(spec=GeminiExtractionClient)
        mock_client._default_model = "gemini-2.5-flash"
        # Mock extract_signals return value for chunks
        mock_client.extract_signals.side_effect = [
            (chunk1, {"prompt_tokens": 1200, "completion_tokens": 180}),
            (chunk2, {"prompt_tokens": 1400, "completion_tokens": 200}),
        ]

        # Execute extraction pipeline with 2 chunks
        chunks = ["Excerpt from part 1 of filing.", "Excerpt from part 2 of filing."]
        record = extract_filing_signals(
            filing_id="0000320193_0000320193-23-000106",
            mda_text=chunks,
            client=mock_client,
            conn=memory_db,
            model="gemini-2.5-flash",
            version=1,
        )

        assert record.signal_id == "0000320193_0000320193-23-000106_v1"
        assert record.prompt_tokens_used == 2600  # 1200 + 1400
        assert record.completion_tokens_used == 380  # 180 + 200
        assert record.revenue_guidance == GuidanceDirection.RAISE
        assert record.restructuring_signals is True

        # Query back from DuckDB
        stored = get_filing_signals(memory_db, "0000320193_0000320193-23-000106")
        assert stored is not None
        assert stored.signal_id == record.signal_id
        assert stored.prompt_tokens_used == 2600
        assert stored.key_risk_topics == record.key_risk_topics
