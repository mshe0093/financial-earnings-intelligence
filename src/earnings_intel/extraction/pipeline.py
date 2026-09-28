"""End-to-end extraction pipeline: MD&A text/chunks -> Gemini -> Merged signals -> DuckDB."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import duckdb

from earnings_intel.edgar.parser import chunk_text
from earnings_intel.extraction.gemini_client import (
    DEFAULT_EXTRACTION_MODEL,
    GeminiExtractionClient,
)
from earnings_intel.extraction.schemas import (
    CapexDirection,
    EarningsSignals,
    ExtractedSignalRecord,
    GuidanceDirection,
    MarginOutlook,
)

logger = logging.getLogger(__name__)


def merge_chunk_signals(
    chunk_results: list[tuple[EarningsSignals, int]],
) -> EarningsSignals:
    """Merge signal extractions across multiple MD&A text chunks into a unified representation.

    Args:
        chunk_results: List of (EarningsSignals, chunk_character_length) tuples.

    Returns:
        Consolidated `EarningsSignals` object.
    """
    if not chunk_results:
        raise ValueError("Cannot merge empty chunk results list.")

    if len(chunk_results) == 1:
        return chunk_results[0][0]

    signals_list = [res[0] for res in chunk_results]
    weights = [res[1] for res in chunk_results]
    total_weight = sum(weights) or 1

    # 1. Weighted continuous metrics
    weighted_sentiment = (
        sum(s.management_sentiment * w for s, w in zip(signals_list, weights, strict=False))
        / total_weight
    )
    weighted_forward_ratio = (
        sum(s.forward_language_ratio * w for s, w in zip(signals_list, weights, strict=False))
        / total_weight
    )
    weighted_confidence = (
        sum(s.guidance_confidence * w for s, w in zip(signals_list, weights, strict=False))
        / total_weight
    )

    # 2. Boolean restructuring flag: true if any chunk flags restructuring
    restructuring = any(s.restructuring_signals for s in signals_list)

    # 3. Deduplicated risk topics and max risk factor count
    seen_topics: set[str] = set()
    merged_topics: list[str] = []
    for s in signals_list:
        for topic in s.key_risk_topics:
            norm_topic = topic.strip().lower()
            if norm_topic and norm_topic not in seen_topics:
                seen_topics.add(norm_topic)
                merged_topics.append(topic.strip())

    risk_count = max(s.risk_factor_count for s in signals_list)

    # 4. Discrete enum resolution strategy
    def resolve_guidance(attr: str) -> GuidanceDirection:
        values = [getattr(s, attr) for s in signals_list]
        # Prioritize explicit guidance over maintain/none
        if GuidanceDirection.LOWER in values and weighted_sentiment < 0:
            return GuidanceDirection.LOWER
        if GuidanceDirection.RAISE in values:
            return GuidanceDirection.RAISE
        if GuidanceDirection.LOWER in values:
            return GuidanceDirection.LOWER
        if GuidanceDirection.MAINTAIN in values:
            return GuidanceDirection.MAINTAIN
        return GuidanceDirection.NONE

    def resolve_margin() -> MarginOutlook:
        values = [s.margin_outlook for s in signals_list]
        if MarginOutlook.CONTRACTING in values and weighted_sentiment < 0:
            return MarginOutlook.CONTRACTING
        if MarginOutlook.EXPANDING in values:
            return MarginOutlook.EXPANDING
        if MarginOutlook.CONTRACTING in values:
            return MarginOutlook.CONTRACTING
        if MarginOutlook.STABLE in values:
            return MarginOutlook.STABLE
        return MarginOutlook.NONE

    def resolve_capex() -> CapexDirection:
        values = [s.capex_direction for s in signals_list]
        for pref in (CapexDirection.INCREASING, CapexDirection.DECREASING, CapexDirection.STABLE):
            if pref in values:
                return pref
        return CapexDirection.NONE

    # Concatenate unique non-empty guidance details
    unique_rev = [
        s.revenue_guidance_detail.strip()
        for s in signals_list
        if s.revenue_guidance_detail.strip()
    ]
    rev_details = "; ".join(dict.fromkeys(unique_rev))

    unique_eps = [
        s.eps_guidance_detail.strip() for s in signals_list if s.eps_guidance_detail.strip()
    ]
    eps_details = "; ".join(dict.fromkeys(unique_eps))

    return EarningsSignals(
        revenue_guidance=resolve_guidance("revenue_guidance"),
        revenue_guidance_detail=rev_details,
        eps_guidance=resolve_guidance("eps_guidance"),
        eps_guidance_detail=eps_details,
        margin_outlook=resolve_margin(),
        capex_direction=resolve_capex(),
        management_sentiment=round(weighted_sentiment, 4),
        forward_language_ratio=round(weighted_forward_ratio, 4),
        risk_factor_count=risk_count,
        key_risk_topics=merged_topics[:10],
        guidance_confidence=round(weighted_confidence, 4),
        restructuring_signals=restructuring,
    )


def extract_filing_signals(
    filing_id: str,
    mda_text: str | list[str],
    client: GeminiExtractionClient,
    conn: duckdb.DuckDBPyConnection | None = None,
    model: str | None = None,
    version: int = 1,
) -> ExtractedSignalRecord:
    """Execute structured signal extraction for an SEC filing's MD&A section.

    Handles single-text or multi-chunk inputs, aggregates token usage, merges chunk signals,
    and optionally stores the result in DuckDB.

    Args:
        filing_id: Unique filing identifier, e.g. '{cik}_{accession_number}'.
        mda_text: Plain text or pre-split chunks of the MD&A section.
        client: Configured `GeminiExtractionClient` instance.
        conn: Optional DuckDB connection for immediate persistence.
        model: Optional model override.
        version: Signal extraction pipeline version number (default 1).

    Returns:
        `ExtractedSignalRecord` containing all 12 metrics and provenance metadata.
    """
    if isinstance(mda_text, str):
        chunks = chunk_text(mda_text, chunk_size=80_000, overlap=2_000)
    else:
        chunks = mda_text

    if not chunks:
        raise ValueError(f"No MD&A content provided for filing {filing_id}")

    logger.info("Extracting signals for filing %s across %d chunk(s)", filing_id, len(chunks))

    chunk_results: list[tuple[EarningsSignals, int]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, chunk in enumerate(chunks, start=1):
        logger.debug("Processing chunk %d/%d (%d chars)", i, len(chunks), len(chunk))
        signals, usage = client.extract_signals(chunk, model=model)
        chunk_results.append((signals, len(chunk)))

        if usage.get("prompt_tokens") is not None:
            total_prompt_tokens += usage["prompt_tokens"] or 0
        if usage.get("completion_tokens") is not None:
            total_completion_tokens += usage["completion_tokens"] or 0

    # Merge signals across chunks
    merged_signals = merge_chunk_signals(chunk_results)

    total_chars = sum(len(c) for c in chunks)
    signal_id = f"{filing_id}_v{version}"
    model_name = model or client._default_model or DEFAULT_EXTRACTION_MODEL

    record = ExtractedSignalRecord(
        signal_id=signal_id,
        filing_id=filing_id,
        extraction_model=model_name,
        extraction_timestamp=datetime.utcnow(),
        revenue_guidance=merged_signals.revenue_guidance,
        revenue_guidance_detail=merged_signals.revenue_guidance_detail,
        eps_guidance=merged_signals.eps_guidance,
        eps_guidance_detail=merged_signals.eps_guidance_detail,
        margin_outlook=merged_signals.margin_outlook,
        capex_direction=merged_signals.capex_direction,
        management_sentiment=merged_signals.management_sentiment,
        forward_language_ratio=merged_signals.forward_language_ratio,
        risk_factor_count=merged_signals.risk_factor_count,
        key_risk_topics=merged_signals.key_risk_topics,
        guidance_confidence=merged_signals.guidance_confidence,
        restructuring_signals=merged_signals.restructuring_signals,
        mda_char_count=total_chars,
        prompt_tokens_used=total_prompt_tokens or None,
        completion_tokens_used=total_completion_tokens or None,
    )

    if conn is not None:
        store_signal_record(conn, record)

    return record


def store_signal_record(
    conn: duckdb.DuckDBPyConnection,
    record: ExtractedSignalRecord,
) -> None:
    """Insert or replace an extracted signal record in the DuckDB `extracted_signals` table."""
    conn.execute(
        """
        INSERT OR REPLACE INTO extracted_signals (
            signal_id, filing_id, extraction_model, extraction_timestamp,
            revenue_guidance, revenue_guidance_detail,
            eps_guidance, eps_guidance_detail,
            margin_outlook, capex_direction,
            management_sentiment, forward_language_ratio,
            risk_factor_count, key_risk_topics,
            guidance_confidence, restructuring_signals,
            mda_char_count, prompt_tokens_used, completion_tokens_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record.signal_id,
            record.filing_id,
            record.extraction_model,
            record.extraction_timestamp,
            record.revenue_guidance.value,
            record.revenue_guidance_detail,
            record.eps_guidance.value,
            record.eps_guidance_detail,
            record.margin_outlook.value,
            record.capex_direction.value,
            record.management_sentiment,
            record.forward_language_ratio,
            record.risk_factor_count,
            record.key_risk_topics,
            record.guidance_confidence,
            record.restructuring_signals,
            record.mda_char_count,
            record.prompt_tokens_used,
            record.completion_tokens_used,
        ],
    )
    logger.info("Stored signal record %s in DuckDB", record.signal_id)


def get_filing_signals(
    conn: duckdb.DuckDBPyConnection,
    filing_id: str,
) -> ExtractedSignalRecord | None:
    """Retrieve the most recent extracted signal record for a given filing_id.

    Returns:
        `ExtractedSignalRecord` if found, else `None`.
    """
    row = conn.execute(
        """
        SELECT
            signal_id, filing_id, extraction_model, extraction_timestamp,
            revenue_guidance, revenue_guidance_detail,
            eps_guidance, eps_guidance_detail,
            margin_outlook, capex_direction,
            management_sentiment, forward_language_ratio,
            risk_factor_count, key_risk_topics,
            guidance_confidence, restructuring_signals,
            mda_char_count, prompt_tokens_used, completion_tokens_used
        FROM extracted_signals
        WHERE filing_id = ?
        ORDER BY extraction_timestamp DESC
        LIMIT 1
        """,
        [filing_id],
    ).fetchone()

    if row is None:
        return None

    return _row_to_record(row)


def _row_to_record(row: tuple[Any, ...]) -> ExtractedSignalRecord:
    """Map a DuckDB query row tuple to an ExtractedSignalRecord."""
    return ExtractedSignalRecord(
        signal_id=row[0],
        filing_id=row[1],
        extraction_model=row[2],
        extraction_timestamp=row[3],
        revenue_guidance=GuidanceDirection(row[4]),
        revenue_guidance_detail=row[5] or "",
        eps_guidance=GuidanceDirection(row[6]),
        eps_guidance_detail=row[7] or "",
        margin_outlook=MarginOutlook(row[8]),
        capex_direction=CapexDirection(row[9]),
        management_sentiment=float(row[10]),
        forward_language_ratio=float(row[11]),
        risk_factor_count=int(row[12]),
        key_risk_topics=list(row[13]) if row[13] else [],
        guidance_confidence=float(row[14]),
        restructuring_signals=bool(row[15]),
        mda_char_count=int(row[16] or 0),
        prompt_tokens_used=row[17],
        completion_tokens_used=row[18],
    )
