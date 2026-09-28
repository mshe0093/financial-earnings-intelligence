"""Gemini structured extraction pipeline for SEC filing MD&A text."""

from earnings_intel.extraction.gemini_client import (
    GeminiExtractionClient,
    GeminiExtractionError,
    GeminiRateLimitError,
    GeminiSchemaValidationError,
)
from earnings_intel.extraction.pipeline import (
    extract_filing_signals,
    get_filing_signals,
    merge_chunk_signals,
    store_signal_record,
)
from earnings_intel.extraction.schemas import (
    CapexDirection,
    EarningsSignals,
    ExtractedSignalRecord,
    GuidanceDirection,
    MarginOutlook,
)

__all__ = [
    "CapexDirection",
    "EarningsSignals",
    "ExtractedSignalRecord",
    "GeminiExtractionClient",
    "GeminiExtractionError",
    "GeminiRateLimitError",
    "GeminiSchemaValidationError",
    "GuidanceDirection",
    "MarginOutlook",
    "extract_filing_signals",
    "get_filing_signals",
    "merge_chunk_signals",
    "store_signal_record",
]
