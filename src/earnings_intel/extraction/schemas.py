"""Pydantic v2 schemas for Gemini structured extraction — 12 key financial signals.

These schemas are passed directly to the Gemini API as `response_schema`
for guaranteed, structured JSON extraction from SEC MD&A text.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GuidanceDirection(StrEnum):
    """Direction of management guidance revision relative to prior period."""

    RAISE = "raise"
    MAINTAIN = "maintain"
    LOWER = "lower"
    NONE = "none"  # No guidance provided or not discussed


class MarginOutlook(StrEnum):
    """Management's outlook on operating or gross margins."""

    EXPANDING = "expanding"
    STABLE = "stable"
    CONTRACTING = "contracting"
    NONE = "none"


class CapexDirection(StrEnum):
    """Direction of capital expenditure and infrastructure spending plans."""

    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    NONE = "none"


class EarningsSignals(BaseModel):
    """12 structured metrics extracted from a filing's MD&A section.

    Conforms to Section 6.1 of PROJECT_PLAN.md and serves as the JSON schema
    passed to Gemini for structured generation.
    """

    # -- Metric 1-2: Revenue Guidance --
    revenue_guidance: GuidanceDirection = Field(
        description="Direction of revenue guidance revision relative to prior period."
    )
    revenue_guidance_detail: str = Field(
        default="",
        description="Verbatim or paraphrased quote supporting the revenue guidance direction.",
    )

    # -- Metric 3-4: EPS Guidance --
    eps_guidance: GuidanceDirection = Field(
        description="Direction of EPS guidance revision relative to prior period."
    )
    eps_guidance_detail: str = Field(
        default="",
        description="Verbatim or paraphrased quote supporting the EPS guidance direction.",
    )

    # -- Metric 5: Margin Outlook --
    margin_outlook: MarginOutlook = Field(
        description="Management's outlook on operating/gross margins."
    )

    # -- Metric 6: CapEx Direction --
    capex_direction: CapexDirection = Field(description="Direction of capital expenditure plans.")

    # -- Metric 7: Management Sentiment --
    management_sentiment: float = Field(
        ge=-1.0,
        le=1.0,
        description="Composite sentiment score from -1.0 (very negative) to 1.0 (very positive).",
    )

    # -- Metric 8: Forward-Looking Language Ratio --
    forward_language_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of forward-looking statements to total statements (0.0 to 1.0).",
    )

    # -- Metric 9: Risk Factor Count --
    risk_factor_count: int = Field(
        ge=0,
        description="Number of distinct risk factors or headwinds mentioned in MD&A.",
    )

    # -- Metric 10: Key Risk Topics --
    key_risk_topics: list[str] = Field(
        default_factory=list,
        description="Top risk themes (e.g., 'supply chain', 'regulatory', 'FX exposure').",
    )

    # -- Metric 11: Guidance Confidence --
    guidance_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in accuracy of extracted guidance (0.0 to 1.0).",
    )

    # -- Metric 12: Restructuring Signals --
    restructuring_signals: bool = Field(
        description="True if mentioning layoffs, restructuring, impairment, or write-downs."
    )


class ExtractedSignalRecord(BaseModel):
    """Full database record corresponding to the `extracted_signals` DuckDB table.

    Combines the 12 extracted metrics with metadata and token provenance.
    """

    signal_id: str = Field(description="Unique ID: '{filing_id}_v{version}'")
    filing_id: str = Field(description="References filings_metadata(filing_id)")
    extraction_model: str = Field(description="Model used, e.g. 'gemini-2.5-flash'")
    extraction_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of extraction",
    )

    # 12 Extracted Metrics
    revenue_guidance: GuidanceDirection
    revenue_guidance_detail: str = ""
    eps_guidance: GuidanceDirection
    eps_guidance_detail: str = ""
    margin_outlook: MarginOutlook
    capex_direction: CapexDirection
    management_sentiment: float
    forward_language_ratio: float
    risk_factor_count: int
    key_risk_topics: list[str] = Field(default_factory=list)
    guidance_confidence: float
    restructuring_signals: bool

    # Provenance
    mda_char_count: int = 0
    prompt_tokens_used: int | None = None
    completion_tokens_used: int | None = None
