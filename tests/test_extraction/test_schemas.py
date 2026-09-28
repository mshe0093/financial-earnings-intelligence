"""Unit tests for Gemini structured extraction Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from earnings_intel.extraction.schemas import (
    CapexDirection,
    EarningsSignals,
    ExtractedSignalRecord,
    GuidanceDirection,
    MarginOutlook,
)


@pytest.fixture()
def valid_signals_payload() -> dict:
    """Fixture providing a dictionary with valid values for all 12 metrics."""
    return {
        "revenue_guidance": "raise",
        "revenue_guidance_detail": "Management expects full-year revenue to grow 10-12%.",
        "eps_guidance": "raise",
        "eps_guidance_detail": "Targeting diluted EPS between $6.50 and $6.80.",
        "margin_outlook": "expanding",
        "capex_direction": "increasing",
        "management_sentiment": 0.75,
        "forward_language_ratio": 0.42,
        "risk_factor_count": 5,
        "key_risk_topics": ["FX volatility", "supply chain constraints", "component costs"],
        "guidance_confidence": 0.90,
        "restructuring_signals": False,
    }


class TestEarningsSignalsSchema:
    """Test validation and constraints on the 12-metric EarningsSignals model."""

    def test_valid_instantiation(self, valid_signals_payload: dict) -> None:
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.revenue_guidance == GuidanceDirection.RAISE
        assert signals.eps_guidance == GuidanceDirection.RAISE
        assert signals.margin_outlook == MarginOutlook.EXPANDING
        assert signals.capex_direction == CapexDirection.INCREASING
        assert signals.management_sentiment == 0.75
        assert signals.forward_language_ratio == 0.42
        assert signals.risk_factor_count == 5
        assert len(signals.key_risk_topics) == 3
        assert signals.guidance_confidence == 0.90
        assert signals.restructuring_signals is False

    def test_json_serialization_roundtrip(self, valid_signals_payload: dict) -> None:
        original = EarningsSignals(**valid_signals_payload)
        json_str = original.model_dump_json()
        restored = EarningsSignals.model_validate_json(json_str)
        assert original == restored

    @pytest.mark.parametrize(
        "guidance",
        ["raise", "maintain", "lower", "none"],
    )
    def test_valid_guidance_enums(self, valid_signals_payload: dict, guidance: str) -> None:
        valid_signals_payload["revenue_guidance"] = guidance
        valid_signals_payload["eps_guidance"] = guidance
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.revenue_guidance.value == guidance
        assert signals.eps_guidance.value == guidance

    def test_invalid_guidance_enum_raises(self, valid_signals_payload: dict) -> None:
        valid_signals_payload["revenue_guidance"] = "skyrocketing"
        with pytest.raises(ValidationError):
            EarningsSignals(**valid_signals_payload)

    @pytest.mark.parametrize(
        "margin",
        ["expanding", "stable", "contracting", "none"],
    )
    def test_valid_margin_enums(self, valid_signals_payload: dict, margin: str) -> None:
        valid_signals_payload["margin_outlook"] = margin
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.margin_outlook.value == margin

    @pytest.mark.parametrize(
        "capex",
        ["increasing", "stable", "decreasing", "none"],
    )
    def test_valid_capex_enums(self, valid_signals_payload: dict, capex: str) -> None:
        valid_signals_payload["capex_direction"] = capex
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.capex_direction.value == capex

    @pytest.mark.parametrize("sentiment", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_valid_sentiment_range(self, valid_signals_payload: dict, sentiment: float) -> None:
        valid_signals_payload["management_sentiment"] = sentiment
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.management_sentiment == sentiment

    @pytest.mark.parametrize("invalid_sentiment", [-1.01, 1.01, -5.0, 10.0])
    def test_invalid_sentiment_raises(
        self, valid_signals_payload: dict, invalid_sentiment: float
    ) -> None:
        valid_signals_payload["management_sentiment"] = invalid_sentiment
        with pytest.raises(ValidationError):
            EarningsSignals(**valid_signals_payload)

    @pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
    def test_valid_forward_language_ratio_range(
        self, valid_signals_payload: dict, ratio: float
    ) -> None:
        valid_signals_payload["forward_language_ratio"] = ratio
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.forward_language_ratio == ratio

    @pytest.mark.parametrize("invalid_ratio", [-0.01, 1.01, 2.0])
    def test_invalid_forward_language_ratio_raises(
        self, valid_signals_payload: dict, invalid_ratio: float
    ) -> None:
        valid_signals_payload["forward_language_ratio"] = invalid_ratio
        with pytest.raises(ValidationError):
            EarningsSignals(**valid_signals_payload)

    def test_negative_risk_factor_count_raises(self, valid_signals_payload: dict) -> None:
        valid_signals_payload["risk_factor_count"] = -1
        with pytest.raises(ValidationError):
            EarningsSignals(**valid_signals_payload)

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_confidence_range(self, valid_signals_payload: dict, confidence: float) -> None:
        valid_signals_payload["guidance_confidence"] = confidence
        signals = EarningsSignals(**valid_signals_payload)
        assert signals.guidance_confidence == confidence

    @pytest.mark.parametrize("invalid_confidence", [-0.1, 1.1])
    def test_invalid_confidence_raises(
        self, valid_signals_payload: dict, invalid_confidence: float
    ) -> None:
        valid_signals_payload["guidance_confidence"] = invalid_confidence
        with pytest.raises(ValidationError):
            EarningsSignals(**valid_signals_payload)


class TestExtractedSignalRecord:
    """Test validation and construction of full database signal records."""

    def test_valid_record_construction(self, valid_signals_payload: dict) -> None:
        record = ExtractedSignalRecord(
            signal_id="0000320193_0000320193-23-000106_v1",
            filing_id="0000320193_0000320193-23-000106",
            extraction_model="gemini-2.5-flash",
            mda_char_count=45000,
            prompt_tokens_used=12000,
            completion_tokens_used=450,
            **valid_signals_payload,
        )
        assert record.signal_id == "0000320193_0000320193-23-000106_v1"
        assert record.mda_char_count == 45000
        assert record.prompt_tokens_used == 12000
        assert record.completion_tokens_used == 450
