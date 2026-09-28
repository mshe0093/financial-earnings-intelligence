"""Unit tests for GeminiExtractionClient — API mocking, structured output, and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import APIError

from earnings_intel.config import Settings
from earnings_intel.extraction.gemini_client import (
    GeminiExtractionClient,
    GeminiExtractionError,
    GeminiRateLimitError,
    GeminiSchemaValidationError,
)
from earnings_intel.extraction.schemas import EarningsSignals, GuidanceDirection


@pytest.fixture()
def mock_signals_json() -> str:
    """JSON string matching EarningsSignals schema."""
    return json.dumps(
        {
            "revenue_guidance": "raise",
            "revenue_guidance_detail": "Revenue anticipated to grow 15% year-over-year.",
            "eps_guidance": "maintain",
            "eps_guidance_detail": "Reaffirming previous diluted EPS guidance range.",
            "margin_outlook": "stable",
            "capex_direction": "increasing",
            "management_sentiment": 0.65,
            "forward_language_ratio": 0.35,
            "risk_factor_count": 4,
            "key_risk_topics": ["inflation", "supply chain", "interest rates"],
            "guidance_confidence": 0.85,
            "restructuring_signals": False,
        }
    )


class TestGeminiExtractionClient:
    """Test suite for GeminiExtractionClient wrapper."""

    def test_extract_signals_success(
        self,
        test_settings: Settings,
        mock_signals_json: str,
    ) -> None:
        client = GeminiExtractionClient(test_settings)

        # Mock generate_content response
        mock_response = MagicMock()
        mock_response.text = mock_signals_json
        mock_response.usage_metadata.prompt_token_count = 1500
        mock_response.usage_metadata.candidates_token_count = 210

        with patch.object(
            client._client.models, "generate_content", return_value=mock_response
        ) as mock_gen:
            signals, usage = client.extract_signals("Sample MD&A text", model="gemini-2.5-flash")

            assert isinstance(signals, EarningsSignals)
            assert signals.revenue_guidance == GuidanceDirection.RAISE
            assert signals.management_sentiment == 0.65
            assert usage["prompt_tokens"] == 1500
            assert usage["completion_tokens"] == 210

            # Verify parameters passed to google-genai
            mock_gen.assert_called_once()
            _, kwargs = mock_gen.call_args
            assert kwargs["model"] == "gemini-2.5-flash"
            assert kwargs["config"].response_mime_type == "application/json"
            assert kwargs["config"].response_schema == EarningsSignals

    def test_extract_signals_empty_response_raises(self, test_settings: Settings) -> None:
        client = GeminiExtractionClient(test_settings)
        mock_response = MagicMock()
        mock_response.text = ""

        with patch.object(client._client.models, "generate_content", return_value=mock_response):
            with pytest.raises(GeminiExtractionError, match="empty response"):
                client.extract_signals("Sample text")

    def test_extract_signals_schema_validation_failure_raises(
        self, test_settings: Settings
    ) -> None:
        client = GeminiExtractionClient(test_settings)
        mock_response = MagicMock()
        # Invalid payload missing required fields and having invalid values
        mock_response.text = json.dumps(
            {"revenue_guidance": "invalid_enum", "management_sentiment": 99.0}
        )

        with patch.object(client._client.models, "generate_content", return_value=mock_response):
            with pytest.raises(GeminiSchemaValidationError):
                client.extract_signals("Sample text")

    def test_extract_signals_rate_limit_maps_and_retries(self, test_settings: Settings) -> None:
        client = GeminiExtractionClient(test_settings)

        # Create APIError with 429
        mock_error = APIError(429, {"message": "Resource exhausted: 429 Rate limit exceeded"})

        with patch.object(client._client.models, "generate_content", side_effect=mock_error):
            with patch("time.sleep"):  # Avoid waiting for tenacity backoff
                with pytest.raises(GeminiRateLimitError):
                    client.extract_signals("Sample text")
