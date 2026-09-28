"""Gemini API client wrapper for structured extraction of financial signals.

Uses the official Google GenAI SDK (`google-genai`) with Pydantic v2 `response_schema`
to guarantee strict, typed output matching the `EarningsSignals` data model.
"""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from earnings_intel.config import Settings, get_settings
from earnings_intel.extraction.prompts import SYSTEM_INSTRUCTION, build_extraction_prompt
from earnings_intel.extraction.schemas import EarningsSignals

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTION_MODEL = "gemini-2.5-flash"


class GeminiExtractionError(Exception):
    """Base exception for Gemini extraction operations."""


class GeminiRateLimitError(GeminiExtractionError):
    """Raised when Gemini API quota or rate limit is exceeded (HTTP 429)."""


class GeminiSchemaValidationError(GeminiExtractionError):
    """Raised when model response cannot be validated against EarningsSignals schema."""


class GeminiExtractionClient:
    """Wrapper around Google GenAI client configured for structured financial signal extraction.

    Usage:
        client = GeminiExtractionClient()
        signals, usage = client.extract_signals(mda_text)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        default_model: str = DEFAULT_EXTRACTION_MODEL,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_key = self._settings.gemini_api_key.get_secret_value()
        self._default_model = default_model

        # Initialize official GenAI SDK client
        self._client = genai.Client(api_key=self._api_key)

    @property
    def client(self) -> genai.Client:
        """Access underlying google-genai Client instance."""
        return self._client

    @retry(
        retry=retry_if_exception_type((GeminiRateLimitError, APIError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_generate_content(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
    ) -> Any:
        """Call Gemini API with retry logic and rate limit mapping."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EarningsSignals,
            temperature=temperature,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        try:
            return self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except APIError as e:
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning("Gemini API rate limit reached: %s. Retrying...", e)
                raise GeminiRateLimitError(str(e)) from e
            raise

    def extract_signals(
        self,
        mda_text: str,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> tuple[EarningsSignals, dict[str, int | None]]:
        """Extract structured financial signals from an MD&A text excerpt.

        Args:
            mda_text: Cleaned text from the MD&A section.
            model: Optional Gemini model name override (defaults to gemini-2.5-flash).
            temperature: Generation temperature (low default 0.1 for factual consistency).

        Returns:
            Tuple of:
            - `EarningsSignals`: validated Pydantic model with all 12 metrics.
            - `dict`: token usage metadata, e.g. `{"prompt_tokens": 1200}`.

        Raises:
            GeminiRateLimitError: If API quota/rate limit is reached after retries.
            GeminiSchemaValidationError: If response JSON violates the schema.
            GeminiExtractionError: For other unexpected API failures.
        """
        model_name = model or self._default_model
        prompt = build_extraction_prompt(mda_text)

        logger.debug(
            "Requesting structured extraction from %s (input length: %d chars)",
            model_name,
            len(mda_text),
        )

        try:
            response = self._call_generate_content(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
            )
        except Exception as e:
            if not isinstance(e, GeminiExtractionError):
                logger.error("Failed to generate content from Gemini API: %s", e)
                raise GeminiExtractionError(f"Gemini API error: {e}") from e
            raise

        response_text = getattr(response, "text", "") or ""
        if not response_text:
            raise GeminiExtractionError("Gemini API returned an empty response.")

        # Parse and validate response text against EarningsSignals
        try:
            signals = EarningsSignals.model_validate_json(response_text)
        except ValidationError as e:
            logger.error("Schema validation failed on Gemini response: %s", e)
            raise GeminiSchemaValidationError(
                f"Failed to validate response against EarningsSignals: {e}"
            ) from e

        # Extract token usage metadata if present
        usage_info: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            usage_info["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
            usage_info["completion_tokens"] = getattr(usage, "candidates_token_count", None)

        logger.info(
            "Successfully extracted signals with %s (tokens: %s prompt, %s completion)",
            model_name,
            usage_info["prompt_tokens"],
            usage_info["completion_tokens"],
        )

        return signals, usage_info
