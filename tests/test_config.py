"""Tests for earnings_intel.config — Pydantic BaseSettings validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from earnings_intel.config import Settings


class TestSettingsLoading:
    """Verify that Settings loads and validates environment variables correctly."""

    def test_settings_loads_from_env(self, test_settings: Settings) -> None:
        """Settings should parse all expected fields from env vars."""
        assert test_settings.gemini_api_key.get_secret_value() == "test-gemini-key-not-real"
        assert test_settings.fred_api_key.get_secret_value() == "test-fred-key-not-real"
        assert test_settings.sec_user_agent == "TestApp test@example.com"
        assert test_settings.log_level == "DEBUG"
        assert test_settings.edgar_rate_limit_rps == 8

    def test_secret_str_redaction(self, test_settings: Settings) -> None:
        """SecretStr fields must NOT leak values in repr or str."""
        repr_str = repr(test_settings)
        assert "test-gemini-key-not-real" not in repr_str
        assert "test-fred-key-not-real" not in repr_str
        # The redacted placeholder should appear instead.
        assert "**********" in repr_str

    def test_defaults_applied(self, mock_env: dict[str, str]) -> None:
        """Default values should be used when optional env vars are absent."""
        # Remove optional overrides from env to test defaults.
        env_minimal = {
            "GEMINI_API_KEY": "key",
            "FRED_API_KEY": "key",
            "SEC_USER_AGENT": "App user@test.com",
        }
        with patch.dict("os.environ", env_minimal, clear=True):
            s = Settings()  # type: ignore[call-arg]
            assert s.duckdb_path == "data/earnings.duckdb"
            assert s.log_level == "INFO"
            assert s.edgar_rate_limit_rps == 8


class TestSecUserAgentValidation:
    """SEC EDGAR requires 'AppName contact@email.com' format."""

    @pytest.mark.parametrize(
        "valid_agent",
        [
            "EarningsApp user@domain.com",
            "MyFinBot researcher@university.edu",
            "SEC-Scraper ops@company.co.uk",
        ],
    )
    def test_valid_user_agents_accepted(self, mock_env: dict[str, str], valid_agent: str) -> None:
        env = {**mock_env, "SEC_USER_AGENT": valid_agent}
        with patch.dict("os.environ", env, clear=True):
            s = Settings()  # type: ignore[call-arg]
            assert s.sec_user_agent == valid_agent

    @pytest.mark.parametrize(
        "invalid_agent",
        [
            "",  # Empty
            "NoEmailHere",  # Missing email
            "user@domain.com",  # Missing app name prefix
            "App user@",  # Incomplete email
            "App user@domain",  # Missing TLD
        ],
    )
    def test_invalid_user_agents_rejected(
        self, mock_env: dict[str, str], invalid_agent: str
    ) -> None:
        env = {**mock_env, "SEC_USER_AGENT": invalid_agent}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(Exception):  # noqa: B017
                Settings()  # type: ignore[call-arg]


class TestLogLevelValidation:
    """log_level must be a valid Python logging level name."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_levels_accepted(self, mock_env: dict[str, str], level: str) -> None:
        env = {**mock_env, "LOG_LEVEL": level}
        with patch.dict("os.environ", env, clear=True):
            s = Settings()  # type: ignore[call-arg]
            assert s.log_level == level

    def test_case_insensitive(self, mock_env: dict[str, str]) -> None:
        env = {**mock_env, "LOG_LEVEL": "debug"}
        with patch.dict("os.environ", env, clear=True):
            s = Settings()  # type: ignore[call-arg]
            assert s.log_level == "DEBUG"

    def test_invalid_level_rejected(self, mock_env: dict[str, str]) -> None:
        env = {**mock_env, "LOG_LEVEL": "VERBOSE"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(Exception):  # noqa: B017
                Settings()  # type: ignore[call-arg]


class TestRateLimitValidation:
    """edgar_rate_limit_rps must be between 1 and 10."""

    def test_zero_rejected(self, mock_env: dict[str, str]) -> None:
        env = {**mock_env, "EDGAR_RATE_LIMIT_RPS": "0"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(Exception):  # noqa: B017
                Settings()  # type: ignore[call-arg]

    def test_eleven_rejected(self, mock_env: dict[str, str]) -> None:
        env = {**mock_env, "EDGAR_RATE_LIMIT_RPS": "11"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(Exception):  # noqa: B017
                Settings()  # type: ignore[call-arg]

    def test_boundary_values_accepted(self, mock_env: dict[str, str]) -> None:
        for val in ("1", "10"):
            env = {**mock_env, "EDGAR_RATE_LIMIT_RPS": val}
            with patch.dict("os.environ", env, clear=True):
                s = Settings()  # type: ignore[call-arg]
                assert s.edgar_rate_limit_rps == int(val)
