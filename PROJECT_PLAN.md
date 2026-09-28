# PROJECT PLAN — LLM-Powered Financial Earnings Intelligence System

> **Version:** 1.0.0  
> **Status:** Draft  
> **Last Updated:** 2026-09-28  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Directory Layout](#2-directory-layout)
3. [Security & Privacy Architecture](#3-security--privacy-architecture)
4. [Data Contracts (DuckDB Schema)](#4-data-contracts-duckdb-schema)
5. [SEC EDGAR Ingestion Pipeline](#5-sec-edgar-ingestion-pipeline)
6. [Gemini Structured Extraction Schema](#6-gemini-structured-extraction-schema)
7. [Quantitative Backtesting Logic](#7-quantitative-backtesting-logic)
8. [Interactive Demo Architecture](#8-interactive-demo-architecture)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Executive Summary

This system ingests SEC EDGAR 10-K and 10-Q filings, extracts forward-looking guidance and risk signals using the Gemini API with structured output, stores all data in a local DuckDB analytical database, computes post-earnings announcement drift (PEAD) metrics, and presents results through an interactive web dashboard. The architecture enforces zero-leak credential management, SEC Fair Access compliance, and fully reproducible backtesting.

---

## 2. Directory Layout

```
financial-earnings-intelligence/
├── .env.example                   # Sanitized placeholder secrets
├── .gitignore                     # Strict ignore rules (security-first)
├── .pre-commit-config.yaml        # detect-secrets + pre-commit hooks
├── pyproject.toml                 # uv-managed project metadata & deps
├── uv.lock                        # Deterministic lockfile (committed)
├── README.md                      # Quick-start, architecture overview
├── PROJECT_PLAN.md                # This document
│
├── src/
│   └── earnings_intel/            # Main Python package
│       ├── __init__.py
│       ├── config.py              # Pydantic v2 BaseSettings (SecretStr)
│       ├── db/
│       │   ├── __init__.py
│       │   ├── connection.py      # DuckDB connection factory
│       │   └── schema.py          # DDL migrations & table definitions
│       ├── edgar/
│       │   ├── __init__.py
│       │   ├── client.py          # SEC EDGAR HTTP client (rate-limited)
│       │   ├── cik.py             # CIK normalization & lookup
│       │   ├── parser.py          # 10-K/10-Q HTML → Item 7 MD&A text
│       │   └── models.py          # Filing metadata Pydantic models
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── gemini_client.py   # Gemini API wrapper (structured output)
│       │   ├── schemas.py         # 12-metric structured extraction schema
│       │   └── pipeline.py        # Filing → structured signals pipeline
│       ├── market_data/
│       │   ├── __init__.py
│       │   ├── fred_client.py     # FRED API client for macro indicators
│       │   └── price_fetcher.py   # Historical price series (yfinance)
│       ├── backtesting/
│       │   ├── __init__.py
│       │   ├── pead.py            # Post-earnings drift computation
│       │   ├── metrics.py         # Sharpe ratio, returns analytics
│       │   └── runner.py          # Backtest orchestrator
│       └── api/
│           ├── __init__.py
│           ├── main.py            # FastAPI application entry point
│           ├── routes.py          # REST endpoints
│           └── ui.py              # Gradio / Streamlit UI integration
│
├── tests/
│   ├── conftest.py                # Shared fixtures (mock configs, temp DB)
│   ├── test_config.py
│   ├── test_edgar/
│   │   ├── test_client.py
│   │   ├── test_cik.py
│   │   └── test_parser.py
│   ├── test_extraction/
│   │   ├── test_schemas.py
│   │   └── test_pipeline.py
│   ├── test_backtesting/
│   │   ├── test_pead.py
│   │   └── test_metrics.py
│   └── test_api/
│       └── test_routes.py
│
├── notebooks/
│   └── exploration.ipynb          # Ad-hoc analysis (git-ignored outputs)
│
├── scripts/
│   ├── bootstrap.sh               # One-shot dev environment setup
│   ├── ingest.py                  # CLI: run EDGAR ingestion pipeline
│   └── backtest.py                # CLI: run backtesting suite
│
├── data/                          # ALL subdirs git-ignored
│   ├── raw/                       # Raw EDGAR HTML filings
│   ├── cache/                     # HTTP response cache
│   └── processed/                 # Parquet/CSV exports
│
└── logs/                          # Application logs (git-ignored)
```

### Dependency Management

This project uses **[uv](https://docs.astral.sh/uv/)** as the sole Python dependency manager.

```toml
# pyproject.toml (key sections)
[project]
name = "earnings-intel"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.9,<3",
    "pydantic-settings>=2.5,<3",
    "duckdb>=1.1,<2",
    "httpx>=0.27,<1",
    "beautifulsoup4>=4.12,<5",
    "lxml>=5.3,<6",
    "google-genai>=1.0,<2",
    "yfinance>=0.2,<1",
    "fredapi>=0.5,<1",
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
    "gradio>=5.0,<6",
    "pandas>=2.2,<3",
    "plotly>=5.24,<6",
    "tenacity>=9.0,<10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<9",
    "pytest-asyncio>=0.24,<1",
    "pytest-cov>=5.0,<6",
    "httpx[http2]",
    "pre-commit>=3.8,<4",
    "detect-secrets>=1.5,<2",
    "ruff>=0.6,<1",
]

[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "S", "B", "A", "RUF"]
# S = bandit security checks

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

---

## 3. Security & Privacy Architecture

### 3.1 Zero-Leak Policy

> [!CAUTION]
> Under **NO circumstances** may API keys, email addresses, full names, or any credential appear in source code, tests, docstrings, notebook outputs, or committed artifacts.

| Layer | Control | Implementation |
|-------|---------|----------------|
| **Git Ignore** | `.gitignore` blocks secrets at the filesystem level | `.env`, `.env.*`, `*.env`, `data/`, `logs/`, `*.duckdb*` |
| **Pre-commit** | `detect-secrets` scans staged diffs | `.pre-commit-config.yaml` with `detect-secrets` hook |
| **Runtime** | Pydantic `SecretStr` redacts on repr/log | `config.py` uses `BaseSettings` + `SecretStr` |
| **Validation** | SEC User-Agent format enforced via regex | Pydantic `field_validator` on `sec_user_agent` |
| **CLI Verify** | Developer runs checks before first commit | `git check-ignore -v .env` and `git status --ignored` |

### 3.2 `.env.example` (Committed — Sanitized Placeholders Only)

```env
# === Financial Earnings Intelligence — Environment Configuration ===
# Copy this file to .env and replace placeholder values with real credentials.
# NEVER commit the .env file.

SEC_USER_AGENT="EarningsApp user@domain.com"
GEMINI_API_KEY="your_api_key_here"
FRED_API_KEY="your_api_key_here"

# Optional overrides
DUCKDB_PATH="data/earnings.duckdb"
LOG_LEVEL="INFO"
EDGAR_RATE_LIMIT_RPS="8"
```

### 3.3 Centralized Config (`src/earnings_intel/config.py`)

```python
"""Centralized configuration — all secrets parsed via Pydantic v2 BaseSettings."""

import re
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded exclusively from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Secrets (SecretStr — redacted in repr, logs, tracebacks) ──
    gemini_api_key: SecretStr
    fred_api_key: SecretStr

    # ── SEC Fair Access ──
    sec_user_agent: str

    # ── Database ──
    duckdb_path: str = "data/earnings.duckdb"

    # ── Operational ──
    log_level: str = "INFO"
    edgar_rate_limit_rps: int = 8

    @field_validator("sec_user_agent")
    @classmethod
    def validate_sec_user_agent(cls, v: str) -> str:
        """SEC requires 'Company/App Contact@Email' format."""
        pattern = r"^.+\s+\S+@\S+\.\S+$"
        if not re.match(pattern, v):
            raise ValueError(
                "sec_user_agent must match 'AppName user@domain.com' format "
                "per SEC EDGAR Fair Access policy."
            )
        return v


def get_settings() -> Settings:
    """Factory — returns a cached singleton in production."""
    return Settings()  # type: ignore[call-arg]
```

### 3.4 Pre-commit Hook Setup

The `.pre-commit-config.yaml` configures:

1. **`detect-secrets`** — scans staged diffs for high-entropy strings, AWS keys, API tokens, and generic credential patterns. Aborts the commit if any potential secret is detected.
2. **Trailing whitespace / EOF fixers** — standard hygiene hooks.

Developer bootstrap:

```bash
uv sync --group dev
uv run pre-commit install          # Install git hooks
uv run pre-commit run --all-files  # Verify clean baseline
```

### 3.5 Manual Verification Checklist

Before the first meaningful commit, the developer **must** run:

```bash
# Verify .env is git-ignored
git check-ignore -v .env
# Expected output: .gitignore:1:.env    .env

# Verify all ignored paths
git status --ignored

# Run detect-secrets baseline
uv run detect-secrets scan > .secrets.baseline
```

---

## 4. Data Contracts (DuckDB Schema)

All analytical data resides in a single DuckDB database file (`data/earnings.duckdb`).

### 4.1 Table: `filings_metadata`

Stores raw SEC EDGAR filing metadata.

```sql
CREATE TABLE IF NOT EXISTS filings_metadata (
    filing_id        VARCHAR PRIMARY KEY,  -- '{cik}_{accession_number}'
    cik              VARCHAR(10) NOT NULL, -- Zero-padded 10-digit CIK
    ticker           VARCHAR(10),
    company_name     VARCHAR NOT NULL,
    form_type        VARCHAR(10) NOT NULL, -- '10-K' or '10-Q'
    filing_date      DATE NOT NULL,
    period_of_report DATE NOT NULL,
    accession_number VARCHAR NOT NULL,
    primary_doc_url  VARCHAR NOT NULL,
    mda_text_hash    VARCHAR,             -- SHA-256 of extracted MD&A text
    ingested_at      TIMESTAMP DEFAULT current_timestamp,

    -- Constraints
    CONSTRAINT valid_form CHECK (form_type IN ('10-K', '10-Q')),
    CONSTRAINT valid_cik  CHECK (length(cik) = 10)
);

CREATE INDEX idx_filings_ticker_date ON filings_metadata (ticker, filing_date);
CREATE INDEX idx_filings_cik ON filings_metadata (cik);
```

### 4.2 Table: `extracted_signals`

Stores the 12 structured metrics extracted by Gemini from each filing's MD&A section.

```sql
CREATE TABLE IF NOT EXISTS extracted_signals (
    signal_id                VARCHAR PRIMARY KEY,  -- '{filing_id}_v{version}'
    filing_id                VARCHAR NOT NULL REFERENCES filings_metadata(filing_id),
    extraction_model         VARCHAR NOT NULL,      -- e.g. 'gemini-2.5-flash'
    extraction_timestamp     TIMESTAMP DEFAULT current_timestamp,

    -- ── 12 Structured Metrics ──
    revenue_guidance         VARCHAR,     -- 'raise' | 'maintain' | 'lower' | 'none'
    revenue_guidance_detail  VARCHAR,     -- Free-text elaboration
    eps_guidance             VARCHAR,     -- 'raise' | 'maintain' | 'lower' | 'none'
    eps_guidance_detail      VARCHAR,
    margin_outlook           VARCHAR,     -- 'expanding' | 'stable' | 'contracting' | 'none'
    capex_direction          VARCHAR,     -- 'increasing' | 'stable' | 'decreasing' | 'none'
    management_sentiment     FLOAT,       -- [-1.0, 1.0] composite score
    forward_language_ratio   FLOAT,       -- Ratio of forward-looking to total sentences
    risk_factor_count        INTEGER,     -- Number of distinct risk factors mentioned
    key_risk_topics          VARCHAR[],   -- Array of top risk themes
    guidance_confidence      FLOAT,       -- [0.0, 1.0] model confidence in extraction
    restructuring_signals    BOOLEAN,     -- Mentions of restructuring/layoffs/impairment

    -- ── Provenance ──
    mda_char_count           INTEGER,
    prompt_tokens_used       INTEGER,
    completion_tokens_used   INTEGER
);

CREATE INDEX idx_signals_filing ON extracted_signals (filing_id);
```

### 4.3 Table: `price_series`

Stores historical daily price data for backtesting.

```sql
CREATE TABLE IF NOT EXISTS price_series (
    ticker       VARCHAR(10) NOT NULL,
    trade_date   DATE NOT NULL,
    open_price   DOUBLE,
    high_price   DOUBLE,
    low_price    DOUBLE,
    close_price  DOUBLE NOT NULL,
    adj_close    DOUBLE NOT NULL,
    volume       BIGINT,
    source       VARCHAR DEFAULT 'yfinance',
    fetched_at   TIMESTAMP DEFAULT current_timestamp,

    PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX idx_price_ticker_date ON price_series (ticker, trade_date);
```

### 4.4 Table: `macro_indicators`

Stores FRED macroeconomic series for contextual analysis.

```sql
CREATE TABLE IF NOT EXISTS macro_indicators (
    series_id    VARCHAR NOT NULL,    -- e.g. 'DFF', 'T10Y2Y', 'VIXCLS'
    obs_date     DATE NOT NULL,
    value        DOUBLE,
    fetched_at   TIMESTAMP DEFAULT current_timestamp,

    PRIMARY KEY (series_id, obs_date)
);
```

---

## 5. SEC EDGAR Ingestion Pipeline

### 5.1 CIK Handling

- CIKs are **always stored and transmitted as zero-padded 10-digit strings**.
- Normalization function: `cik.zfill(10)` applied at ingestion boundary.
- The EDGAR company search API (`efts.sec.gov`) and full-text search API (`efts.sec.gov/LATEST/search-index?q=...`) both accept padded CIKs.
- A local `ticker → CIK` mapping table is maintained in DuckDB, bootstrapped from the SEC's `company_tickers.json` endpoint.

```python
def normalize_cik(raw_cik: str | int) -> str:
    """Normalize CIK to zero-padded 10-digit string."""
    return str(raw_cik).strip().zfill(10)
```

### 5.2 Rate Limiting & Fair Access Compliance

> [!IMPORTANT]
> SEC EDGAR Fair Access policy mandates **≤ 10 requests per second** per source. This system enforces a conservative ceiling of **8 req/s** with exponential backoff on HTTP 429.

| Control | Implementation |
|---------|---------------|
| Token bucket rate limiter | `asyncio.Semaphore` + `asyncio.sleep(1/rps)` |
| Configurable ceiling | `Settings.edgar_rate_limit_rps` (default `8`) |
| Backoff on 429/503 | `tenacity.retry` with exponential backoff (max 60s) |
| User-Agent injection | Every request includes `Settings.sec_user_agent` in headers |
| Request logging | All requests logged with timestamp, URL, status code |

```python
# Simplified EDGAR client skeleton
class EdgarClient:
    """Async HTTP client for SEC EDGAR with rate-limiting."""

    def __init__(self, settings: Settings):
        self._rate_limit = settings.edgar_rate_limit_rps
        self._headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,
            follow_redirects=True,
        )
        self._semaphore = asyncio.Semaphore(self._rate_limit)

    async def _throttled_get(self, url: str) -> httpx.Response:
        async with self._semaphore:
            response = await self._client.get(url)
            await asyncio.sleep(1.0 / self._rate_limit)
            return response
```

### 5.3 Item 7 MD&A Extraction

Parsing strategy for extracting Management's Discussion & Analysis from 10-K/10-Q HTML:

1. **Streaming HTML parsing** — Use `lxml.etree.iterparse` or `BeautifulSoup` with `lxml` parser in streaming mode to avoid loading multi-MB filings entirely into memory.
2. **Section boundary detection** — Regex patterns identify Item 7 (10-K) / Item 2 (10-Q) boundaries:
   - Start: `r"(?i)item\s*7[.\s]*management.s\s*discussion"`
   - End: `r"(?i)item\s*7a[.\s]*quantitative"` or `r"(?i)item\s*8"`
3. **Text normalization** — Strip HTML tags, collapse whitespace, decode HTML entities, remove embedded tables (financial tables are noise for LLM extraction).
4. **Chunking for large filings** — If extracted MD&A exceeds 100,000 characters, split into overlapping chunks (chunk_size=80,000, overlap=2,000) for sequential Gemini processing, then merge results.
5. **Memory budget** — Peak RSS monitored; filings are processed sequentially (not concurrently loaded) to maintain a ~500 MB ceiling.

```python
def extract_mda_text(html_content: bytes, form_type: str) -> str:
    """Extract Item 7 MD&A (10-K) or Item 2 MD&A (10-Q) from filing HTML.

    Uses streaming parsing to avoid memory blowup on large filings.
    Returns cleaned plain text of the MD&A section.
    """
    # Implementation uses BeautifulSoup with lxml parser
    # Regex boundary detection for section start/end
    # Returns normalized plain text
    ...
```

---

## 6. Gemini Structured Extraction Schema

### 6.1 Pydantic v2 Schemas for Structured Output

The extraction pipeline sends MD&A text to the Gemini API and requests **structured JSON output** conforming to these schemas:

```python
"""Pydantic v2 schemas for Gemini structured extraction — 12 key metrics."""

from enum import Enum
from pydantic import BaseModel, Field


class GuidanceDirection(str, Enum):
    """Direction of management guidance revision."""
    RAISE = "raise"
    MAINTAIN = "maintain"
    LOWER = "lower"
    NONE = "none"  # No guidance provided


class MarginOutlook(str, Enum):
    EXPANDING = "expanding"
    STABLE = "stable"
    CONTRACTING = "contracting"
    NONE = "none"


class CapexDirection(str, Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    NONE = "none"


class EarningsSignals(BaseModel):
    """12 structured metrics extracted from a filing's MD&A section.

    This schema is passed to the Gemini API as a response_schema
    for structured output generation.
    """

    # ── Metric 1–2: Revenue Guidance ──
    revenue_guidance: GuidanceDirection = Field(
        description="Direction of revenue guidance revision relative to prior period."
    )
    revenue_guidance_detail: str = Field(
        default="",
        description="Verbatim or paraphrased quote supporting the revenue guidance direction."
    )

    # ── Metric 3–4: EPS Guidance ──
    eps_guidance: GuidanceDirection = Field(
        description="Direction of EPS guidance revision relative to prior period."
    )
    eps_guidance_detail: str = Field(
        default="",
        description="Verbatim or paraphrased quote supporting the EPS guidance direction."
    )

    # ── Metric 5: Margin Outlook ──
    margin_outlook: MarginOutlook = Field(
        description="Management's outlook on operating/gross margins."
    )

    # ── Metric 6: CapEx Direction ──
    capex_direction: CapexDirection = Field(
        description="Direction of capital expenditure plans."
    )

    # ── Metric 7: Management Sentiment ──
    management_sentiment: float = Field(
        ge=-1.0, le=1.0,
        description="Composite sentiment score from -1.0 (very negative) to 1.0 (very positive)."
    )

    # ── Metric 8: Forward-Looking Language Ratio ──
    forward_language_ratio: float = Field(
        ge=0.0, le=1.0,
        description="Ratio of forward-looking statements to total statements (0.0–1.0)."
    )

    # ── Metric 9: Risk Factor Count ──
    risk_factor_count: int = Field(
        ge=0,
        description="Number of distinct risk factors or headwinds mentioned in MD&A."
    )

    # ── Metric 10: Key Risk Topics ──
    key_risk_topics: list[str] = Field(
        default_factory=list,
        description="Top risk themes mentioned (e.g., 'supply chain', 'regulatory', 'FX exposure')."
    )

    # ── Metric 11: Guidance Confidence ──
    guidance_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's self-assessed confidence in the accuracy of extracted guidance (0.0–1.0)."
    )

    # ── Metric 12: Restructuring Signals ──
    restructuring_signals: bool = Field(
        description="Whether the filing mentions restructuring, layoffs, asset impairment, or write-downs."
    )
```

### 6.2 Gemini API Call Pattern

```python
from google import genai

client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_with_mda_text,
    config=genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EarningsSignals,
        temperature=0.1,  # Low temperature for factual extraction
    ),
)

signals = EarningsSignals.model_validate_json(response.text)
```

### 6.3 Prompt Engineering Strategy

The extraction prompt follows a structured template:

1. **System context** — "You are a financial analyst specializing in SEC filing analysis."
2. **Task definition** — "Extract the following 12 structured metrics from the MD&A section."
3. **Metric definitions** — Each metric described with examples of what constitutes each enum value.
4. **Input** — The cleaned MD&A text (or chunk).
5. **Output constraint** — "Respond with a JSON object conforming to the provided schema."

---

## 7. Quantitative Backtesting Logic

### 7.1 Post-Earnings Announcement Drift (PEAD)

PEAD measures the abnormal return in the 30 trading days following a filing date, capturing the market's delayed reaction to earnings information.

```python
def compute_pead(
    filing_date: date,
    ticker: str,
    price_series: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    window_days: int = 30,
) -> PEADResult:
    """Compute 30-day post-earnings announcement drift.

    Returns:
        PEADResult with raw_return, benchmark_return, abnormal_return,
        and cumulative abnormal return (CAR) series.
    """
    # 1. Find the next trading day on or after filing_date
    # 2. Get close prices for [t, t+window_days] for both ticker and benchmark
    # 3. Compute daily returns: r_t = (P_t / P_{t-1}) - 1
    # 4. Compute daily abnormal returns: AR_t = r_ticker_t - r_benchmark_t
    # 5. Compute CAR = cumsum(AR)
    # 6. Final PEAD = CAR[window_days]
    ...
```

#### PEAD Data Model

```python
class PEADResult(BaseModel):
    """Result of post-earnings announcement drift computation."""
    filing_id: str
    ticker: str
    filing_date: date
    window_days: int
    raw_return: float        # Total return over window
    benchmark_return: float  # SPY return over same window
    abnormal_return: float   # raw_return - benchmark_return
    car_series: list[float]  # Daily cumulative abnormal returns
```

### 7.2 Sharpe Ratio & Portfolio Metrics

```python
def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0,
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = (mean(R - Rf) / std(R - Rf)) * sqrt(annualization_factor)
    """
    excess_returns = returns - (risk_free_rate / annualization_factor)
    if excess_returns.std() == 0:
        return 0.0
    return float(
        (excess_returns.mean() / excess_returns.std())
        * (annualization_factor ** 0.5)
    )
```

### 7.3 Backtesting Strategy

| Signal | Long Entry | Short Entry |
|--------|-----------|-------------|
| Revenue guidance raised + positive sentiment (>0.3) | ✓ | |
| Revenue guidance lowered + negative sentiment (<-0.3) | | ✓ |
| High forward-language ratio (>0.4) + expanding margins | ✓ | |
| Restructuring signals + contracting margins | | ✓ |

- **Universe**: S&P 500 constituents with 10-K/10-Q filings in DuckDB.
- **Holding period**: 30 trading days post-filing.
- **Benchmark**: SPY total return over same window.
- **Output metrics**: Sharpe ratio, hit rate, average abnormal return, max drawdown, Calmar ratio.

---

## 8. Interactive Demo Architecture

### 8.1 System Architecture

```
┌────────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Gradio/Streamlit │────▶│   FastAPI Backend │────▶│   DuckDB    │
│   (Browser UI)     │◀────│   (REST + WS)    │◀────│   (Local)   │
└────────────────────┘     └──────┬───────────┘     └─────────────┘
                                  │
                           ┌──────┴───────────┐
                           │   Gemini API      │
                           │   (On-demand)     │
                           └──────────────────┘
```

### 8.2 FastAPI Backend (`src/earnings_intel/api/main.py`)

```python
from fastapi import FastAPI
from earnings_intel.config import get_settings

app = FastAPI(
    title="Earnings Intelligence API",
    version="0.1.0",
    docs_url="/docs",
)
```

#### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/filings` | List filings with pagination, filter by ticker/date/form |
| `GET` | `/api/filings/{filing_id}/signals` | Get extracted signals for a filing |
| `POST` | `/api/filings/{filing_id}/extract` | Trigger on-demand Gemini extraction |
| `GET` | `/api/backtests/{ticker}` | Get PEAD results for a ticker |
| `GET` | `/api/backtests/portfolio` | Portfolio-level backtest metrics |
| `GET` | `/api/health` | Health check |

### 8.3 Gradio / Streamlit UI

The UI provides three main views:

1. **Filing Explorer** — Searchable table of ingested filings with signal summaries. Click a row to view full extracted metrics and MD&A highlights.
2. **Signal Dashboard** — Time-series charts of the 12 extracted metrics across filings for a selected company. Overlaid with stock price for visual correlation.
3. **Backtest Results** — PEAD distribution histogram, cumulative return chart, Sharpe ratio heatmap by signal combination, and portfolio equity curve.

**Technology choice**: Gradio is the primary UI framework (faster prototyping, native Plotly support, direct Python integration). Streamlit is supported as an alternative via a CLI flag.

---

## 9. Implementation Roadmap

### Milestone 1: Repository Security Setup ✦ Foundation

**Goal**: Establish zero-leak security baseline and project scaffolding.

- [x] Create `.gitignore` with comprehensive exclusion patterns
- [x] Create `.env.example` with sanitized placeholders
- [x] Create `.pre-commit-config.yaml` with `detect-secrets`
- [x] Create `PROJECT_PLAN.md` (this document)
- [x] Initialize git repository and push to private GitHub remote
- [ ] Initialize `pyproject.toml` with `uv init`
- [ ] Run `uv sync --group dev` to create virtual environment
- [ ] Install pre-commit hooks: `uv run pre-commit install`
- [ ] Verify: `git check-ignore -v .env` confirms `.env` is ignored
- [ ] Implement `src/earnings_intel/config.py` with Pydantic BaseSettings

**Exit criteria**: Clean repo with working pre-commit hooks, no secrets in any committed file.

---

### Milestone 2: SEC EDGAR Ingestion Pipeline

**Goal**: Reliably ingest 10-K and 10-Q filings with Fair Access compliance.

- [ ] Implement CIK normalization and `ticker → CIK` mapping from `company_tickers.json`
- [ ] Build async EDGAR HTTP client with token-bucket rate limiter (≤8 req/s)
- [ ] Implement User-Agent header injection from `Settings.sec_user_agent`
- [ ] Build `efts.sec.gov` search query builder for 10-K/10-Q forms
- [ ] Implement streaming HTML parser for Item 7 / Item 2 MD&A extraction
- [ ] Add text normalization pipeline (strip tags, collapse whitespace, remove embedded tables)
- [ ] Implement chunking logic for filings >100K characters
- [ ] Create DuckDB schema migrations (`filings_metadata` table)
- [ ] Write integration tests with recorded HTTP fixtures (VCR-style)

**Exit criteria**: `python scripts/ingest.py --ticker AAPL --forms 10-K` ingests the last 5 years of filings into DuckDB. Memory stays below 500 MB.

---

### Milestone 3: Gemini Structured Extraction

**Goal**: Extract 12 forward-looking metrics from MD&A text via Gemini structured output.

- [ ] Implement `EarningsSignals` Pydantic schema (12 metrics)
- [ ] Build Gemini API client wrapper with retry logic and token tracking
- [ ] Design and iterate on extraction prompt template
- [ ] Implement extraction pipeline: filing → MD&A text → chunks → Gemini → merge → validate
- [ ] Create DuckDB schema for `extracted_signals` table
- [ ] Add provenance tracking (model version, token counts, extraction timestamp)
- [ ] Write unit tests with mocked Gemini responses
- [ ] Manual validation: compare extracted signals against human-labeled sample (≥20 filings)

**Exit criteria**: Extraction pipeline processes 100 filings with <5% schema validation failures. Token usage tracked per extraction.

---

### Milestone 4: Market Data & Backtesting Engine

**Goal**: Compute PEAD and portfolio-level performance metrics.

- [ ] Implement `yfinance` price fetcher with DuckDB caching (`price_series` table)
- [ ] Implement FRED API client for macro indicators (`macro_indicators` table)
- [ ] Build PEAD computation module (30-day window, SPY-benchmarked abnormal returns)
- [ ] Implement Sharpe ratio, hit rate, max drawdown, Calmar ratio calculators
- [ ] Build backtest runner with configurable signal filters
- [ ] Create backtest result storage in DuckDB
- [ ] Write unit tests for all metrics with known-answer test cases
- [ ] Validate: run backtest on ≥50 filings and produce summary statistics

**Exit criteria**: `python scripts/backtest.py` produces a portfolio report with Sharpe ratio, hit rate, and equity curve.

---

### Milestone 5: Interactive Dashboard

**Goal**: Ship a working web UI for filing exploration and backtest visualization.

- [ ] Implement FastAPI backend with all REST endpoints
- [ ] Build Gradio UI with three views (Filing Explorer, Signal Dashboard, Backtest Results)
- [ ] Add Plotly charts: PEAD distribution, cumulative returns, sentiment time series
- [ ] Implement on-demand extraction trigger from UI
- [ ] Add filing detail view with MD&A highlights and extracted signals
- [ ] Write API integration tests
- [ ] Manual QA: end-to-end workflow from ticker search → extraction → backtest → visualization

**Exit criteria**: `uvicorn earnings_intel.api.main:app` serves a working dashboard at `http://localhost:8000`.

---

### Milestone 6: Polish & Deployment Readiness

**Goal**: Production hardening, documentation, and deployment.

- [ ] Add comprehensive `README.md` with quick-start instructions
- [ ] Add `Dockerfile` with multi-stage build
- [ ] Add `docker-compose.yml` for local development
- [ ] Add GitHub Actions CI workflow (lint, test, security scan)
- [ ] Run `ruff check` and `ruff format` across entire codebase
- [ ] Run full test suite with `pytest --cov` — target ≥80% coverage
- [ ] Security audit: run `detect-secrets scan`, verify no credentials in git history
- [ ] Performance profiling: ensure EDGAR ingestion pipeline handles 500+ filings/hour
- [ ] Write `CONTRIBUTING.md` with development setup instructions
- [ ] Tag `v0.1.0` release

**Exit criteria**: Fully documented, tested, containerized application ready for deployment. Zero secrets in git history.

---

## Appendix A: Key External APIs

| API | Purpose | Rate Limit | Auth Method |
|-----|---------|-----------|-------------|
| SEC EDGAR | Filing metadata & documents | ≤10 req/s (we use 8) | User-Agent header |
| Gemini API | MD&A structured extraction | Per-project quota | API key (`SecretStr`) |
| FRED | Macroeconomic indicators | 120 req/min | API key (`SecretStr`) |
| yfinance | Historical price data | Unofficial (best-effort) | None |

## Appendix B: SEC Filing Form Types

| Form | Description | MD&A Location |
|------|-------------|--------------|
| 10-K | Annual report | Item 7 |
| 10-Q | Quarterly report | Item 2 |

## Appendix C: Key Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Credential leak | Zero-leak policy + detect-secrets + SecretStr |
| SEC rate limit ban | Conservative 8 req/s ceiling + backoff |
| Memory blowup on large filings | Streaming parser + sequential processing |
| Gemini extraction failures | Schema validation + retry with backoff |
| yfinance API instability | DuckDB price cache + graceful degradation |
