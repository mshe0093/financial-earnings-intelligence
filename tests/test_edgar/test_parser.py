"""Tests for EDGAR HTML parser — Item 7/Item 2 extraction, normalization, and chunking."""

from __future__ import annotations

import pytest

from earnings_intel.edgar.parser import (
    MDAExtractionError,
    chunk_text,
    compute_text_hash,
    extract_and_process,
    extract_mda_text,
    normalize_text,
)

SAMPLE_10K_HTML = """
<!DOCTYPE html>
<html>
<head><title>Form 10-K Apple Inc.</title></head>
<body>
<h1>ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)</h1>

<h2>Item 6. Selected Financial Data</h2>
<p>Some financial metrics from previous years appear here.</p>

<h2>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>Our company experienced significant growth in fiscal year 2023. Net sales increased by 12% driven by strong demand in our Cloud Services and AI Solutions divisions.</p>
<p>Operating margins expanded by 180 basis points due to operational efficiencies and favorable product mix. Gross profit reached $85.4 billion compared to $76.2 billion in the prior fiscal period.</p>
<table>
    <tr><th>Segment</th><th>Revenue</th></tr>
    <tr><td>Cloud</td><td>$45.2B</td></tr>
    <tr><td>Hardware</td><td>$40.2B</td></tr>
</table>
<p>We anticipate continued momentum into fiscal 2024, with full-year revenue expected to grow in the high single digits. Capital expenditures are projected to increase to $14 billion to fund data center expansions.</p>
<p>Management believes our balance sheet and liquidity position remain exceptionally strong.</p>

<h2>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</h2>
<p>We are exposed to market risk from changes in foreign currency exchange rates and interest rates.</p>

<h2>Item 8. Financial Statements and Supplementary Data</h2>
<p>Consolidated financial statements follow below.</p>
</body>
</html>
"""

SAMPLE_10Q_HTML = """
<!DOCTYPE html>
<html>
<body>
<h2>Item 1. Financial Statements</h2>
<p>Unaudited balance sheet...</p>

<h2>Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations</h2>
<p>During the three months ended June 30, revenue contracted 3% primarily reflecting foreign exchange headwinds and macroeconomic softness in European consumer markets.</p>
<p>Operating expenses decreased 5% as a result of recent restructuring actions and reduced headcount. We expect margins to stabilize in the second half of the fiscal year.</p>

<h2>Item 3. Quantitative and Qualitative Disclosures About Market Risk</h2>
<p>Interest rate risk...</p>
</body>
</html>
"""


class TestMdaExtraction:
    """Verify section extraction from HTML."""

    def test_extract_mda_10k_success(self) -> None:
        text = extract_mda_text(SAMPLE_10K_HTML, form_type="10-K")
        assert "Management's Discussion" in text or "significant growth" in text
        assert "Cloud Services" in text
        assert "85.4 billion" in text
        # Content from Item 6 or Item 7A should NOT be included
        assert "Selected Financial Data" not in text
        assert "Quantitative and Qualitative Disclosures" not in text

    def test_extract_mda_10q_success(self) -> None:
        text = extract_mda_text(SAMPLE_10Q_HTML, form_type="10-Q")
        assert "foreign exchange headwinds" in text
        assert "restructuring actions" in text
        assert "Financial Statements" not in text

    def test_tables_removed_from_output(self) -> None:
        text = extract_mda_text(SAMPLE_10K_HTML, form_type="10-K")
        # Table column header and row content should be stripped
        assert "Segment" not in text
        assert "$45.2B" not in text

    def test_bytes_input_handled(self) -> None:
        text = extract_mda_text(SAMPLE_10K_HTML.encode("utf-8"), form_type="10-K")
        assert len(text) > 100

    def test_invalid_form_type_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_mda_text(SAMPLE_10K_HTML, form_type="8-K")

    def test_missing_mda_raises_error(self) -> None:
        html_without_mda = "<html><body><p>Just some random text</p></body></html>"
        with pytest.raises(MDAExtractionError):
            extract_mda_text(html_without_mda, form_type="10-K")


class TestTextNormalization:
    """Verify text normalization functions."""

    def test_normalize_collapses_whitespace(self) -> None:
        raw = "Hello   \n\n\n   world!   This  is   a   test."
        norm = normalize_text(raw)
        assert "   " not in norm
        assert norm == "Hello\n\nworld! This is a test."

    def test_normalize_decodes_entities(self) -> None:
        raw = "Revenue &amp; profits &gt; $50M &euro;50M"
        norm = normalize_text(raw)
        assert "&amp;" not in norm
        assert "Revenue & profits > $50M €50M" == norm

    def test_compute_text_hash(self) -> None:
        h1 = compute_text_hash("test text")
        h2 = compute_text_hash("test text")
        h3 = compute_text_hash("different text")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA-256 hex string


class TestChunking:
    """Verify text chunking logic."""

    def test_short_text_not_chunked(self) -> None:
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunked_with_overlap(self) -> None:
        # Create a ~5000 char text composed of sentences
        sentences = [
            f"Sentence number {i} describes important business conditions." for i in range(100)
        ]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=1000, overlap=100)
        assert len(chunks) > 1
        # Every chunk should be non-empty
        assert all(len(c) > 0 for c in chunks)


class TestFullPipeline:
    """Verify extract_and_process orchestration."""

    def test_extract_and_process_returns_mda_result(self) -> None:
        result = extract_and_process(SAMPLE_10K_HTML, form_type="10-K")
        assert result.char_count > 200
        assert len(result.text_hash) == 64
        assert len(result.chunks) >= 1
        assert result.form_type == "10-K"
        assert not result.is_chunked  # Under 80k chars
