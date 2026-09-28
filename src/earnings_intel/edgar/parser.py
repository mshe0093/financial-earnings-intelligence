"""Extract Item 7 MD&A text from 10-K/10-Q filing HTML documents.

Parsing strategy:
1. BeautifulSoup with lxml parser for fast and forgiving HTML parsing
2. Table removal: financial tables are stripped (noise for LLM sentiment extraction)
3. Regex boundary detection:
   - 10-K: Item 7 (start) -> Item 7A or Item 8 (end)
   - 10-Q: Item 2 (start) -> Item 3 or Item 4 (end)
4. Text normalization: decode HTML entities, collapse whitespace, strip line numbers
5. Chunking for large filings (>80K characters) with sentence boundary awareness
6. SHA-256 text hashing for cache validation
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class MDAExtractionError(Exception):
    """Raised when Item 7/Item 2 MD&A section cannot be extracted from filing."""


# ── Regex Patterns for Section Boundaries ──────────────────────────────────

# 10-K: Item 7 starts MD&A, Item 7A or Item 8 ends it
_10K_START_PATTERNS = [
    re.compile(
        r"(?i)item\s*7[.:\s\-\u2013\u2014]+(?:management['\u2019]s\s+discussion|m\s*d\s*&\s*a)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*7\b[.:\s\-\u2013\u2014]", re.DOTALL),
]

_10K_END_PATTERNS = [
    re.compile(
        r"(?i)item\s*7a[.:\s\-\u2013\u2014]+(?:quantitative|market\s+risk)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*7a\b", re.DOTALL),
    re.compile(
        r"(?i)item\s*8[.:\s\-\u2013\u2014]+(?:financial\s+statements)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*8\b", re.DOTALL),
]

# 10-Q: Item 2 starts MD&A, Item 3 or Item 4 ends it
_10Q_START_PATTERNS = [
    re.compile(
        r"(?i)item\s*2[.:\s\-\u2013\u2014]+(?:management['\u2019]s\s+discussion|m\s*d\s*&\s*a)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*2\b[.:\s\-\u2013\u2014]", re.DOTALL),
]

_10Q_END_PATTERNS = [
    re.compile(
        r"(?i)item\s*3[.:\s\-\u2013\u2014]+(?:quantitative|market\s+risk)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*3\b", re.DOTALL),
    re.compile(
        r"(?i)item\s*4[.:\s\-\u2013\u2014]+(?:controls\s+and\s+procedures)",
        re.DOTALL,
    ),
    re.compile(r"(?i)item\s*4\b", re.DOTALL),
]


# ── Output Data Model ──────────────────────────────────────────────────────


@dataclass
class MDAResult:
    """Extracted and processed MD&A section result."""

    text: str
    char_count: int
    text_hash: str
    chunks: list[str] = field(default_factory=list)
    form_type: str = "10-K"

    @property
    def is_chunked(self) -> bool:
        """True if the document was split into multiple chunks."""
        return len(self.chunks) > 1


# ── Core Functions ─────────────────────────────────────────────────────────


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for caching and deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def remove_tables(soup: BeautifulSoup) -> None:
    """Strip all <table> elements in-place to remove financial tables noise."""
    for table in soup.find_all("table"):
        table.decompose()


def normalize_text(raw_text: str) -> str:
    """Clean and normalize extracted filing text.

    Steps:
    1. Unescape HTML entities (&nbsp;, &#160;, &amp;, etc.)
    2. Replace non-breaking spaces and special whitespace with standard space
    3. Remove page numbers / isolated numbers left over from tables
    4. Collapse runs of whitespace into single spaces
    5. Strip leading/trailing whitespace
    """
    text = html.unescape(raw_text)
    # Replace various Unicode spaces and non-breaking spaces
    text = re.sub(r"[\xa0\u2000-\u200b\u202f\u205f\u3000]", " ", text)
    # Remove lines containing only digits/punctuation (page numbers, table scraps)
    text = re.sub(r"(?m)^\s*[\d\s\.,\-\$%\(\)]+\s*$", "", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Strip spaces around newlines
    text = re.sub(r" ?\n ?", "\n", text)
    # Collapse multiple newlines into double-newline (paragraph break)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_mda_text(
    html_content: bytes | str,
    form_type: str = "10-K",
) -> str:
    """Extract Item 7 (10-K) or Item 2 (10-Q) MD&A text from filing HTML.

    Args:
        html_content: Raw HTML document bytes or string.
        form_type: "10-K" or "10-Q".

    Returns:
        Cleaned, normalized plain text of the MD&A section.

    Raises:
        MDAExtractionError: If MD&A boundaries could not be located.
    """
    if isinstance(html_content, bytes):
        # Decode with fallback for common financial report encodings
        try:
            html_str = html_content.decode("utf-8")
        except UnicodeDecodeError:
            html_str = html_content.decode("latin-1", errors="replace")
    else:
        html_str = html_content

    # Fast check: does the HTML contain Item 7/2 mentions at all?
    norm_form = form_type.strip().upper()
    if norm_form == "10-K":
        start_patterns = _10K_START_PATTERNS
        end_patterns = _10K_END_PATTERNS
    elif norm_form == "10-Q":
        start_patterns = _10Q_START_PATTERNS
        end_patterns = _10Q_END_PATTERNS
    else:
        raise ValueError(f"Unsupported form_type: '{form_type}'")

    # Parse with BeautifulSoup + lxml
    soup = BeautifulSoup(html_str, "lxml")

    # Remove non-content elements before text extraction
    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()

    # Strip tables — financial statements and tabular data add massive noise
    remove_tables(soup)

    # Extract all text preserving paragraph separation
    full_text = soup.get_text(separator="\n\n")

    # Locate section start
    start_pos: int | None = None
    for pattern in start_patterns:
        matches = list(pattern.finditer(full_text))
        if matches:
            # If multiple matches (e.g. Table of Contents entry vs actual section),
            # pick the match that is furthest in or has substantial body text after it.
            # TOC entries typically appear in the first 15% of the document.
            if len(matches) > 1:
                # Discard TOC matches in first 10,000 chars if later matches exist
                deep_matches = [m for m in matches if m.start() > 10_000]
                match = deep_matches[0] if deep_matches else matches[-1]
            else:
                match = matches[0]
            start_pos = match.start()
            break

    if start_pos is None:
        raise MDAExtractionError(
            f"Could not locate Item {7 if norm_form == '10-K' else 2} start boundary "
            f"in {form_type} filing."
        )

    # Search for end boundary starting from after the start position
    text_after_start = full_text[start_pos:]
    end_pos: int | None = None

    for pattern in end_patterns:
        match = pattern.search(text_after_start)
        if match and match.start() > 500:  # Must have at least 500 chars of MD&A
            end_pos = start_pos + match.start()
            break

    if end_pos is not None:
        mda_raw = full_text[start_pos:end_pos]
    else:
        # If no explicit end boundary found, take up to 150,000 characters
        logger.warning("End boundary for MD&A not found; capturing next 150k characters.")
        mda_raw = full_text[start_pos : start_pos + 150_000]

    cleaned = normalize_text(mda_raw)

    if len(cleaned) < 200:
        raise MDAExtractionError(
            f"Extracted MD&A section is suspiciously short ({len(cleaned)} chars)."
        )

    return cleaned


def chunk_text(
    text: str,
    chunk_size: int = 80_000,
    overlap: int = 2_000,
) -> list[str]:
    """Split long text into overlapping chunks, snapping to sentence boundaries.

    Args:
        text: The normalized text to split.
        chunk_size: Target character count per chunk.
        overlap: Overlap in characters between adjacent chunks.

    Returns:
        List of text chunks. Returns ``[text]`` if under ``chunk_size``.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Look for sentence boundary (. followed by space or newline) in overlap zone
        search_zone = text[end - overlap : end]
        sentence_end = search_zone.rfind(". ")
        if sentence_end != -1:
            split_point = end - overlap + sentence_end + 2
        else:
            # Fall back to paragraph break
            para_end = search_zone.rfind("\n\n")
            if para_end != -1:
                split_point = end - overlap + para_end + 2
            else:
                # Hard split at chunk_size
                split_point = end

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, maintaining overlap
        start = split_point - overlap
        if start <= 0 or start >= len(text):
            break

    return chunks


def extract_and_process(
    html_content: bytes | str,
    form_type: str = "10-K",
    chunk_size: int = 80_000,
    overlap: int = 2_000,
) -> MDAResult:
    """Full extraction pipeline: HTML -> clean MD&A -> chunked result.

    Args:
        html_content: Raw filing HTML.
        form_type: "10-K" or "10-Q".
        chunk_size: Max characters per chunk.
        overlap: Character overlap between chunks.

    Returns:
        ``MDAResult`` containing the extracted text, chunks, and metadata.
    """
    text = extract_mda_text(html_content, form_type)
    text_hash = compute_text_hash(text)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return MDAResult(
        text=text,
        char_count=len(text),
        text_hash=text_hash,
        chunks=chunks,
        form_type=form_type,
    )
