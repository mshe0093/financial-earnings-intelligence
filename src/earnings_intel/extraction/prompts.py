"""Prompt construction and templates for Gemini structured extraction of SEC MD&A text."""

from __future__ import annotations

SYSTEM_INSTRUCTION = """\
You are an expert equity research analyst and quantitative financial modeler specializing in
SEC 10-K and 10-Q filing analysis.

Your task is to analyze the provided Management's Discussion and Analysis (MD&A) excerpt
from an SEC filing and extract 12 forward-looking guidance and risk signals with high precision.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Analyze the following MD&A section from an SEC filing. Extract the 12 forward-looking signals
adhering strictly to these guidelines:

1. revenue_guidance:
   - "raise": Management explicitly raises revenue forecast or describes demand tracking above.
   - "maintain": Guidance reiterated or reaffirmed within prior ranges.
   - "lower": Revenue targets reduced, decelerating growth warnings, or downward revisions.
   - "none": No forward revenue guidance is provided.

2. revenue_guidance_detail: Quote or succinct summary of statement supporting revenue guidance.

3. eps_guidance:
   - "raise": Management raises EPS or profitability targets.
   - "maintain": EPS range confirmed.
   - "lower": EPS guidance cut or margin squeeze leading to profit revision.
   - "none": No forward EPS guidance provided.

4. eps_guidance_detail: Quote or succinct summary of statement supporting EPS guidance.

5. margin_outlook:
   - "expanding": Clear expectations of gross/operating margin growth.
   - "stable": Margins projected roughly flat.
   - "contracting": Cost pressures, wage inflation, or pricing headwinds compressing margins.
   - "none": No explicit margin trend discussed.

6. capex_direction:
   - "increasing": Capital expenditures or investment spending planned to rise.
   - "stable": CapEx expected in line with prior years.
   - "decreasing": CapEx cuts, project deferrals, or cash preservation.
   - "none": CapEx outlook omitted.

7. management_sentiment: Score from -1.0 (pessimistic) to 1.0 (bullish). 0.0 is neutral.

8. forward_language_ratio: Float from 0.0 to 1.0 estimating proportion of forward statements
   ("we expect", "we anticipate", "plans to", "forecasts") relative to historical discussion.

9. risk_factor_count: Integer count of distinct macroeconomic, competitive, regulatory, or
   operational risks cited in this text.

10. key_risk_topics: List of top 3 to 7 concise risk keywords/themes (e.g., ["FX volatility"]).

11. guidance_confidence: Float from 0.0 to 1.0 rating confidence in clarity/specificity
    of forward-looking guidance.

12. restructuring_signals: Boolean (true if mentioning layoffs, facility closures, severance,
    or asset impairment/write-downs; otherwise false).

---
MD&A EXCERPT:
{mda_text}
---

Provide your extraction matching the requested JSON schema.
"""


def build_extraction_prompt(mda_text: str) -> str:
    """Build the complete extraction prompt for a given MD&A text chunk.

    Args:
        mda_text: Cleaned text from the MD&A section.

    Returns:
        Formatted prompt string.
    """
    return EXTRACTION_PROMPT_TEMPLATE.format(mda_text=mda_text.strip())
